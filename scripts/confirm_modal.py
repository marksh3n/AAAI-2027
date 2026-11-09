# Modal wrapper for the confirmation data run; usage and order in docs/RUN_CONFIRM_GPU.md. Same
# pattern as modal_app.py, its own app and Volume ("e0-confirm"). With E0_MODEL set (spec.PROFILES) it is
# a replication run on that model instead: app and Volume "e0-<model>", cut at spec.Q1_CUT only, the bank
# on the mixed-problem cohort only, no t0 bank, smoke on an L4 (bf16):
#   /vol/inputs   problems.jsonl (the frozen problem list)
#   /vol/outputs  t0.jsonl, sources.jsonl, feats/, cuts.jsonl, bank.jsonl (+ .config.json / ledgers)
#   /vol/hf       HF_HOME
# every GPU step resumes: run_rollouts skips (prefix, round) rows already written.
import os, subprocess, sys, threading, time
from pathlib import Path

import modal

NAME = f"e0-{spec.MODEL_KEY.replace('.', '-')}" if spec.REPLICATION else "e0-confirm"
ROOT = Path(__file__).resolve().parent.parent
VOL = "/vol"
INP, OUT = f"{VOL}/inputs", f"{VOL}/outputs"
COMMIT_S = 120
DAY = 24 * 3600

# locally the repo's src/confirm; inside the container the same files, mounted at /root/src
for p in (ROOT / "src" / "confirm", Path("/root/src/confirm")):
    if (p / "spec.py").exists():
        sys.path.insert(0, str(p))
        break
import spec  # noqa: E402

app = modal.App(NAME)
vol = modal.Volume.from_name(NAME, create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    # the pins are spec.VLLM_VERSION / spec.TRANSFORMERS_VERSION; pandas / pyarrow for feats.parquet
    .uv_pip_install(f"vllm=={spec.VLLM_VERSION}", f"transformers=={spec.TRANSFORMERS_VERSION}", "datasets",
                    "math-verify>=0.5", "pandas", "pyarrow", "scikit-learn", "joblib")
    .env({"HF_HOME": f"{VOL}/hf", "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false",
          "E0_MODEL": spec.MODEL_KEY})
    .add_local_dir(ROOT / "src", "/root/src", ignore=["**/__pycache__", "**/*.pyc"])
)
RUNNER = "/root/src/recovery/run_rollouts.py"
PREP = "/root/src/confirm/gpu_prep.py"
SAMPLING = ["--temp", str(spec.TEMPERATURE), "--top-p", str(spec.TOP_P), "--revision", spec.MODEL_REVISION,
            "--model", spec.MODEL, "--dtype", spec.DTYPE]


def _work(cmds):
    stop = threading.Event()

    def commit_loop():
        while not stop.wait(COMMIT_S):
            try:
                vol.commit()
            except Exception as e:
                print(f"[modal] volume commit failed: {e}", flush=True)

    os.makedirs(INP, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)
    threading.Thread(target=commit_loop, daemon=True).start()
    try:
        for cmd in cmds:
            cmd = [sys.executable, *cmd]
            print("[modal] " + " ".join(cmd), flush=True)
            code = subprocess.call(cmd, cwd="/root")
            if code:
                raise RuntimeError(f"{' '.join(cmd[1:3])} exited with {code}")
    finally:
        stop.set()
        vol.commit()
        print("[modal] volume committed", flush=True)


common = dict(image=image, volumes={VOL: vol}, timeout=DAY)


@app.function(gpu="H100", **common)
def on_h100(cmds: list):
    _work(cmds)


@app.function(gpu="H200", **common)
def on_h200(cmds: list):
    _work(cmds)


@app.function(gpu="T4", **common)
def on_t4(cmds: list):
    _work(cmds)


@app.function(gpu="L4", **common)
def on_l4(cmds: list):
    _work(cmds)


@app.function(cpu=4, memory=16384, **common)
def on_cpu(cmds: list):
    _work(cmds)


GPU_FNS = {"h100": on_h100, "h200": on_h200, "t4": on_t4, "l4": on_l4}


def runner(prefix_file, out, rnd, n, ids, gpu, budget, key, extra=()):
    argv = [RUNNER, "--prefix-file", prefix_file, "--out", out, "--round", str(rnd), "--n", str(n),
            "--save-token-ids", ids, "--gpu", gpu, "--ledger-key", key, *SAMPLING, *extra]
    return argv + (["--budget-usd", str(budget)] if budget else [])


@app.local_entrypoint()
def main(step: str, gpu: str = "h100", budget: float = 0, samples: int = 0):
    """step: smoke | t0 | sources | extract | cuts | bank | status. samples: source traces per problem."""
    key = f"{step}-{gpu}-{time.strftime('%Y%m%d-%H%M%S')}"
    t0, src, cuts, bank, feats = (f"{OUT}/t0.jsonl", f"{OUT}/sources.jsonl", f"{OUT}/cuts.jsonl",
                                  f"{OUT}/bank.jsonl", f"{OUT}/feats")
    print(f"[e0] model {spec.MODEL} @ {spec.MODEL_REVISION[:8]}, app/volume {NAME}"
          + (" (replication)" if spec.REPLICATION else ""), flush=True)
    if step == "smoke":
        # the whole chain, small, on a T4 (an L4 for bf16 models): 2 bundled problems, 2 sources each capped
        # at 300 tokens, checkpoints 64 / 128, 2 continuations per prefix capped at 64; ends with gpu_prep
        # check --recompute
        s = f"{OUT}/smoke"
        cap = ["--max-tokens-cap"]
        sg = "l4" if spec.DTYPE == "bfloat16" else "t4"
        GPU_FNS[sg].remote([
            [PREP, "t0", "--problems", "/root/src/confirm/smoke_problems.jsonl", "--out", f"{s}/t0.jsonl", "--smoke"],
            runner(f"{s}/t0.jsonl", f"{s}/sources.jsonl", 0, 2, "full", sg, 0, key, cap + ["300"]),
            [PREP, "extract", "--sources", f"{s}/sources.jsonl", "--t0", f"{s}/t0.jsonl", "--out", f"{s}/feats",
             "--checkpoints", "64,128", "--smoke"],
            [PREP, "cuts", "--sources", f"{s}/sources.jsonl", "--t0", f"{s}/t0.jsonl", "--out", f"{s}/cuts.jsonl",
             "--checkpoints", "64,128", "--smoke"],
            runner(f"{s}/t0.jsonl", f"{s}/bank.jsonl", 1, 2, "sha", sg, 0, key, cap + ["64"]),
            runner(f"{s}/cuts.jsonl", f"{s}/bank.jsonl", 0, 2, "sha", sg, 0, key, cap + ["64"]),
            [PREP, "check", "--sources", f"{s}/sources.jsonl", "--t0", f"{s}/t0.jsonl", "--feats", f"{s}/feats",
             "--bank", f"{s}/bank.jsonl", "--cuts", f"{s}/cuts.jsonl", "--smoke", "--recompute"],
        ])
        return
    if step == "t0":
        on_cpu.remote([[PREP, "t0", "--problems", f"{INP}/problems.jsonl", "--out", t0]])
    elif step == "sources":
        n = samples or spec.SLOTS["samples_per_problem"]
        if not n:
            raise SystemExit("--samples is required until spec.SLOTS['samples_per_problem'] is frozen")
        if not budget:
            raise SystemExit("--budget is required (the Modal credit left, in $)")
        GPU_FNS[gpu].remote([runner(t0, src, 0, n, "full", gpu, budget, key)])
    elif step == "extract":
        GPU_FNS[gpu].remote([[PREP, "extract", "--sources", src, "--t0", t0, "--out", feats]])
    elif step == "cuts":
        on_cpu.remote([[PREP, "cuts", "--sources", src, "--t0", t0, "--out", cuts]])
    elif step == "bank":
        if not budget:
            raise SystemExit("--budget is required (the Modal credit left, in $)")
        # t0 bank first (round 1: round 0 of these records is the sources), then every live cut.
        # a replication run has no t0 bank: its cuts only
        first = [] if spec.REPLICATION else [runner(t0, bank, 1, spec.K_T0, "sha", gpu, budget, key)]
        GPU_FNS[gpu].remote(first + [runner(cuts, bank, 0, spec.K_PREFIX, "sha", gpu, budget, key)])
    elif step == "status":
        on_cpu.remote([[RUNNER, "--prefix-file", t0, "--out", src, "--status"],
                       [RUNNER, "--prefix-file", cuts, "--out", bank, "--status"]])
    else:
        raise SystemExit(f"unknown step {step!r}")
