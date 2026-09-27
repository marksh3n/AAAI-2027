# Modal wrapper for the r(t) ceiling run. Every step runs one of the src/recovery scripts
# in a container with the "e0-recovery" Volume at /vol:
#   /vol/inputs   traces_B.jsonl, prefixes_ceiling.jsonl   (scripts/modal_upload.sh)
#   /vol/outputs  rollouts, temperature check, smoke files
#   /vol/hf       HF_HOME, so the model downloads once
# the scripts write straight to the Volume; a thread commits it every 2 min and once more
# at the end. Modal restarts a preempted call on the same input, so a restart gets the
# same arguments (same --ledger-key) and resumes: done prefixes are skipped and the spend
# of the lost attempt stays counted. commands: RUN_MODAL.md
import os, subprocess, sys, threading, time
from pathlib import Path

import modal

NAME = "e0-recovery"
ROOT = Path(__file__).resolve().parent.parent
VOL = "/vol"
INPUTS, OUTPUTS = f"{VOL}/inputs", f"{VOL}/outputs"
PREFIXES = f"{INPUTS}/prefixes_ceiling.jsonl"
ROLLOUTS = f"{OUTPUTS}/rollouts.jsonl"
TEMP = 0.7  # tag-B sampling temperature; check_temp.py gates it
COMMIT_S = 120
DAY = 24 * 3600

app = modal.App(NAME)
vol = modal.Volume.from_name(NAME, create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    # vllm 0.10.2 leaves transformers unpinned, and transformers 5 both breaks it and loads
    # this model's tokenizer as Qwen2Tokenizer, which drops the BOS that tag-B's string
    # prompts got (two BOS). 4.56.2 is vllm 0.10.2's contemporary. generate.py imports
    # datasets and math_verify
    .uv_pip_install("vllm==0.10.2", "transformers==4.56.2", "datasets", "math-verify>=0.5")
    .env({"HF_HOME": f"{VOL}/hf", "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_dir(ROOT / "src", "/root/src", ignore=["**/__pycache__", "**/*.pyc"])
)
SCRIPTS = {"rollouts": "run_rollouts.py", "check_temp": "check_temp.py",
           "build_prefixes": "build_prefixes.py"}


def _work(script, argv):
    # run one script with the Volume committed in the background; its exit code is the result
    stop = threading.Event()

    def commit_loop():
        while not stop.wait(COMMIT_S):
            try:
                vol.commit()
            except Exception as e:  # a failed commit is retried on the next tick
                print(f"[modal] volume commit failed: {e}", flush=True)

    os.makedirs(INPUTS, exist_ok=True)
    os.makedirs(OUTPUTS, exist_ok=True)
    threading.Thread(target=commit_loop, daemon=True).start()
    cmd = [sys.executable, f"/root/src/recovery/{SCRIPTS[script]}", *argv]
    print("[modal] " + " ".join(cmd), flush=True)
    try:
        code = subprocess.call(cmd, cwd="/root")
    finally:
        stop.set()
        vol.commit()
        print("[modal] volume committed", flush=True)
    if code:
        raise RuntimeError(f"{SCRIPTS[script]} exited with {code}")


common = dict(image=image, volumes={VOL: vol}, timeout=DAY)


# one function per GPU type: a plain choice at launch, no dependence on with_options
@app.function(gpu="H100", **common)
def on_h100(script: str, argv: list):
    _work(script, argv)


@app.function(gpu="H200", **common)
def on_h200(script: str, argv: list):
    _work(script, argv)


@app.function(gpu="T4", **common)
def on_t4(script: str, argv: list):
    _work(script, argv)


@app.function(cpu=4, memory=8192, **common)
def on_cpu(script: str, argv: list):
    _work(script, argv)


GPU_FNS = {"h100": on_h100, "h200": on_h200, "t4": on_t4}


@app.local_entrypoint()
def main(step: str, gpu: str = "", minutes: float = 0, budget: float = 0, extra: str = ""):
    """step: build | temp | smoke | bench | run | status. extra: more args for the script."""
    more = extra.split()
    rollouts = ["--prefix-file", PREFIXES, "--temp", str(TEMP)]
    if step == "build":
        fn, script = on_cpu, "build_prefixes"
        argv = ["--design", "ceiling", "--data-dir", INPUTS, "--out", PREFIXES]
    elif step == "temp":
        fn, script = GPU_FNS[gpu or "t4"], "check_temp"
        argv = ["--traces", f"{INPUTS}/traces_B.jsonl", "--prefix-file", PREFIXES,
                "--out", f"{OUTPUTS}/temp_check.json"]
    elif step == "smoke":
        gpu = gpu or "t4"
        fn, script = GPU_FNS[gpu], "rollouts"
        argv = rollouts + ["--smoke", "--gpu", gpu, "--out", f"{OUTPUTS}/smoke.jsonl"]
    elif step in ("bench", "run"):
        gpu = gpu or "h100"
        if step == "bench" and not minutes:
            minutes = 5
        if step == "run" and not budget:
            raise SystemExit("--budget is required for a full run (the credit left, in $)")
        fn, script = GPU_FNS[gpu], "rollouts"
        # benchmark rows are real rows under the same config, so they go to the same file
        # and the full run skips them
        argv = rollouts + ["--gpu", gpu, "--out", ROLLOUTS,
                           "--ledger-key", f"{step}-{gpu}-{time.strftime('%Y%m%d-%H%M%S')}"]
        if minutes:
            argv += ["--max-minutes", str(minutes)]
        if budget:
            argv += ["--budget-usd", str(budget)]
    elif step == "status":
        fn, script = on_cpu, "rollouts"
        argv = ["--prefix-file", PREFIXES, "--out", ROLLOUTS, "--status"]
    else:
        raise SystemExit(f"unknown step {step!r}")
    fn.remote(script, argv + more)
