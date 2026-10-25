# the development dry run end to end on synthetic files shaped like the real ones: traces_B.jsonl,
# meta_B / acts_B at 2048 / 4096 / 6144, and ceiling rollouts (t0, p512, p2048, p4096). runs
# fit_monitors.py -> dev_bundle.py -> analyze.py as scripts, the way the Colab cells do.
# plain script, CPU only, exits non-zero on failure.
import copy, json, os, subprocess, sys, tempfile
import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src", "probe_cmp"))
import data as D  # noqa: E402

fails = []


def check(name, cond, extra=None):
    print(f"{'ok  ' if cond else 'FAIL'} {name}{(' -- ' + str(extra)) if extra is not None else ''}")
    if not cond:
        fails.append(name)


rng = np.random.default_rng(0)
d = tempfile.mkdtemp()
DIM, LAYERS = 32, [-1, 0, 4, 8, 12, 16, 20, 24, 27]
POS = [2048, 4096, 6144]
traces, rows, roll = [], [], []
for p in range(40):
    base = rng.uniform(0.3, 0.7)
    for k in range(6):
        tid = f"{p}_{k}"
        n = int(rng.uniform(1500, 9000))
        trunc = p == 0 and k == 0
        ok = bool(rng.random() < base) and not trunc
        body = "".join(f"w{rng.integers(0, 50)} " + ("\n\n" if rng.random() < 0.05 else "") for _ in range(n))
        text = body + ("" if trunc else "</think> The answer is \\boxed{" + ("7" if ok else "9") + "}")
        traces.append({"trace_id": tid, "problem_id": p, "problem": "q", "text": text, "n_gen": 32768 if trunc else n,
                       "trunc": trunc, "is_correct": ok, "answer": None if trunc else ("7" if ok else "9")})
        for pos in POS:
            if pos <= n:
                rows.append((tid, p, pos, ok, 32768 if trunc else n, body[: pos * 3]))
        if n >= 4096 and not trunc:  # the ceiling cohort: reaches 4096 and finishes
            for ck in ("p512", "p2048", "p4096"):
                r = 0.1 if not ok else 0.8
                roll.append({"prefix_id": f"ceil/{tid}/{ck}", "round": 0, "problem_id": p, "trace_id": tid, "checkpoint": ck,
                             "samples": [{"sample_idx": i, "correct": bool(rng.random() < r), "n_gen": int(rng.uniform(500, 4000)),
                                          "truncated": False, "text": "x </think>", "answer": "7"} for i in range(8)]})
    roll.append({"prefix_id": f"ceil/{p}/t0", "round": 0, "problem_id": p, "trace_id": None, "checkpoint": "t0",
                 "samples": [{"sample_idx": i, "correct": bool(c), "n_gen": int(rng.uniform(2000, 8000)), "truncated": False,
                              "text": "y </think>", "answer": "7" if c else "9"}
                             for i, c in enumerate(rng.random(8) < base)]})
with open(os.path.join(d, "traces_B.jsonl"), "w") as f:
    for t in traces:
        f.write(json.dumps(t) + "\n")
mb = pd.DataFrame(rows, columns=["trace_id", "problem_id", "pos", "is_correct", "n_gen", "prefix"])
mb["row"] = np.arange(len(mb))
mb.to_parquet(os.path.join(d, "meta_B.parquet"))
acts = rng.normal(size=(len(mb), len(LAYERS), DIM)).astype(np.float16)
acts[:, LAYERS.index(20), 0] += 1.5 * (~mb.is_correct.values).astype(float)
np.save(os.path.join(d, "acts_B.npy"), acts)
reg = copy.deepcopy(D.load_registry())
reg["hidden_dim"] = DIM
reg["sources"] = [s for s in reg["sources"] if s["name"] == "B_fixed"]
regp = os.path.join(d, "registry.json")
json.dump(reg, open(regp, "w"))
rollp = os.path.join(d, "recovery_ceiling")
with open(rollp, "w") as f:
    f.write('{"torn header"\n')
    for r in roll:
        f.write(json.dumps(r) + "\n")

py = sys.executable
mon, bun, out = (os.path.join(d, x) for x in ("monitors", "bundle", "dryrun"))
steps = [
    [py, "src/confirm/fit_monitors.py", "--data", d, "--registry", regp, "--out", mon],
    [py, "src/confirm/dev_bundle.py", "--data", d, "--registry", regp, "--rollouts", rollp, "--monitors", mon, "--out", bun],
    [py, "src/confirm/analyze.py", "--bundle", bun, "--out", out, "--n-boot", "200"],
    [py, "src/confirm/provenance_6144.py", "--data", d, "--registry", regp, "--out", os.path.join(d, "prov")],
]
for cmd in steps:
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    check(f"{os.path.basename(cmd[1])} runs", r.returncode == 0, (r.stdout + r.stderr)[-1500:] if r.returncode else None)
    if r.returncode:
        break
else:
    man = json.load(open(os.path.join(mon, "manifest.json")))
    check("a monitor per (name, checkpoint) in the spec", set(man["monitors"]) == {
        "lr_raw_L20@2048", "lr_raw_L20@4096", "lr_raw_L20@6144", "hgb_topk50_L20@6144",
        "len@2048", "len@4096", "len@6144", "tfidf@2048", "tfidf@4096", "tfidf@6144"}, sorted(man["monitors"]))
    check("manifest binds inputs and code by sha256", all(len(v["sha256"]) == 64 for v in man["inputs"].values())
          and "analyze.py" in man["code"])
    meta = json.load(open(os.path.join(bun, "bundle.json")))
    check("dev bundle: banks at 512, 2048, 4096 (+ t0), none at 6144", meta["bank_checkpoints"] == [0, 512, 2048, 4096], meta["bank_checkpoints"])
    src = pd.read_parquet(os.path.join(bun, "sources.parquet"))
    check("dev bundle: every trace of the rolled-out problems, truncated ones included",
          len(src) == 240 and src[src.trace_id == "0_0"].correct.iloc[0] == 0)
    pol = pd.read_csv(os.path.join(out, "policies.csv"))
    check("policies at 2048 and 4096 only", set(pol.checkpoint) == {2048, 4096}, set(pol.checkpoint))
    auc = pd.read_csv(os.path.join(out, "auroc.csv"))
    check("AUROC at 6144 for the tree on live pre-answer traces",
          ((auc.checkpoint == 6144) & (auc.score == "hgb_topk50_L20") & (auc.population == "live, pre-answer")).any())
    md = open(os.path.join(out, "report.md")).read()
    check("report is labelled a development dry run", "DEVELOPMENT DRY RUN" in md)
    prov = pd.read_csv(os.path.join(d, "prov", "auroc_by_seed.csv"))
    check("provenance: tree and baselines per fold seed on the 6144 cohort",
          {"hgb/topk50", "lr/len", "lr/tfidf", "lr/raw"} <= set(prov.arm) and {"0", "1", "2"} <= set(prov.seed.astype(str)))
    tests = json.load(open(os.path.join(out, "tests.json")))
    check("primary contrasts computed on the dev bundle", tests["primary vs random"] is not None)

print(f"\n{'all passed' if not fails else f'{len(fails)} FAILED: {fails}'}")
sys.exit(1 if fails else 0)
