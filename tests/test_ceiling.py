# ceiling.py on synthetic rollouts and probe predictions: pooling, alignment, AUROC, paired deltas.
# plain script, CPU only, exits non-zero on failure.
import json, os, sys, tempfile
import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src", "probe_cmp"))
import ceiling as C  # noqa: E402

fails = []


def check(name, cond, extra=""):
    print(f"{'ok  ' if cond else 'FAIL'} {name}{(' -- ' + str(extra)) if extra else ''}")
    if not cond:
        fails.append(name)


rng = np.random.default_rng(0)
d = tempfile.mkdtemp()
traces, rows, preds = [], [], {c: [] for c in C.CKPTS.values()}
for pid in range(12):
    for j in range(4):
        tid, ok = f"{pid}_{j}", j < 2
        traces.append({"trace_id": tid, "problem_id": pid, "is_correct": ok, "n_gen": 5000, "text": "x",
                       "trunc": False})
        for rc, pc in C.CKPTS.items():
            k = 6 if ok else 2  # r_hat separates every pair perfectly
            rows.append({"prefix_id": f"ceil/{tid}/{rc}", "round": 0, "problem_id": pid, "trace_id": tid,
                         "checkpoint": rc, "samples": [{"sample_idx": i, "correct": i < k, "truncated": i == 7}
                                                       for i in range(8)]})
            for seed in (0, 1):
                for arm, (model, feat, layer) in {"raw20": ("lr", "raw", 20), "raw16": ("lr", "raw", 16),
                                                  "len": ("lr", "len", C.NO_LAYER),
                                                  "tfidf": ("lr", "tfidf", C.NO_LAYER)}.items():
                    p = float(not ok) if arm == "len" else rng.random()  # len perfect, the rest noise
                    preds[pc].append({"ckpt": pc, "layer": layer, "model": model, "feature": feat, "seed": seed,
                                      "trace_id": tid, "problem_id": pid, "y": int(not ok), "p": p})
    rows.append({"prefix_id": f"ceil/{pid}/t0", "round": 0, "problem_id": pid, "trace_id": None, "checkpoint": "t0",
                 "samples": [{"sample_idx": i, "correct": i < 4, "truncated": False} for i in range(8)]})
# a top-up round on one prefix, and a torn last line
rows.append({"prefix_id": "ceil/0_0/p512", "round": 1, "problem_id": 0, "trace_id": "0_0", "checkpoint": "p512",
             "samples": [{"sample_idx": i, "correct": True, "truncated": False} for i in range(8)]})
rp = os.path.join(d, "rollouts.jsonl")
with open(rp, "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
    f.write('{"prefix_id": "ceil/torn')
tp = os.path.join(d, "traces.jsonl")
with open(tp, "w") as f:
    for t in traces:
        f.write(json.dumps(t) + "\n")
pd_dir = os.path.join(d, "preds")
os.makedirs(pd_dir)
for pc, v in preds.items():
    pd.DataFrame(v).to_parquet(os.path.join(pd_dir, f"{pc}.parquet"))

print("1. loading")
roll, torn = C.load_rollouts(rp)
check("torn last line skipped", torn == 1)
check("one row per prefix: 12 problems x (4 traces x 3 cuts + t0)", len(roll) == 12 * 13, len(roll))
top = roll.set_index("prefix_id").loc["ceil/0_0/p512"]
check("rounds pooled: 16 samples, 14 correct", (top.n, top.k) == (16, 14), (top.n, top.k))
x = roll.set_index("prefix_id").loc["ceil/1_2/p512"]
check("r counts truncated as not correct; r_finished drops them", x.r == 2 / 8 and x.r_finished == 2 / 7,
      (x.r, x.r_finished))

print("\n2. ceiling vs probes")
tr = C.load_traces(tp)
out = os.path.join(d, "out")
allr, desc = C.run(roll, tr, pd_dir, out, drift={"0_0", "0_1"}, n_boot=200, log=lambda *_: None)
a = allr[allr.variant == "all traces"].set_index(["checkpoint", "score"])
check("ceiling separates every pair -> AUROC 1", np.isclose(a.loc[("512", "ceiling r_hat"), "auroc"], 1.0))
check("noise probe near chance", 0.25 < a.loc[("512", "lr/raw L20"), "auroc"] < 0.75, a.loc[("512", "lr/raw L20"), "auroc"])
check("a probe equal to the labels ties the ceiling", np.isclose(a.loc[("512", "len"), "delta_ceiling_minus"], 0.0))
check("ceiling - noise probe positive, CI above 0",
      a.loc[("512", "lr/raw L20"), "delta_ceiling_minus"] > 0 and a.loc[("512", "lr/raw L20"), "delta_ci_low"] > 0)
check("identical traces: 48 traces, 12 problems, 48 pairs at every checkpoint",
      set(zip(a.n_traces, a.n_problems, a.n_pairs)) == {(48, 12, 48)}, set(zip(a.n_traces, a.n_problems, a.n_pairs)))
v = allr[allr.variant != "all traces"]
check("drift variant drops the flagged traces; problem 0 loses its correct side and leaves the pairs",
      set(v.n_traces) == {46} and set(v.n_problems) == {11}, (set(v.n_traces), set(v.n_problems)))
check("t0 rows described but never scored",
      "t0" in set(desc["source trace"]) and "t0" not in set(allr.checkpoint))
check("summary and csvs written", all(os.path.exists(os.path.join(out, f)) for f in
                                      ("ceiling_summary.md", "ceiling_vs_probe.csv", "r_by_source.csv")))

print("\n3. alignment is enforced")
p = pd.read_parquet(os.path.join(pd_dir, "512.parquet"))
p[p.trace_id != "3_1"].to_parquet(os.path.join(pd_dir, "512.parquet"))
try:
    C.run(roll, tr, pd_dir, out, n_boot=50, log=lambda *_: None)
    check("missing probe prediction for a rollout trace stops the run", False)
except SystemExit as e:
    check("missing probe prediction for a rollout trace stops the run", "no probe prediction" in str(e), e)

print()
if fails:
    print(f"{len(fails)} FAILED: {fails}")
    sys.exit(1)
print("all passed")
