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
traces, rows, preds, meta = [], [], {c: [] for c in C.CKPTS.values()}, []


def sample(i, k):
    # correct continuations answer 7 (gold), the rest 9, the truncated one none
    return {"sample_idx": i, "correct": i < k, "truncated": i == 7, "answer": None if i == 7 else ("7" if i < k else "9"),
            "answer_from_prefix": False}


for pid in range(12):
    for j in range(4):
        tid, ok = f"{pid}_{j}", j < 2
        traces.append({"trace_id": tid, "problem_id": pid, "is_correct": ok, "n_gen": 5000, "text": "x",
                       "trunc": False, "answer": "7" if ok else "9"})
        for rc, pc in C.CKPTS.items():
            k = 6 if ok else 2  # r_hat separates every pair perfectly
            rows.append({"prefix_id": f"ceil/{tid}/{rc}", "round": 0, "problem_id": pid, "trace_id": tid,
                         "checkpoint": rc, "samples": [sample(i, k) for i in range(8)]})
            # half the problems' right attempts have boxed their answer by 4096; one states it at 2048
            text = "let me think"
            if ok and pc == "4096" and pid % 2 == 0:
                text += " so \\boxed{7}"
            if tid == "1_0" and pc == "2048":
                text += " so the answer is 7 then"
            meta.append({"trace_id": tid, "pos": int(pc), "prefix": text, "restarts": 1 if ok else 3})
            for seed in (0, 1):
                for arm, (model, feat, layer) in {"raw20": ("lr", "raw", 20), "raw16": ("lr", "raw", 16),
                                                  "len": ("lr", "len", C.NO_LAYER),
                                                  "tfidf": ("lr", "tfidf", C.NO_LAYER)}.items():
                    p = float(not ok) if arm == "len" else rng.random()  # len perfect, the rest noise
                    preds[pc].append({"ckpt": pc, "layer": layer, "model": model, "feature": feat, "seed": seed,
                                      "trace_id": tid, "problem_id": pid, "y": int(not ok), "p": p})
    rows.append({"prefix_id": f"ceil/{pid}/t0", "round": 0, "problem_id": pid, "trace_id": None, "checkpoint": "t0",
                 "samples": [dict(sample(i, 4), truncated=False, answer="7" if i < 4 else "9") for i in range(8)]})
# a top-up round on one prefix, and a torn last line
rows.append({"prefix_id": "ceil/0_0/p512", "round": 1, "problem_id": 0, "trace_id": "0_0", "checkpoint": "p512",
             "samples": [{"sample_idx": i, "correct": True, "truncated": False, "answer": "7"} for i in range(8)]})
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
mp = os.path.join(d, "meta.parquet")
pd.DataFrame(meta).to_parquet(mp)

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
allr, desc = C.run(roll, tr, pd_dir, out, drift={"0_0", "0_1"}, meta=C.load_meta(mp), n_boot=200,
                   log=lambda *_: None)
a = allr[allr.variant == "all traces"].set_index(["checkpoint", "score"])
check("ceiling separates every pair -> AUROC 1", np.isclose(a.loc[("512", "r_hat"), "auroc"], 1.0))
check("noise probe near chance", 0.25 < a.loc[("512", "lr/raw L20"), "auroc"] < 0.75, a.loc[("512", "lr/raw L20"), "auroc"])
check("a probe equal to the labels ties the ceiling", np.isclose(a.loc[("512", "len"), "delta_r_hat_minus"], 0.0))
check("ceiling - noise probe positive, CI above 0",
      a.loc[("512", "lr/raw L20"), "delta_r_hat_minus"] > 0 and a.loc[("512", "lr/raw L20"), "delta_ci_low"] > 0)
check("identical traces: 48 traces, 12 problems, 48 pairs at every checkpoint",
      set(zip(a.n_traces, a.n_problems, a.n_pairs)) == {(48, 12, 48)}, set(zip(a.n_traces, a.n_problems, a.n_pairs)))
v = allr[allr.variant.str.startswith("without")]
check("drift variant drops the flagged traces; problem 0 loses its correct side and leaves the pairs",
      set(v.n_traces) == {46} and set(v.n_problems) == {11}, (set(v.n_traces), set(v.n_problems)))

print("\n3. agreement, answer-in-prefix, restarts, curves")
check("answers kept per prefix", roll.set_index("prefix_id").loc["ceil/2_0/p512", "answers"][:2] == ["7", "7"])
dsc = desc.set_index(["checkpoint", "source trace"])
check("agreement with the source's own answer: 6/8 for right sources, 5/8 for wrong ones",
      np.isclose(dsc.loc[("p2048", "correct"), "mean agreement"], 0.75)
      and np.isclose(dsc.loc[("p2048", "wrong"), "mean agreement"], 5 / 8), dsc["mean agreement"].to_dict())
check("committed cuts: half the right attempts at 4096, none earlier",
      np.isclose(dsc.loc[("p4096", "correct"), "cuts committed"], 0.5)
      and dsc.loc[("p2048", "correct"), "cuts committed"] == 0, dsc["cuts committed"].to_dict())
b = allr[allr.variant.str.startswith("before the answer is committed")].set_index(["checkpoint", "score"])
check("before-commit variant drops the 12 boxed cuts at 4096 only",
      b.loc[("4096", "r_hat"), "n_traces"] == 36 and b.loc[("512", "r_hat"), "n_traces"] == 48,
      (b.loc[("4096", "r_hat"), "n_traces"], b.loc[("512", "r_hat"), "n_traces"]))
w = allr[allr.variant.str.startswith("before the answer value")].set_index(["checkpoint", "score"])
check("value-written variant also drops the 2048 cut that states 7",
      w.loc[("2048", "r_hat"), "n_traces"] == 47, w.loc[("2048", "r_hat"), "n_traces"])
check("raw restart count scored on the same pairs",
      np.isclose(a.loc[("512", "restart density (raw)"), "auroc"], 1.0))
check("recovery curves written", os.path.exists(os.path.join(out, "recovery_curves.csv"))
      and os.path.exists(os.path.join(out, "recovery_curves.png")))
check("answer flags: boxed and </think> commit; a bare value only counts when it stands alone",
      C.answer_flags("x </think> y", "5") == (True, True) and C.answer_flags("got 15 and 7.5", "5") == (False, False)
      and C.answer_flags("so it is 5.", "5") == (False, True))
check("t0 rows described but never scored",
      "t0" in set(desc["source trace"]) and "t0" not in set(allr.checkpoint))
check("summary and csvs written", all(os.path.exists(os.path.join(out, f)) for f in
                                      ("ceiling_summary.md", "ceiling_vs_probe.csv", "r_by_source.csv")))

print("\n4. alignment is enforced")
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
