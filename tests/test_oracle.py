# oracle.py on synthetic rollouts: halves, fractional ties, cross-fitting, the random baseline, the
# resample control and the restart ceiling. plain script, CPU only, exits non-zero on failure.
import json, os, sys, tempfile
import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src", "probe_cmp"))
import oracle as O  # noqa: E402
from ceiling import load_traces  # noqa: E402

fails = []


def check(name, cond, extra=None):
    print(f"{'ok  ' if cond else 'FAIL'} {name}{(' -- ' + str(extra)) if extra is not None and str(extra) else ''}")
    if not cond:
        fails.append(name)


# 1. fractional ties: a group straddling the budget shares what is left
w = O.flag_weights([3, 2, 2, 1], 0.5)
check("tie group split evenly", np.allclose(w, [1, 0.5, 0.5, 0]), w)
w = O.flag_weights(np.arange(10.0), 0.3)
check("top 3 of 10 flagged", np.allclose(w, [0] * 7 + [1] * 3), w)
check("q=1 flags all", np.allclose(O.flag_weights([1, 1, 0], 1.0), 1))
check("weights sum to q * n", np.isclose(O.flag_weights(np.random.default_rng(0).integers(0, 4, 50), 0.37).sum(), 18.5))

# 2. cross-fitting: half 0 calls trace A hopeless and B fine, half 1 the reverse. with q=0.5 half 0
# flags A and is scored on half 1 (A's resample succeeds), half 1 flags B, scored on half 0
t = pd.DataFrame({"trace_id": ["A", "B"], "problem_id": [0, 0], "y": [1, 1], "src_tokens": [5000, 5000],
                  "cut": [512, 512], "rp0": [0.0, 1.0], "rp1": [1.0, 0.0], "Lp0": [100.0, 300.0],
                  "Lp1": [200.0, 400.0], "rt0": [0.5, 0.5], "rt1": [0.5, 0.5], "Lt0": [3000.0, 3000.0],
                  "Lt1": [3000.0, 3000.0]})
fn = O.scores(t, None)["oracle r_hat (low prefix r first)"]
acc, tok, flag = O.policy(t, fn, "resample", 0.5)
# A: flagged by half 0 only, scored on half 1: gain (1 - 0) / 2; B: flagged by half 1, scored on half 0: (1 - 0) / 2
check("cross-fit scores the replacement on the other half", np.allclose(acc, [0.5, 0.5]), acc)
check("cross-fit tokens from the other half", np.allclose(tok, [(512 + 200 - 5000) / 2, (512 + 300 - 5000) / 2]), tok)
check("each trace flagged in one half of two", np.allclose(flag, [0.5, 0.5]), flag)
racc, _, _ = O.policy(t, None, "restart", 0.4)
check("random flagging = q x mean gain over halves", np.allclose(racc, 0.4 * 0.5), racc)

# 3. synthetic population. each trace has a latent r; its own outcome and all its rollouts are draws
# from it, so the source and a resample are exchangeable given the prefix. every problem's t0 is 0.5
rng = np.random.default_rng(1)
d = tempfile.mkdtemp()
traces, rows = [], []
for pid in range(80):
    for j in range(6):
        tid = f"{pid}_{j}"
        r = rng.uniform(0, 1)
        ok = bool(rng.random() < r)
        traces.append({"trace_id": tid, "problem_id": pid, "is_correct": ok, "n_gen": 6000, "text": "x",
                       "trunc": False, "answer": "1"})
        for c in O.CKPTS:
            rows.append({"prefix_id": f"ceil/{tid}/{c}", "round": 0, "problem_id": pid, "trace_id": tid,
                         "checkpoint": c, "samples": [{"sample_idx": i, "correct": bool(rng.random() < r),
                                                       "n_gen": 4000, "truncated": False} for i in range(8)]})
    rows.append({"prefix_id": f"ceil/{pid}/t0", "round": 0, "problem_id": pid, "trace_id": None, "checkpoint": "t0",
                 "samples": [{"sample_idx": i, "correct": i % 2 == 0, "n_gen": 5000, "truncated": False}
                             for i in range(8)]})
# a rewritten sample keeps the last copy; a torn last line is skipped
rows.append({"prefix_id": "ceil/0_0/p512", "round": 0, "problem_id": 0, "trace_id": "0_0", "checkpoint": "p512",
             "samples": [{"sample_idx": 0, "correct": True, "n_gen": 1234, "truncated": False}]})
rp, tp = os.path.join(d, "r.jsonl"), os.path.join(d, "t.jsonl")
with open(rp, "w") as f:
    f.write("\n".join(json.dumps(r) for r in rows) + "\n" + '{"prefix_id": "torn')
with open(tp, "w") as f:
    f.write("\n".join(json.dumps(t) for t in traces) + "\n")
samples, torn = O.load_samples(rp)
check("torn line skipped", torn == 1)
check("rewritten sample kept once, last copy",
      ((samples.prefix_id == "ceil/0_0/p512") & (samples.sample_idx == 0)).sum() == 1 and
      samples[(samples.prefix_id == "ceil/0_0/p512") & (samples.sample_idx == 0)].n_gen.iloc[0] == 1234)
check("halves alternate, 4 + 4 per prefix", (samples.groupby(["prefix_id", "half"]).size() == 4).all())

out = os.path.join(d, "out")
res = O.run(samples, load_traces(tp), out=out, log=lambda *_: None, n_boot=500)
get = lambda c, a, s, q: res[(res.checkpoint == c) & (res.action == a) & (res.score == s) & (res.q == q)].iloc[0]
ctl = get("2048", "resample", "oracle r_hat (low prefix r first)", 0.3)
check("resample control: prefix-only oracle has no value over random", ctl.over_ci_low < 0 < ctl.over_ci_high,
      f"{ctl.over_random:+.3f} [{ctl.over_ci_low:+.3f}, {ctl.over_ci_high:+.3f}]")
lab = get("2048", "resample", "label oracle (source wrong first)", 0.3)
check("resample with the source's outcome known does gain", lab.over_ci_low > 0, f"{lab.over_random:+.3f}")
adv = get("2048", "restart", "oracle advantage (r_t0 - r_prefix)", 0.2)
labr = get("2048", "restart", "label oracle (source wrong first)", 0.2)
check("restart: oracle advantage beats random", adv.over_ci_low > 0, f"{adv.over_random:+.3f}")
check("restart: label oracle at least as good as the prefix-only oracle", labr.over_random >= adv.over_random,
      f"{labr.over_random:+.3f} vs {adv.over_random:+.3f}")
rnd = get("2048", "restart", "random", 0.5)
src = 1 - np.mean([not t["is_correct"] for t in traces])
check("random restart at q=0.5 moves accuracy halfway to t0", np.isclose(rnd.d_acc, 0.5 * (0.5 - src)),
      f"{rnd.d_acc:.4f} vs {0.5 * (0.5 - src):.4f}")
check("random restart tokens: q x (cut + t0 length - source length)",
      np.isclose(rnd.d_tokens_per_trace, 0.5 * (2048 + 5000 - 6000)), rnd.d_tokens_per_trace)
check("q=1 flags everything", np.allclose(res[res.q == 1.0].flagged, 1))
check("summary written", "In plain words" in open(os.path.join(out, "oracle_summary.md")).read())
check("curve csv written", len(pd.read_csv(os.path.join(out, "oracle_curve.csv"))) == len(res))

# 4. a problem without t0 cannot be scored for restart
bad = samples[~((samples.checkpoint == "t0") & (samples.problem_id == 3))]
try:
    O.table(bad, load_traces(tp), "p512")
    check("missing t0 refused", False)
except SystemExit:
    check("missing t0 refused", True)

print(f"\n{len(fails)} failure(s)" + (f": {fails}" if fails else ""))
sys.exit(1 if fails else 0)
