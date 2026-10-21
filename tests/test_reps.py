# representation test: extract_reps.py's window / last-token arithmetic across chunk boundaries (torch
# stubbed with a numpy-backed fake model), and reps.py on a planted signal. plain script, CPU only.
import contextlib, json, os, sys, tempfile, types
import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src", "probe_cmp"))

fails = []


def check(name, cond, extra=None):
    print(f"{'ok  ' if cond else 'FAIL'} {name}{(' -- ' + str(extra)) if extra is not None else ''}")
    if not cond:
        fails.append(name)


# --- 1. extract_reps.reps with a fake model -------------------------------------------------------
class T:
    # just enough of a tensor: slicing, float, sum over rows, cpu, numpy
    def __init__(self, a):
        self.a = np.asarray(a)

    def __getitem__(self, k):
        return T(self.a[k])

    def float(self):
        return T(self.a.astype(np.float32))

    def sum(self, axis):
        return T(self.a.sum(axis))

    def cpu(self):
        return self

    def numpy(self):
        return self.a

    @property
    def shape(self):
        return self.a.shape


sys.modules["torch"] = types.SimpleNamespace(tensor=lambda x, device=None: T(x), no_grad=contextlib.nullcontext,
                                         Tensor=type("Tensor", (), {}))  # scipy probes torch.Tensor
DIM = 3


class FakeModel:
    # hidden_states[l + 1] at absolute position p = (p, 10 l, the token id); a kv cache is just the offset
    config = types.SimpleNamespace(hidden_size=DIM)

    def __init__(self):
        self.model = self

    def __call__(self, piece, past_key_values=None, use_cache=True, output_hidden_states=True):
        start = past_key_values or 0
        ids = piece.a[0]
        pos = np.arange(start, start + len(ids))
        hs = [T(np.stack([pos, np.full(len(ids), 10 * (l - 1)), ids], 1)[None].astype(np.float32))
              for l in range(30)]
        return types.SimpleNamespace(hidden_states=hs, past_key_values=start + len(ids))


import extract_reps as X  # noqa: E402

prompt, trace = list(range(1000, 1007)), list(range(50))  # 7 prompt tokens, 50 generated
cuts, window = [5, 20, 40], 8
for chunk in (4, 7, 64):  # windows straddle chunk boundaries, or all in one chunk
    last, mean = X.reps(FakeModel(), "cpu", prompt, trace, [20, 16], cuts, window, chunk)
    P = len(prompt)
    ok_last = all(last[ci, 0, 0] == P + c - 1 and last[ci, 0, 2] == trace[c - 1] for ci, c in enumerate(cuts))
    # window [P + c - 8, P + c): positions average to P + c - 4.5; the cut-5 window reaches back into the prompt
    ok_mean = all(np.isclose(mean[ci, 0, 0], P + c - 4.5) for ci, c in enumerate(cuts))
    ok_prompt = np.isclose(mean[0, 0, 2], np.mean((prompt + trace)[P + 5 - 8: P + 5]))
    ok_layer = np.allclose(last[:, 0, 1], 200) and np.allclose(last[:, 1, 1], 160)  # hidden_states[l + 1]
    check(f"chunk {chunk}: last token = index pos-1", ok_last, last[:, 0, 0])
    check(f"chunk {chunk}: mean = trailing window", ok_mean, mean[:, 0, 0])
    check(f"chunk {chunk}: window may reach into the prompt", ok_prompt)
    check(f"chunk {chunk}: layer l reads hidden_states[l + 1]", ok_layer)
check("window bounds", X.window_bounds(7, 128, 128) == (7, 135))

# population from rollouts skips t0 rows and a torn line
d = tempfile.mkdtemp()
rp = os.path.join(d, "r.jsonl")
with open(rp, "w") as f:
    f.write(json.dumps({"trace_id": "1_0"}) + "\n" + json.dumps({"trace_id": None}) + "\n{\"torn")
check("population = rollout source traces", X.population(rp) == {"1_0"})

# --- 2. reps.py on a planted signal ---------------------------------------------------------------
import reps as RP  # noqa: E402
from ceiling import load_rollouts, load_traces  # noqa: E402

rng = np.random.default_rng(0)
NP, K, D, CUTS = 40, 6, 60, [128, 256, 512, 1024, 2048, 4096]
traces, rows, tids = [], [], []
last = rng.normal(size=(NP * K, len(CUTS), 1, D)).astype(np.float32)
mean = rng.normal(size=(NP * K, len(CUTS), 1, D)).astype(np.float32)
for p in range(NP):
    off = rng.normal(0, 3, D)  # problem identity dominates every view
    for k in range(K):
        i, wrong = p * K + k, k < 3
        tid = f"{p}_{k}"
        tids.append(tid)
        traces.append({"trace_id": tid, "problem_id": p, "is_correct": not wrong, "n_gen": 5000, "text": "x",
                       "trunc": False, "answer": "1"})
        last[i] += off
        mean[i] += off
        mean[i, :, 0, 0] += 3.0 * wrong  # only the trailing mean carries the outcome
        for c in ("p512", "p2048", "p4096"):
            r = 0.2 if wrong else 0.8
            rows.append({"prefix_id": f"{tid}/{c}", "round": 0, "problem_id": p, "trace_id": tid, "checkpoint": c,
                         "samples": [{"sample_idx": s, "correct": bool(rng.random() < r), "truncated": False,
                                      "answer": None} for s in range(8)]})
# one trace has no rollout at 4096: dropped from every 4096 score
rows = [r for r in rows if not (r["trace_id"] == "0_0" and r["checkpoint"] == "p4096")]
tp, rp = os.path.join(d, "t.jsonl"), os.path.join(d, "roll.jsonl")
open(tp, "w").write("\n".join(json.dumps(t) for t in traces) + "\n")
open(rp, "w").write("\n".join(json.dumps(r) for r in rows) + "\n")
R = {"trace_id": np.array(tids), "last": last, "mean": mean, "layers": np.array([20]), "cuts": np.array(CUTS)}
roll, _ = load_rollouts(rp)
logs = []
res = RP.run(R, roll, load_traces(tp), layer=20, seeds=[0], out=os.path.join(d, "out"), log=logs.append, n_boot=300)
get = lambda c, s: res[(res.checkpoint == c) & (res.score == s)].iloc[0]
m, l = get(2048, "mean128 lr/raw"), get(2048, "last lr/raw")
check("planted trailing-mean signal found", m.auroc > 0.8, f"{m.auroc:.3f}")
check("last token without signal near 0.5", abs(l.auroc - 0.5) < 0.12, f"{l.auroc:.3f}")
check("mean128 - last is positive and paired", m["minus last lr/raw"] > 0.2 and m["minus last lr/raw ci"].startswith("[+"),
      f"{m['minus last lr/raw']:+.3f} {m['minus last lr/raw ci']}")
dm = get(2048, "delta mean128 lr/raw")
check("delta of a constant-per-trace shift carries no signal", abs(dm.auroc - 0.5) < 0.12, f"{dm.auroc:.3f}")
check("r_hat scored on the same traces", get(512, "r_hat").auroc > 0.8)
check("4096 drops the trace without rollouts", get(4096, "r_hat").n_pairs == get(2048, "r_hat").n_pairs - 3,
      (get(4096, "r_hat").n_pairs, get(2048, "r_hat").n_pairs))
check("drop is logged", any("without rollouts" in s for s in logs))
check("Holm column on probe rows only", res[res.score == "r_hat"]["holm_vs_0.5"].isna().all()
      and res[res.score.str.contains(" lr/")]["holm_vs_0.5"].notna().all())
summary = open(os.path.join(d, "out", "reps_summary.md")).read()
check("summary written and flagged exploratory", "Exploratory" in summary)
# 4 views x 2 arms x 3 cuts; a probe_cmp reference row (which has no Holm p) must not join the count
res2 = pd.concat([res, res[res.score == "last lr/raw"].assign(score="last lr/raw L20 (probe_cmp extraction)",
                                                               **{"holm_vs_0.5": np.nan})], ignore_index=True)
check("Holm count over the 24 probe cells only", "of 24" in RP.report(res2, 20))

print(f"\n{len(fails)} failure(s)" + (f": {fails}" if fails else ""))
sys.exit(1 if fails else 0)
