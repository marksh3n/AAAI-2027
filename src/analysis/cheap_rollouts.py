# X2: how much of r_hat's within-problem AUROC survives with fewer, shorter continuations, and what it costs.
# per live prefix at 512 / 2048 / 4096: K' of the 8 ceiling continuations (averaged over random subsets),
# each capped at L tokens. three scores, all P(wrong) with the sign fixed in advance:
#   correct   1 - share of the K' that end correct within L (needs gold)
#   finished  1 - share that finish within L (gold-free; a remaining-length proxy)
#   agree     1 - modal-answer count / K' among those finished within L (gold-free)
# cost per trace = expected sum of min(n_gen, L) over the K' continuations. target: the source trace's own
# outcome. AUROCs pair-weighted within problem, problem-cluster bootstrap shared by all rows at a cut, so every
# delta vs the full r_hat (K' 8, no cap) is paired. answers compared by normalized string (no math_verify here).
#   python src/analysis/cheap_rollouts.py --rollouts data/recovery_ceiling --traces <traces_B.jsonl> \
#       --split results/split.json --half explore --out <dir>
import argparse, json, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "probe_cmp"))
import stats as S  # noqa: E402
from ceiling import norm  # noqa: E402

CUTS = {"p512": 512, "p2048": 2048, "p4096": 4096}
KS = (1, 2, 4, 8)
LS = (256, 512, 1024, 2048, 4096, 32768)
N_SUB = 20
K_FULL = 8


def load(path):
    # (prefix_id, round, sample_idx) -> last sample written; first 8 by (round, idx) per prefix
    last = {}
    for line in open(path):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r["checkpoint"] not in CUTS:
            continue
        for s in r["samples"]:
            last[(r["prefix_id"], r["round"], s["sample_idx"])] = (r, s)
    by = {}
    for (pid, rd, i), (r, s) in sorted(last.items(), key=lambda kv: kv[0]):
        by.setdefault(pid, {"trace_id": r["trace_id"], "problem_id": r["problem_id"],
                            "cut": CUTS[r["checkpoint"]], "s": []})["s"].append(s)
    rows = []
    for pid, d in by.items():
        s = d["s"][:K_FULL]
        if len(s) < K_FULL:
            continue
        ans = [norm(x["answer"]) if x.get("answer") is not None else None for x in s]
        cls = {a: i for i, a in enumerate(dict.fromkeys(a for a in ans if a))}
        rows.append({"prefix_id": pid, "trace_id": d["trace_id"], "problem_id": d["problem_id"], "cut": d["cut"],
                     "n": np.array([x["n_gen"] for x in s]), "ok": np.array([bool(x["correct"]) for x in s]),
                     "fin": np.array([x["finish_reason"] == "stop" and not x["truncated"] for x in s]),
                     "cls": np.array([cls[a] if a else -1 for a in ans])})
    return pd.DataFrame(rows)


def subsets(n_rows, k, rng):
    # one random k-subset of the 8 per row and draw: (draws, rows, k)
    d = 1 if k == K_FULL else N_SUB
    return np.argsort(rng.random((d, n_rows, K_FULL)), axis=2)[:, :, :k]


def score(t, idx, L, kind):
    # idx (rows, k) -> P(wrong) per row
    n, ok, fin, cls = (np.stack(t[c].values) for c in ("n", "ok", "fin", "cls"))
    take = lambda a: np.take_along_axis(a, idx, axis=1)
    within = take(n) <= L
    if kind == "correct":
        return 1 - (take(ok) & within).mean(1)
    if kind == "finished":
        return 1 - (take(fin) & within).mean(1)
    c = np.where(take(fin) & within, take(cls), -1)
    modal = np.array([np.bincount(r[r >= 0]).max() if (r >= 0).any() else 0 for r in c])
    return 1 - modal / idx.shape[1]


def run(t, y, n_boot=2000, seed=0):
    rows = []
    for cut in sorted(t.cut.unique()):
        tc = t[t.cut == cut].reset_index(drop=True)
        yc = y.loc[tc.trace_id].values
        pairs = S.Pairs(yc, tc.problem_id.values)
        draws = S.problem_draws(len(pairs.pids), n_boot, seed)
        full = score(tc, np.tile(np.arange(K_FULL), (len(tc), 1)), LS[-1], "correct")
        full_b = S.boot_within(pairs, full, draws)[0]
        n_all = np.stack(tc.n.values)
        rng = np.random.default_rng(seed)
        subs = {k: subsets(len(tc), k, rng) for k in KS}
        for kind in ("correct", "finished", "agree"):
            for k in KS:
                for L in LS:
                    est, boot = [], []
                    for idx in subs[k]:
                        s = score(tc, idx, L, kind)
                        est.append(pairs.auroc(s)["weighted"])
                        boot.append(S.boot_within(pairs, s, draws)[0])
                    a, b = float(np.mean(est)), np.mean(boot, axis=0)
                    tok = k * float(np.minimum(n_all, L).mean())
                    rows.append({"cut": int(cut), "score": kind, "K": k, "L": L, "tokens_per_trace": tok,
                                 "auroc": a, "lo": S.ci(b)[0], "hi": S.ci(b)[1],
                                 "delta_vs_full": a - pairs.auroc(full)["weighted"],
                                 "dlo": S.ci(b - full_b)[0], "dhi": S.ci(b - full_b)[1],
                                 "traces": len(tc), "problems": len(pairs.pids), "pairs": int(pairs.pairs.sum())})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", required=True)
    ap.add_argument("--traces", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--half", choices=["explore", "holdout", "all"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()
    t = load(a.rollouts)
    if a.half != "all":
        t = t[t.problem_id.isin(set(json.load(open(a.split))[a.half]))]
    tr = pd.DataFrame(json.loads(line) for line in open(a.traces)).set_index("trace_id")
    y = (~tr.is_correct.astype(bool)).astype(int)
    res = run(t, y, a.n_boot)
    res["half"] = a.half
    res["mean_source_tokens"] = float(tr.loc[t.trace_id.unique(), "n_gen"].mean())
    os.makedirs(a.out, exist_ok=True)
    res.to_csv(os.path.join(a.out, f"x2_{a.half}.csv"), index=False)
    print(res[(res.L.isin([1024, 32768]))].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
