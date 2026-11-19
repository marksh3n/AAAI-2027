# token confidence (DeepConf, arXiv 2508.15260) as a free readout at the 4,096 cut: capture of D and
# within-problem AUROC against the source outcome (Table T8). per-token values come from
# src/replication/confidence.py (raw model distribution, temperature 1, top 20 for token confidence):
#   conf_mean      mean token confidence
#   lowest_group   lowest mean over sliding 2,048-token windows (stride 1)
#   bottom10       mean of the lowest 10% of those window means
#   tail           mean over the last 2,048 tokens before the cut
#   entropy_mean, logp_mean   mean full-vocabulary entropy; mean log-prob of the sampled tokens
# DeepSeek's ids are re-tokenized text, so a DeepSeek trace is dropped when its ids fail the count check, the
# prefix hash check, or put more tokens outside the T 0.7 / top-p 0.95 nucleus than any stored-id trace of the
# other two models. D is recomputed on the kept traces under the cohort rule.
#   python src/analysis/confidence_capture.py --out results/confidence
import argparse, os
import numpy as np
import pandas as pd

import inputs as I
from inputs import DT, S

WIN, TAIL, BOTTOM = 2048, 2048, 0.10
FEATS = ("conf_mean", "lowest_group", "bottom10", "tail", "entropy_mean", "logp_mean")

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)


def tokens(model):
    d = f"{I.DATA}/confidence/{model}"
    idx = pd.read_parquet(f"{d}/index.parquet")
    with np.load(f"{d}/tokens.npz") as z:  # decompress each array once; NpzFile re-reads on every access
        tok = {k: z[k] for k in z.files}
    off = np.concatenate([[0], np.cumsum(idx.n_tokens.values)])
    return idx.set_index("trace_id"), {tid: {k: v[off[i]:off[i + 1]] for k, v in tok.items()}
                                       for i, tid in enumerate(idx.trace_id)}


def windows(x, n):
    if len(x) <= n:
        return np.array([x.mean()])
    c = np.concatenate([[0.0], np.cumsum(x, dtype=np.float64)])
    return (c[n:] - c[:-n]) / n


def features(t, cut):
    conf = t["conf"][:cut].astype(np.float64)
    w = windows(conf, WIN)
    k = max(1, int(np.ceil(BOTTOM * len(w))))
    return {"conf_mean": conf.mean(), "lowest_group": w.min(), "bottom10": np.sort(w)[:k].mean(),
            "tail": conf[-TAIL:].mean(), "entropy_mean": t["entropy"][:cut].mean(), "logp_mean": t["logp"][:cut].mean(),
            "outside_nucleus": int((t["mass_above"][:cut] >= 0.95).sum())}


data = {m: (*I.bank(m), *tokens(m)) for m in I.LABELS}
# nucleus tolerance: the highest per-trace out-of-nucleus rate among stored-id traces (HF vs vLLM numerics)
tol = max(float(np.mean(t["mass_above"] >= 0.95)) for m in ("qwen3", "ministral") for t in data[m][3].values())
rows, checks = [], []
for m, (tr, t, ix, arr) in data.items():
    cut = I.CUT
    f = pd.DataFrame([features(arr[tid], cut) for tid in t.trace_id])
    keep = pd.Series(True, index=f.index)
    why = {}
    if m == "deepseek":
        nucleus = (f.outside_nucleus / cut <= tol).values
        canon = ix.loc[t.trace_id, "canonical"].values.astype(bool)
        sha = np.array([cut not in b for b in ix.loc[t.trace_id, "prefix_sha_bad_cuts"]])
        why = {"outside nucleus": int((~nucleus).sum()), "count check": int((~canon).sum()),
               "prefix_sha": int((~sha).sum())}
        keep &= nucleus & canon & sha
    checks.append({"model": I.LABELS[m], "cuts": len(t), "kept": int(keep.sum()),
                   "share_outside_nucleus": float((f.outside_nucleus / cut).mean()),
                   **{f"dropped: {k}": v for k, v in why.items()}})
    tc, f = t[keep.values].reset_index(drop=True), f[keep.values].reset_index(drop=True)
    ok2 = tc.groupby("problem_id").trace_id.transform("size").values >= 2
    tc, f = tc[ok2].reset_index(drop=True), f[ok2].reset_index(drop=True)
    yy = tr.loc[tc.trace_id, "is_correct"].astype(bool).values
    both = pd.Series(yy).groupby(tc.problem_id.values).transform("nunique").values == 2
    tc, f = tc[both].reset_index(drop=True), f[both].reset_index(drop=True)

    g = tc.problem_id.values
    A, B, _ = DT.halves(tc)
    s_fix, s_tot = DT.sums(g, A, B), DT.total_sums(tc)
    pids = s_fix.index.values
    draws = S.problem_draws(len(pids), 2000, 0)
    fix, tot = s_fix.loc[pids].values, s_tot.loc[pids].values
    D = fix.sum() / tot.sum()
    lo, hi = S.ci((draws @ fix) / (draws @ tot))
    pairs = S.Pairs((~tr.loc[tc.trace_id, "is_correct"].astype(bool)).astype(int).values, g)
    rows.append({"model": I.LABELS[m], "what": "determined", "est": D, "lo": lo, "hi": hi, "traces": len(tc),
                 "problems": len(pids)})
    for k in FEATS:
        x = f[k].values
        sa, sb, sf = (DT.sums(g, x, h).loc[pids].values for h in (A, B, x))
        cap = sa.sum() * sb.sum() / (sf.sum() * fix.sum())
        lo, hi = S.ci((draws @ sa) * (draws @ sb) / ((draws @ sf) * (draws @ fix)))
        rows.append({"model": I.LABELS[m], "what": f"capture:{k}", "est": cap, "lo": lo, "hi": hi,
                     "share_of_total": cap * D, "wp_auroc_y": pairs.auroc(-x)["weighted"], "traces": len(tc),
                     "problems": len(pids)})
res, chk = pd.DataFrame(rows), pd.DataFrame(checks)
res.to_csv(f"{a.out}/capture.csv", index=False)
chk.to_csv(f"{a.out}/checks.csv", index=False)
print(f"nucleus tolerance {tol:.2e}\n{chk.to_string(index=False)}\n\n{res.round(3).to_string(index=False)}")
