# remaining rows of the three-model table (appendix) at the 4,096 cut: D by MATH level, D on
# pre-answer cuts (no </think> or \boxed in the prefix), mean source length of cohort traces, median re-run
# length from the cut, and the sampler gap (mean of r_hat minus the source's own outcome, problem bootstrap).
# each under the cohort rule ("pairs") and on every re-run problem ("all"). DeepSeek's pre-answer row is in
# robustness.py.
#   python src/analysis/cohort_stats.py --out results/cohort_stats.csv
import argparse
import numpy as np
import pandas as pd

import inputs as I
from inputs import S

ap = argparse.ArgumentParser()
ap.add_argument("--limit-hits", choices=["drop", "wrong"], default="drop")
ap.add_argument("--out", required=True)
a = ap.parse_args()


def gap(t, y):
    d = pd.DataFrame({"p": t.problem_id.values, "g": t.ok.map(np.mean).values - t.trace_id.map(y).astype(float).values})
    per = d.groupby("p").g.agg(["sum", "size"])
    dr = S.problem_draws(len(per), 2000, 0)
    lo, hi = S.ci((dr @ per["sum"].values) / (dr @ per["size"].values))
    return {"gap_pp": 100 * per["sum"].sum() / per["size"].sum(), "gap_lo": 100 * lo, "gap_hi": 100 * hi}


rows = []
for model in I.LABELS:
    if model == "qwen3" and a.limit_hits == "wrong":
        continue  # Qwen3's bank has no limit-hit sources
    tr, t = I.bank(model, a.limit_hits)
    y = tr.is_correct.astype(bool)
    lv = I.levels(model)
    pre = None if model == "deepseek" else set(I.states(model)[0].query("pre_answer").trace_id)
    for rule, sub in (("all", t), ("pairs", I.both_outcomes(t, tr))):
        base = {"model": model, "rule": rule}
        rows.append({**base, "what": "D", **I.d_ci(sub)})
        for k in sorted(sub.problem_id.map(lv).unique()):
            rows.append({**base, "what": f"D level {k}", **I.d_ci(sub[sub.problem_id.map(lv) == k])})
        if pre is not None:
            rows.append({**base, "what": "D pre-answer", **I.d_ci(sub[sub.trace_id.isin(pre)])})
        rows.append({**base, "what": "lengths", "mean_source_tokens": float(tr.loc[sub.trace_id.unique(), "n_gen"].mean()),
                     "median_rerun_tokens": float(np.median(np.concatenate(sub.n.values))), **gap(sub, y)})
out = pd.DataFrame(rows)
out.to_csv(a.out, index=False)
print(out.round(3).to_string(index=False))
