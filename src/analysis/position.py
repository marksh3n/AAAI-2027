# exploratory: D by the position of the cut in the trace (Figure P1). two definitions of position:
#   eventual   cut / the source trace's own final length. not observable at the cut (it uses the future)
#   expected   cut / median final length of the problem's other source traces (leave-one-out)
# per-trace contributions: fixed_i = (rA_i - mean_p rA)(rB_i - mean_p rB), centred within (problem, cut), and
# total_i = the problem-cut's total sum shared equally by its traces. summed over all traces they give D
# exactly; a bin's D is sum fixed / sum total over its traces, with a problem bootstrap shared across bins.
# all cuts: DeepSeek 512 / 2,048 / 4,096, Qwen3 and Ministral 4,096; cohort rule as replicate.py.
#   python src/analysis/position.py --out results/position_bins.csv
import argparse
import numpy as np
import pandas as pd

import inputs as I
from inputs import DT, S

BINS = [0.0, 0.10, 0.20, 0.30, 0.45, 1.0]
N_BOOT, SEED = 2000, 0

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
a = ap.parse_args()


def contributions(t):
    rows = []
    for cut, tc in t.groupby("cut"):
        tc = tc.reset_index(drop=True)
        A, B, _ = DT.halves(tc)
        df = pd.DataFrame({"problem_id": tc.problem_id, "trace_id": tc.trace_id, "cut": cut, "A": A, "B": B})
        df["fixed"] = ((df.A - df.groupby("problem_id").A.transform("mean"))
                       * (df.B - df.groupby("problem_id").B.transform("mean")))
        df["total"] = df.problem_id.map(DT.total_sums(tc)) / df.groupby("problem_id").trace_id.transform("size")
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def by_bin(df, col):
    pids = np.array(sorted(df.problem_id.unique()))
    draws = S.problem_draws(len(pids), N_BOOT, SEED)
    out = []
    for key, g in df.groupby(col):
        f = g.groupby("problem_id").fixed.sum().reindex(pids, fill_value=0).values
        tt = g.groupby("problem_id").total.sum().reindex(pids, fill_value=0).values
        lo, hi = S.ci((draws @ f) / (draws @ tt))
        out.append({"bin": key, "D": f.sum() / tt.sum(), "lo": lo, "hi": hi, "traces": len(g),
                    "problems": g.problem_id.nunique(), "mean_position": g[col.replace("bin", "position")].mean()})
    return pd.DataFrame(out)


rows = []
for model in I.LABELS:
    tr, t = I.bank(model, cuts=(512, 2048, 4096))
    t = I.both_outcomes(t, tr)
    n = tr.n_gen
    loo = {}
    for _, g in tr.groupby("problem_id"):
        for tid in g.index:
            others = g.n_gen.drop(tid)
            loo[tid] = float(others.median()) if len(others) else np.nan
    c = contributions(t)
    c["position"] = c.cut / c.trace_id.map(n)
    c["position_expected"] = c.cut / c.trace_id.map(loo)
    if (c.position >= 1).any():
        raise SystemExit(f"{model}: a cut at or past its source's end")
    c["bin"] = pd.cut(c.position, BINS, right=False).astype(str)
    c["bin_expected"] = pd.cut(c.position_expected.clip(upper=0.999), BINS, right=False).astype(str)
    for pdef, col in (("eventual", "bin"), ("expected", "bin_expected")):
        b = by_bin(c, col)
        b.insert(0, "model", I.LABELS[model])
        b.insert(1, "position_def", pdef)
        rows.append(b)
    print(f"{I.LABELS[model]}: {t.problem_id.nunique()} problems, D over all cuts {c.fixed.sum() / c.total.sum():.3f}")
out = pd.concat(rows, ignore_index=True)
out.to_csv(a.out, index=False)
print(out.round(3).to_string(index=False))
