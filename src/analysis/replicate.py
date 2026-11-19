# D, readout capture, re-run capture and re-run AUROC at the 4,096 cut for one model (three-model table, Figure 4).
# the cohort keeps problems whose cohort cuts include both source outcomes; --all-problems also writes every
# re-run problem. Qwen3 and Ministral reuse the DeepSeek code unchanged: no explore / holdout split.
#   python src/analysis/replicate.py --model qwen3 --out results/qwen3
#   python src/analysis/replicate.py --model ministral --limit-hits wrong --out results/ministral_limit
import argparse, os
import pandas as pd

import inputs as I
from inputs import CR, DT
import forward_capture as FC

ap = argparse.ArgumentParser()
ap.add_argument("--model", choices=list(I.LABELS), required=True)
ap.add_argument("--limit-hits", choices=["drop", "wrong"], default="drop",
                help="sources that reach the 32,768-token limit: left out, or kept and counted as wrong")
ap.add_argument("--all-problems", action="store_true")
ap.add_argument("--out", required=True)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

tr, t = I.bank(a.model, a.limit_hits)
meta, acts = I.states(a.model)
y = (~tr.is_correct.astype(bool)).astype(int)
cohorts = [("pairs", I.both_outcomes(t, tr))] + ([("all", t)] if a.all_problems else [])
summary = []
for name, s in cohorts:
    print(f"\n==== {a.model} {name}: {len(s)} cuts, {s.problem_id.nunique()} problems", flush=True)
    stem = f"{a.model}_{name}"
    x1 = DT.run(s, tr, meta, acts)
    x1.to_csv(f"{a.out}/x1_{stem}.csv", index=False)
    FC.run(s).to_csv(f"{a.out}/forward_{stem}.csv", index=False)
    x2 = CR.run(s, y)
    x2["mean_source_tokens"] = float(tr.loc[s.trace_id.unique(), "n_gen"].mean())
    x2.to_csv(f"{a.out}/x2_{stem}.csv", index=False)
    d = x1[x1.what == "determined"].iloc[0]
    cap = x1[x1.what.str.startswith("capture:") & ~x1.what.str.contains("oracle")]
    full = x2[(x2.score == "correct") & (x2.K == 8) & (x2.L == 32768)].iloc[0]
    summary.append({"model": a.model, "limit_hits": a.limit_hits, "cohort": name, "problems": s.problem_id.nunique(),
                    "cuts": len(s), "D": d.est, "D_lo": d.lo, "D_hi": d.hi, "max_free_capture": cap.est.max(),
                    "max_free_capture_hi": cap.hi.max(), "auroc_rhat8": full.auroc, "auroc_lo": full.lo,
                    "auroc_hi": full.hi})
out = pd.DataFrame(summary)
out.to_csv(f"{a.out}/summary_{a.model}.csv", index=False)
print(out.round(3).to_string(index=False))
