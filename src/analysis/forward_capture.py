# capture of D by monitors that run the model forward. samples 0-1 of each prefix make
# the monitor (capped at L), samples 2-4 and 5-7 are the two halves that estimate the true r, so the monitor is
# never scored against its own noise. monitors: share of the 2 that finish within L (gold-free, finish time)
# and share correct within L (gold). capture as in determination.py.
#   python src/analysis/forward_capture.py --rollouts data/recovery_ceiling --split results/split.json \
#       --half explore --out <dir>
import argparse, json, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "probe_cmp"))
import stats as S  # noqa: E402
from cheap_rollouts import load  # noqa: E402
from determination import sums  # noqa: E402

MON, HA, HB = [0, 1], [2, 3, 4], [5, 6, 7]
LS = (2048, 4096, 32768)


def run(t, n_boot=2000, seed=0, log=print):
    rows = []
    for cut in sorted(t.cut.unique()):
        tc = t[t.cut == cut].reset_index(drop=True)
        tc = tc[tc.groupby("problem_id").trace_id.transform("size") >= 2].reset_index(drop=True)
        g = tc.problem_id.values
        ok, fin, n = (np.stack(tc[c].values) for c in ("ok", "fin", "n"))
        A, B = ok[:, HA].mean(1), ok[:, HB].mean(1)
        fix = sums(g, A, B)
        pids = fix.index.values
        fix = fix.loc[pids].values
        draws = S.problem_draws(len(pids), n_boot, seed)
        for L in LS:
            within = n[:, MON] <= L
            for name, x in (("finished", (fin[:, MON] & within).mean(1)), ("correct", (ok[:, MON] & within).mean(1))):
                sa, sb, sf = (sums(g, x, h).loc[pids].values for h in (A, B, x))
                cap = sa.sum() * sb.sum() / (sf.sum() * fix.sum())
                capb = (draws @ sa) * (draws @ sb) / ((draws @ sf) * (draws @ fix))
                tok = 2 * float(np.minimum(n[:, MON], L).mean(1).mean())
                rows.append({"cut": int(cut), "monitor": f"2 runs, {name} within {L}", "tokens": tok,
                             "capture": cap, "lo": S.ci(capb)[0], "hi": S.ci(capb)[1], "problems": len(pids)})
                log(f"{cut} {name} L{L}: capture {cap:.3f} [{S.ci(capb)[0]:.3f}, {S.ci(capb)[1]:.3f}], {tok:,.0f} tokens")
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    for k in ("rollouts", "split", "out"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--half", choices=["explore", "holdout"], required=True)
    a = ap.parse_args()
    t = load(a.rollouts)
    t = t[t.problem_id.isin(set(json.load(open(a.split))[a.half]))]
    res = run(t)
    res["half"] = a.half
    os.makedirs(a.out, exist_ok=True)
    res.to_csv(os.path.join(a.out, f"forward_{a.half}.csv"), index=False)


if __name__ == "__main__":
    main()
