# representation test, step 2 (CPU). is the null probe result a readout problem? on the ceiling
# population (identical traces to the r_hat ceiling), the same probe is fitted on three views of one
# layer at each rollout cut:
#   last      the last token's state (what every earlier probe read)
#   mean128   the mean over the trailing 128 tokens
#   delta     the change since the previous probe_cmp checkpoint (512-256, 2048-1024, 4096-2048), for
#             both last and mean128
# probe: probe_cmp's lr/raw and lr/svd50 arms (fold-local scaling / SVD, C by inner grouped CV),
# probe_cmp's 5-fold grouped folds, 3 seeds averaged per trace. scored as ceiling.py does: within-
# problem pair AUROC, P(wrong), problem bootstrap with draws shared by every score at a cut, so each
# delta (view - last, r_hat - view) is paired. the views were chosen after the null was seen, so every
# number here is exploratory.
#   python src/probe_cmp/reps.py --reps <dir>/reps.npz --rollouts <recovery_ceiling> \
#       --traces data/traces_B.jsonl [--preds <probe_cmp out>/preds] --out <dir>
import argparse, os
import numpy as np
import pandas as pd

import models as M
import stats as S
from ceiling import CKPTS, load_preds, load_rollouts, load_traces
from run import SEEDS, fold_indices, make_folds

PREV = {512: 256, 2048: 1024, 4096: 2048}
ARMS = ["raw", "svd50"]
N_BOOT = 2000


def views(R, layer):
    # every view at every rollout cut: {(view, cut): (n, dim) float32}
    cuts = R["cuts"].tolist()
    li = R["layers"].tolist().index(layer)
    at = lambda key, c: R[key][:, cuts.index(c), li, :].astype(np.float32)
    out = {}
    for c in PREV:
        out[("last", c)] = at("last", c)
        out[("mean128", c)] = at("mean", c)
        out[("delta last", c)] = at("last", c) - at("last", PREV[c])
        out[("delta mean128", c)] = at("mean", c) - at("mean", PREV[c])
    return out


def fit(X, y, groups, folds, seeds):
    # seed-averaged out-of-fold P(wrong) per arm
    res = {}
    for arm in ARMS:
        ps = []
        for s in seeds:
            p, _ = M.oof(M.Hidden(X, groups, seed=s), arm, "lr", y, groups, folds[s], s)
            ps.append(p)
        res[f"lr/{arm}"] = np.mean(ps, 0)
    return res


def score_table(scores, y, groups, n_boot=N_BOOT):
    # AUROC, CI, p against 0.5, and paired deltas against the last-token probe and against r_hat
    pairs = S.Pairs(y, groups)
    draws = S.problem_draws(len(pairs.pids), n_boot, seed=0)
    boot, rows = {}, []
    for name, s in scores.items():
        boot[name], _ = S.boot_within(pairs, s, draws)
        lo, hi = S.ci(boot[name])
        rows.append({"score": name, "auroc": pairs.auroc(s)["weighted"], "ci_low": lo, "ci_high": hi,
                     "p_vs_0.5": S.boot_p(boot[name] - 0.5), "n_pairs": int(pairs.pairs.sum()),
                     "n_problems": len(pairs.pids)})
    out = pd.DataFrame(rows).set_index("score")
    for ref in ("last lr/raw", "last lr/svd50", "r_hat"):
        if ref not in boot:
            continue
        for name in scores:
            d = boot[name] - boot[ref]
            out.loc[name, f"minus {ref}"] = out.at[name, "auroc"] - out.at[ref, "auroc"]
            out.loc[name, f"minus {ref} ci"] = "–" if name == ref else "[{:+.3f}, {:+.3f}]".format(*S.ci(d))
            out.loc[name, f"minus {ref} p"] = np.nan if name == ref else S.boot_p(d)
    return out.reset_index()


def run(R, roll, traces, preds_dir=None, layer=20, seeds=SEEDS, out=None, log=print, n_boot=N_BOOT):
    ids = R["trace_id"].astype(str)
    pop = traces.loc[ids, ["trace_id", "problem_id", "y"]].reset_index(drop=True)
    if pop.groupby("problem_id").y.nunique().min() < 2:
        raise SystemExit("a problem in reps.npz lacks a correct or a wrong trace: not the ceiling population")
    y, groups = pop.y.values, pop.problem_id.values
    fmap = make_folds(pop, seeds)  # probe_cmp's own fold construction on this population
    folds = {s: fold_indices(pop, fmap[str(s)]) for s in seeds}
    V = views(R, layer)
    tables = []
    for c in PREV:
        rc = next(k for k, v in CKPTS.items() if int(v) == c)
        r = roll[roll.checkpoint == rc].set_index("trace_id")
        keep = np.array([t in r.index for t in ids])  # ceiling.py had 447 of 448 at 4096
        if not keep.all():
            log(f"{c}: {int((~keep).sum())} trace(s) without rollouts here, left out of every score: "
                f"{list(ids[~keep])}")
        scores = {}
        for v in ("last", "mean128", "delta last", "delta mean128"):
            # fitted on the full population (the folds stay probe_cmp's); scored on traces with rollouts
            for arm, p in fit(V[(v, c)], y, groups, folds, seeds).items():
                scores[f"{v} {arm}"] = p[keep]
        scores["r_hat"] = 1 - r.loc[ids[keep], "r"].values
        if preds_dir:
            p = load_preds(os.path.join(preds_dir, f"{c}.parquet"))
            for arm, lay, name in (("len", -999, "len (probe_cmp)"), ("tfidf", -999, "tfidf (probe_cmp)"),
                                   ("lr/raw", layer, f"last lr/raw L{layer} (probe_cmp extraction)")):
                a = p[(p.arm == arm) & (p.layer == lay)].set_index("trace_id")
                if len(a):
                    scores[name] = a.loc[ids[keep], "p"].values
        t = score_table(scores, y[keep], groups[keep], n_boot)
        t.insert(0, "checkpoint", c)
        tables.append(t)
        log(f"{c}: " + ", ".join(f"{n} {v:.3f}" for n, v in zip(t.score, t.auroc)))
    res = pd.concat(tables, ignore_index=True)
    # Holm across the probe cells (view x cut) of each arm, for "above chance"
    for arm in ARMS:
        m = res.score.str.endswith(f" lr/{arm}")
        res.loc[m, "holm_vs_0.5"] = S.holm(res.loc[m, "p_vs_0.5"].values)
    if out:
        os.makedirs(out, exist_ok=True)
        res.to_csv(os.path.join(out, "reps_auroc.csv"), index=False)
        md = report(res, layer)
        open(os.path.join(out, "reps_summary.md"), "w").write(md)
        log("\n" + md)
    return res


def report(res, layer):
    f = lambda r: f"{r['auroc']:.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}]"
    d = lambda r, ref: ("–" if pd.isna(r.get(f"minus {ref} p")) else
                        f"{r[f'minus {ref}']:+.3f} {r[f'minus {ref} ci']}")
    lines = [f"# Representation test, layer {layer}\n",
             "> **Exploratory.** The views were chosen after the last-token null was seen.\n",
             "- Same traces as the r_hat ceiling; within-problem pair AUROC, P(wrong); problem bootstrap shared "
             "by every row of a checkpoint, so deltas are paired. Probe: probe_cmp's lr arms, C by inner grouped "
             "CV, its 5-fold grouped folds, 3 seeds averaged.",
             "- *delta* = state at the cut minus the state at the previous probe_cmp checkpoint "
             "(512-256, 2048-1024, 4096-2048).\n",
             "| checkpoint | score | AUROC [95% CI] | Holm p vs 0.5 | − last (same arm) [95% CI] | r_hat − score |",
             "|---|---|---|---|---|---|"]
    for r in res.to_dict("records"):
        arm = r["score"].split()[-1]
        ref = f"last {arm}" if arm in ("lr/raw", "lr/svd50") and f"minus last {arm}" in r else None
        holm = "–" if pd.isna(r.get("holm_vs_0.5")) else f"{r['holm_vs_0.5']:.3f}"
        to_r = "–" if r["score"] == "r_hat" else f"{-r['minus r_hat']:+.3f}"
        lines.append(f"| {r['checkpoint']} | {r['score']} | {f(r)} | {holm} | "
                     f"{d(r, ref) if ref else '–'} | {to_r} |")
    probes = res[res.score.str.contains(" lr/")]
    best = probes.loc[probes.auroc.idxmax()]
    sig = probes[probes["holm_vs_0.5"] < 0.05]
    lines += ["", "## In plain words\n",
              f"- Best probe cell: {best.score} at {best.checkpoint}, AUROC {best.auroc:.3f} "
              f"[{best.ci_low:.3f}, {best.ci_high:.3f}]; r_hat at that cut "
              f"{res[(res.checkpoint == best.checkpoint) & (res.score == 'r_hat')].auroc.iloc[0]:.3f}.",
              f"- Probe cells above 0.5 after Holm: {len(sig)} of {len(probes)}"
              + (": " + ", ".join(f"{r.score} @ {r.checkpoint}" for r in sig.itertuples()) if len(sig) else "")
              + ".",
              "- If no view beats the last token and all sit far below r_hat, the gap is not a pooling or "
              "single-token artifact of the readout at this layer; if one does, that view is the one to "
              "pre-register for the confirmation set."]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", required=True, help="reps.npz from extract_reps.py")
    ap.add_argument("--rollouts", required=True)
    ap.add_argument("--traces", default="data/traces_B.jsonl")
    ap.add_argument("--preds", default=None, help="optional preds/ of the <=4096 probe_cmp run: len, tfidf")
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    R = dict(np.load(a.reps))
    roll, torn = load_rollouts(a.rollouts)
    print(f"reps: {len(R['trace_id'])} traces, layers {R['layers'].tolist()}, cuts {R['cuts'].tolist()}; "
          f"rollouts: {len(roll)} prefixes, torn lines {torn}")
    run(R, roll, load_traces(a.traces), a.preds, a.layer, out=a.out)


if __name__ == "__main__":
    main()
