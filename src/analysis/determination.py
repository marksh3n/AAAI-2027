# outcome determination D and linear capture (determination and capture tables, Figure 1): how much of the
# within-problem outcome variance is already fixed by the prefix at 512 / 2048 / 4096, and how much of that fixed part each prefix-only predictor captures.
# per problem p, traces i, rollout halves A (samples 0-3) and B (4-7) of prefix i:
#   fixed part   Cov_p(r_A, r_B) across p's traces: the variance of the true r_i, since the halves' noise is
#                independent given the prefix (no binomial correction needed)
#   total        q(1-q) N/(N-1) over p's N pooled continuations: the variance of one continuation's outcome
#   determined   D = sum_p S_fixed / sum_p S_total, each S a within-problem sum of products (ratio of sums;
#                negatives kept). by the law of total variance, D is the within-problem R^2 ceiling for any
#                prefix-only predictor of a ROLLOUT outcome (not of the source outcome)
#   capture(f)   Cov(f, r_A) Cov(f, r_B) / (Var f  Cov(r_A, r_B)), pooled within problem: f's squared
#                correlation with the true r_i (one global slope), unbiased in the numerator because the two
#                halves' noise is independent of f. capture * D = share of total variance f explains
# predictors, all from the prefix only: len (log1p paragraphs), restarts, committed (</think> or \boxed),
# probe_y (lr/raw L20 fit on source outcomes, grouped out-of-fold over all dev problems), probe_r (ridge on
# L20 fit to within-problem-centered r_hat of OTHER cohort problems, so never scored against its own noise).
#   python src/analysis/determination.py --rollouts data/recovery_ceiling --traces <traces_B.jsonl> \
#       --meta <meta_B.parquet> --acts <acts_B.npy> --split results/split.json --half explore --out <dir>
import argparse, json, os, sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "probe_cmp"))
import stats as S  # noqa: E402
from data import paragraphs  # noqa: E402
from cheap_rollouts import load  # noqa: E402

LAYER_SLOT = 6  # layer 20 in acts_B's (-1, 0, 4, 8, 12, 16, 20, 24, 27)
CS = (1e-4, 1e-3, 1e-2, 1e-1)
ALPHAS = (1e1, 1e2, 1e3, 1e4)
PREDICTORS = ("len", "restarts", "committed", "probe_y", "probe_r", "probe_len", "oracle_remaining")
# oracle_remaining = the source trace's own log remaining tokens: uses the future, a diagnostic, never a monitor


def sums(g, x, y):
    # per-problem within-problem sum of products of x and y, and degrees of freedom
    df = pd.DataFrame({"g": g, "x": x, "y": y})
    df["x"] -= df.groupby("g").x.transform("mean")
    df["y"] -= df.groupby("g").y.transform("mean")
    return (df.x * df.y).groupby(df.g).sum()


def halves(t):
    ok = np.stack(t.ok.values).astype(float)
    return ok[:, :4].mean(1), ok[:, 4:].mean(1), ok.mean(1)


def total_sums(t):
    # (n_traces - 1) * unbiased Bernoulli variance of one continuation, per problem, so it sits on the same
    # footing as the (n - 1)-weighted covariance sums
    ok = np.stack(t.ok.values).astype(float)
    out = {}
    for p, idx in t.groupby("problem_id").indices.items():
        v, n = ok[idx].ravel(), ok[idx].size
        q = v.mean()
        out[p] = (len(idx) - 1) * q * (1 - q) * n / (n - 1)
    return pd.Series(out)


def inner_pick(X, y, g, make, grid, score):
    best, best_s = grid[0], -np.inf
    for v in grid:
        s = []
        for tr, va in GroupKFold(3).split(X, y, g):
            m = make(v).fit(X[tr], y[tr])
            s.append(score(m, X[va], y[va]))
        if np.mean(s) > best_s:
            best, best_s = v, np.mean(s)
    return best


def probe_y(X, y, g):
    # grouped 5-fold out-of-fold logit of P(wrong), C by inner grouped CV on log loss
    out = np.zeros(len(y))
    make = lambda c: LogisticRegression(C=c, max_iter=2000)
    ll = lambda m, X, y: -np.mean(np.log(np.clip(m.predict_proba(X)[np.arange(len(y)), y], 1e-9, 1)))
    for tr, te in GroupKFold(5).split(X, y, g):
        sc = StandardScaler().fit(X[tr])
        c = inner_pick(sc.transform(X[tr]), y[tr], g[tr], make, CS, ll)
        out[te] = make(c).fit(sc.transform(X[tr]), y[tr]).decision_function(sc.transform(X[te]))
    return out


def probe_r(X, r, g):
    # grouped 5-fold ridge on within-problem-centered features and r_hat; predictions for held-out problems
    Xc = X - pd.DataFrame(X).groupby(g).transform("mean").values
    rc = r - pd.Series(r).groupby(g).transform("mean").values
    out = np.zeros(len(r))
    make = lambda a: Ridge(alpha=a)
    r2 = lambda m, X, y: -np.mean((m.predict(X) - y) ** 2)
    for tr, te in GroupKFold(5).split(Xc, rc, g):
        sc = StandardScaler().fit(Xc[tr])
        a = inner_pick(sc.transform(Xc[tr]), rc[tr], g[tr], make, ALPHAS, r2)
        out[te] = make(a).fit(sc.transform(Xc[tr]), rc[tr]).predict(sc.transform(Xc[te]))
    return out


def probe_len(X, target, g):
    # grouped 5-fold ridge predicting log remaining tokens, alpha by inner grouped CV
    out = np.zeros(len(target))
    make = lambda a: Ridge(alpha=a)
    mse = lambda m, X, y: -np.mean((m.predict(X) - y) ** 2)
    for tr, te in GroupKFold(5).split(X, target, g):
        sc = StandardScaler().fit(X[tr])
        a = inner_pick(sc.transform(X[tr]), target[tr], g[tr], make, ALPHAS, mse)
        out[te] = make(a).fit(sc.transform(X[tr]), target[tr]).predict(sc.transform(X[te]))
    return out


def features(tc, tr, meta, acts, cut, all_traces):
    # prefix-only predictors for the cohort rows tc at this cut
    m = meta.set_index(["trace_id", "pos"])
    pre = [m.at[(i, cut), "prefix"] for i in tc.trace_id]
    f = {"len": np.log1p([paragraphs(x) for x in pre]),
         "restarts": np.array([m.at[(i, cut), "restarts"] for i in tc.trace_id], float),
         "committed": np.array([("</think>" in x) or ("\\boxed" in x) for x in pre], float)}
    # probe_y: fit on every dev trace at this cut (all problems), out of fold by problem
    rows = meta[meta.pos == cut]
    rows = rows[rows.trace_id.isin(all_traces)]
    X = acts[rows.row.values, LAYER_SLOT].astype(np.float32)
    y = (~tr.loc[rows.trace_id, "is_correct"].astype(bool)).astype(int).values
    oof = pd.Series(probe_y(X, y, rows.problem_id.values), index=rows.trace_id.values)
    f["probe_y"] = oof.loc[tc.trace_id].values
    # the same scores over every trace at the cut: pooled AUROC (rewards difficulty) next to within-problem AUROC
    g_all = rows.problem_id.values
    pids_all = np.unique(g_all)
    boot = np.random.default_rng(0)
    pooled = []
    for _ in range(500):
        idx = np.concatenate([np.where(g_all == p)[0] for p in boot.choice(pids_all, len(pids_all))])
        if 0 < y[idx].sum() < len(idx):
            pooled.append(roc_auc_score(y[idx], oof.values[idx]))
    f["_probe_y_all"] = {"pooled": roc_auc_score(y, oof.values), "pooled_lo": np.percentile(pooled, 2.5),
                         "pooled_hi": np.percentile(pooled, 97.5),
                         "within": S.Pairs(y, g_all).auroc(oof.values)["weighted"], "traces": len(y),
                         "problems": len(pids_all)}
    rem = np.log1p(np.maximum(tr.loc[rows.trace_id, "n_gen"].values - cut, 0).astype(float))
    oof_len = pd.Series(probe_len(X, rem, rows.problem_id.values), index=rows.trace_id.values)
    f["probe_len"] = oof_len.loc[tc.trace_id].values
    f["oracle_remaining"] = np.log1p(np.maximum(tr.loc[tc.trace_id, "n_gen"].values - cut, 0).astype(float))
    # how well probe_len tracks the remaining length it was trained on, within problem (all dev traces)
    pr = pd.DataFrame({"g": rows.problem_id.values, "p": oof_len.values, "r": rem})
    pr[["p", "r"]] -= pr.groupby("g")[["p", "r"]].transform("mean")
    f["_probe_len_wp_corr"] = float(np.corrcoef(pr.p, pr.r)[0, 1])
    Xc = acts[m.loc[[(i, cut) for i in tc.trace_id], "row"].values, LAYER_SLOT].astype(np.float32)
    f["probe_r"] = probe_r(Xc, halves(tc)[2], tc.problem_id.values)
    return f


def run(t, tr, meta, acts, n_boot=2000, seed=0, log=print):
    rows = []
    for cut in sorted(t.cut.unique()):
        tc = t[t.cut == cut].reset_index(drop=True)
        tc = tc[tc.groupby("problem_id").trace_id.transform("size") >= 2].reset_index(drop=True)
        g = tc.problem_id.values
        A, B, _ = halves(tc)
        s_fix, s_tot = sums(g, A, B), total_sums(tc)
        pids = s_fix.index.values
        draws = S.problem_draws(len(pids), n_boot, seed)
        fix, tot = s_fix.loc[pids].values, s_tot.loc[pids].values
        D, Db = fix.sum() / tot.sum(), (draws @ fix) / (draws @ tot)
        rows.append({"cut": int(cut), "what": "determined", "est": D, "lo": S.ci(Db)[0], "hi": S.ci(Db)[1],
                     "traces": len(tc), "problems": len(pids)})
        log(f"{cut}: D {D:.3f} {S.ci(Db)}")
        f = features(tc, tr, meta, acts, cut, set(tr.index))
        yc = (~tr.loc[tc.trace_id, "is_correct"].astype(bool)).astype(int).values
        pairs = S.Pairs(yc, g)
        log(f"  probe_len within-problem corr with remaining length (all dev traces): {f['_probe_len_wp_corr']:.3f}")
        pa = f["_probe_y_all"]
        rows.append({"cut": int(cut), "what": "probe_y auroc, all traces", "est": pa["pooled"], "lo": pa["pooled_lo"],
                     "hi": pa["pooled_hi"], "wp_auroc_y": pa["within"], "traces": pa["traces"],
                     "problems": pa["problems"]})
        log(f"  probe_y on all traces at the cut: pooled AUROC {pa['pooled']:.3f} "
            f"[{pa['pooled_lo']:.3f}, {pa['pooled_hi']:.3f}], within problem {pa['within']:.3f}")
        for name in PREDICTORS:
            x = f[name]
            sa, sb, sf = (sums(g, x, h).loc[pids].values for h in (A, B, x))
            cap = sa.sum() * sb.sum() / (sf.sum() * fix.sum()) if sf.sum() > 0 else np.nan
            capb = (draws @ sa) * (draws @ sb) / ((draws @ sf) * (draws @ fix)) if sf.sum() > 0 else np.full(n_boot, np.nan)
            au = pairs.auroc(x)["weighted"]
            rows.append({"cut": int(cut), "what": f"capture:{name}", "est": cap, "lo": S.ci(capb)[0],
                         "hi": S.ci(capb)[1], "share_of_total": cap * D, "wp_auroc_y": au,
                         "traces": len(tc), "problems": len(pids)})
            log(f"  {name}: capture {cap:.3f} {S.ci(capb)}, WP AUROC vs source outcome {au:.3f}")
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    for k in ("rollouts", "traces", "meta", "acts", "split", "out"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--half", choices=["explore", "holdout"], required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()
    t = load(a.rollouts)
    t = t[t.problem_id.isin(set(json.load(open(a.split))[a.half]))]
    tr = pd.DataFrame(json.loads(line) for line in open(a.traces)).set_index("trace_id")
    meta = pd.read_parquet(a.meta, columns=["trace_id", "problem_id", "pos", "prefix", "restarts", "row"])
    acts = np.load(a.acts, mmap_mode="r")
    res = run(t, tr, meta, acts, a.n_boot)
    res["half"] = a.half
    os.makedirs(a.out, exist_ok=True)
    res.to_csv(os.path.join(a.out, f"x1_{a.half}.csv"), index=False)


if __name__ == "__main__":
    main()
