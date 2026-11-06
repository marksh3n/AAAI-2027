# the confirmation analysis, frozen in spec.py and docs/CONFIRMATION_SPEC.md. CPU only, no model.
# input: a bundle directory (bundle.py writes one from the GPU outputs, dev_bundle.py from the
# development data, same schema), plus the frozen monitors' scores (score.py):
#   sources.parquet   trace_id, problem_id, n (generated tokens, EOS included), correct (0/1), answer_key
#   cuts.parquet      trace_id, checkpoint, live, pre_answer (spec.PRE_ANSWER_PRIMARY), pre_answer_strict,
#                     pre_answer_think_boxed (the two pre-registered sensitivity rules)
#   bank.parquet      problem_id, trace_id ("" for t0), checkpoint (0 for t0), k, r, L, r_h0, r_h1, L_h0, L_h1
#   t0_votes.parquet  problem_id, sample_idx, answer_key, correct, n_gen
#   scores.parquet    trace_id, checkpoint, monitor, score, threshold
#   bundle.json       kind ("confirmation" | "development"), spec_sha256, provenance
# populations: accuracy endpoints are over ALL attempts of ALL problems (mixed-problem selection
# conditions on outcomes); within-problem AUROC is over the mixed problems among the rows scored,
# since other problems have no (correct, wrong) pairs. at each checkpoint every LIVE trace is
# evaluated (the available population); the matched cohort (live at every checkpoint) is reported
# separately for timing comparisons.
#   python src/confirm/analyze.py --bundle <dir> --out <dir>
import argparse, json, math, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "probe_cmp"))
import spec  # noqa: E402
import policy as P  # noqa: E402
import stats as S  # noqa: E402


def load(bundle):
    rd = lambda n: pd.read_parquet(os.path.join(bundle, n))
    meta = json.load(open(os.path.join(bundle, "bundle.json")))
    return meta, rd("sources.parquet"), rd("cuts.parquet"), rd("bank.parquet"), rd("t0_votes.parquet"), rd("scores.parquet")


def check(meta, src, cuts, bank, votes, scores, log=print):
    # returns the set of (trace, checkpoint) live cuts that cannot be evaluated (no bank). on
    # confirmation data any such cut, a spec mismatch or a duplicate is an error
    conf = meta["kind"] == "confirmation"
    problems = []
    if meta.get("spec_sha256") != spec.spec_sha():
        problems.append(f"bundle was written under spec {str(meta.get('spec_sha256'))[:16]}, this is {spec.spec_sha()[:16]}")
    if src.trace_id.duplicated().any():
        problems.append("duplicate trace_id in sources")
    if cuts.duplicated(["trace_id", "checkpoint"]).any():
        problems.append("duplicate (trace, checkpoint) in cuts")
    if scores.duplicated(["trace_id", "checkpoint", "monitor"]).any():
        problems.append("duplicate scores")
    noprob = set(src.problem_id) - set(bank[bank.checkpoint == 0].problem_id)
    if noprob:
        problems.append(f"{len(noprob)} problems have no t0 bank (restart and aggregation cannot be scored)")
    banked = set(zip(bank.trace_id, bank.checkpoint))
    missing = sorted(set(spec.CHECKPOINTS) - set(bank.checkpoint))
    if missing:
        problems.append(f"no rollout bank at checkpoints {missing}: no policy is evaluated there")
    live = cuts[cuts.live & cuts.checkpoint.isin(set(bank.checkpoint))]
    unbanked = {(t, c) for t, c in zip(live.trace_id, live.checkpoint) if (t, c) not in banked}
    if unbanked:
        problems.append(f"{len(unbanked)} live cuts have no rollout bank")
    if conf and problems:
        raise SystemExit("confirmation bundle fails its checks:\n  " + "\n  ".join(problems))
    for p in problems:
        log(f"  WARNING (development dry run): {p}")
    return unbanked


def attempts(src, cuts, bank, votes, c, unbanked, keep="outcome"):
    # one row per attempt at checkpoint c, every problem: outcome, cost, live flag and the banks
    t = src.set_index("trace_id")
    cc = cuts[cuts.checkpoint == c].set_index("trace_id")
    t["live"] = cc.live.reindex(t.index).fillna(False).astype(bool)
    for col in ("pre_answer", "pre_answer_strict", "pre_answer_think_boxed"):
        v = cc[col] if col in cc else cc.pre_answer  # older bundles hold the primary rule only
        t[col] = v.reindex(t.index).fillna(False).astype(bool)
    t["unbanked"] = [(i, c) in unbanked for i in t.index]
    t["live_all"] = t.live.copy()  # monitors are scored on every live trace
    t["live"] &= ~t.unbanked  # dev only: a live cut with no bank cannot be flagged
    b0 = bank[bank.checkpoint == 0].set_index("problem_id")
    t["r0"], t["L0"] = t.problem_id.map(b0.r), t.problem_id.map(b0.L)
    bc = bank[bank.checkpoint == c].set_index("trace_id")
    for col in ("r", "L", "r_h0", "r_h1", "L_h0", "L_h1"):
        t[col] = bc[col].reindex(t.index)
    prev = spec.BACKTRACK_TO.get(c)
    if prev is not None:
        bp = bank[bank.checkpoint == prev].set_index("trace_id")
        t["r_prev"], t["L_prev"] = bp.r.reindex(t.index), bp.L.reindex(t.index)
    if keep == "rollout":
        # sensitivity: a kept live trace is valued at its own prefix's bank mean instead of its outcome
        # (oracle.py's selection-free scoring); used when the resample validity check fails
        m = t.live & t.r.notna()
        t["correct"] = t.correct.astype(float)
        t.loc[m, "correct"] = t.loc[m, "r"]
    keys = votes.sort_values(["problem_id", "sample_idx"]).groupby("problem_id").answer_key.apply(list)
    for m in spec.AGG_EXTRA:
        t[f"A{m}"] = [P.aggregate_acc(k, keys[p], m) for k, p in zip(t.answer_key, t.problem_id)]
    return t.reset_index().sort_values(["problem_id", "trace_id"]).reset_index(drop=True)


def wald_p(est, boot):
    se = np.nanstd(boot, ddof=1)
    return 1.0 if not se else math.erfc(abs(est) / se / math.sqrt(2))


def summ(v):
    # (estimate, lo, hi) from a (B + 1,) vector whose row 0 is the estimate
    lo, hi = S.ci(v[1:])
    return float(v[0]), lo, hi


def policies(t, c, scores, draws, log=print):
    """every arm at checkpoint c. returns rows (one per arm x comparator) and, for the tests, the
    per-draw vectors keyed by (monitor, action)."""
    sums = P.Sums(t.problem_id.values, draws)
    base_acc, base_tok = t.correct.values.astype(float), t.n.values.astype(float)
    A_none, T_none = sums.mean(base_acc), sums.mean(base_tok)
    rows, vec = [], {}

    def row(**kw):
        rows.append({"checkpoint": c, **kw})

    row(arm="none", action="-", monitor="-", comparator="-", acc=summ(A_none), tokens=summ(T_none), delta=None,
        flag_share=0.0, pi=None)
    agg = {}
    for m in spec.AGG_EXTRA:
        gain = t[f"A{m}"].values - base_acc
        extra = m * t.L0.values
        agg[m] = (gain, extra)
        row(arm=f"aggregate m={m} (all attempts)", action="aggregate", monitor="-", comparator="-",
            acc=summ(sums.mean(base_acc + gain)), tokens=summ(sums.mean(base_tok + extra)), delta=None, flag_share=1.0, pi=1.0)
    actions = [a for a in spec.ACTIONS if a == "restart" or c in spec.BACKTRACK_TO]
    have = scores[scores.checkpoint == c]
    for action in actions:
        gain, extra = P.arm_effect(t, action, c)
        live = t.live.values
        row(arm=f"fixed {action} (every live trace)", action=action, monitor="-", comparator="-",
            acc=summ(sums.mean(base_acc + gain)), tokens=summ(sums.mean(base_tok + extra)), delta=None,
            flag_share=float(live.mean()), pi=1.0)
        for mon, g in have.groupby("monitor"):
            s = g.set_index("trace_id").score.reindex(t.trace_id).values
            tau = float(g.threshold.iloc[0])
            w = P.flags(np.nan_to_num(s, nan=-np.inf), tau, live).astype(float)
            pa, pt = base_acc + w * gain, base_tok + w * extra
            A_p, T_p = sums.mean(pa), sums.mean(pt)
            share = float(w.sum() / max(live.sum(), 1))
            row(arm=f"{mon}-triggered {action}", action=action, monitor=mon, comparator="-", acc=summ(A_p),
                tokens=summ(T_p), delta=None, flag_share=share, pi=None)
            row(arm=f"{mon}-triggered {action}", action=action, monitor=mon, comparator="none",
                acc=summ(A_p), tokens=summ(T_p), delta=summ(A_p - A_none), flag_share=share, pi=None)
            A_r, pi_r, st_r = P.matched(sums, base_acc, base_tok, pa, pt, gain, extra)
            row(arm=f"{mon}-triggered {action}", action=action, monitor=mon, comparator=f"random {action}, equal tokens",
                acc=summ(A_p), tokens=summ(T_p), delta=summ(A_p - A_r), flag_share=share, pi=float(pi_r[0]),
                match_status=int(st_r[0]), match_failed_draws=float(np.mean(st_r[1:] != 0)))
            vec[(mon, action, "random")] = A_p - A_r
            for m, (ag, ae) in agg.items():
                A_a, pi_a, st_a = P.matched(sums, base_acc, base_tok, pa, pt, ag, ae)
                row(arm=f"{mon}-triggered {action}", action=action, monitor=mon,
                    comparator=f"aggregate m={m}, equal tokens", acc=summ(A_p), tokens=summ(T_p),
                    delta=summ(A_p - A_a), flag_share=share, pi=float(pi_a[0]), match_status=int(st_a[0]),
                    match_failed_draws=float(np.mean(st_a[1:] != 0)))
                vec[(mon, action, f"aggregate{m}")] = A_p - A_a
        if action == "restart":
            # validity: a fresh continuation of the same prefix, for every live trace and for the primary
            # monitor's flags. ~0 when sources and rollouts share an engine (the dev gap did not vanish)
            rg, re_ = P.arm_effect(t, "resample", c)
            for who, w in (("every live trace", live.astype(float)),
                           (spec.PRIMARY_MONITOR, None)):
                if w is None:
                    g = have[have.monitor == spec.PRIMARY_MONITOR]
                    if not len(g):
                        continue
                    s = g.set_index("trace_id").score.reindex(t.trace_id).values
                    w = P.flags(np.nan_to_num(s, nan=-np.inf), float(g.threshold.iloc[0]), live).astype(float)
                A_v = sums.mean(base_acc + w * rg)
                row(arm=f"resample validity check ({who})", action="resample", monitor=who, comparator="none",
                    acc=summ(A_v), tokens=summ(sums.mean(base_tok + w * re_)), delta=summ(A_v - A_none),
                    flag_share=float(w.sum() / max(live.sum(), 1)), pi=None)
        if action == "restart" and t.r_h0.notna().any():
            acc, tok = P.oracle_restart(t.fillna({"r_h0": 1.0, "r_h1": 1.0, "L_h0": 0.0, "L_h1": 0.0}),
                                        spec.Q_DEV, spec.K_PREFIX // 2, c)
            row(arm=f"rollout-ranked restart, q={spec.Q_DEV} (diagnostic; includes {spec.K_PREFIX // 2} "
                    f"selection rollouts per live trace)", action=action, monitor="rollout r_hat", comparator="none",
                acc=summ(sums.mean(acc)), tokens=summ(sums.mean(tok)), delta=summ(sums.mean(acc) - A_none),
                flag_share=spec.Q_DEV, pi=None)
    return rows, vec


def aurocs(t, scores, c, n_boot, label):
    # within-problem AUROC of every monitor and of rollout-estimated r_hat over the rows in t
    # (mixed problems among them have pairs), paired deltas on shared draws
    y = 1 - t.correct.values.astype(int)  # 1 = wrong, every score is P(wrong)
    pairs = S.Pairs(y, t.problem_id.values)
    if not len(pairs.pids):
        return [], {}
    draws = S.problem_draws(len(pairs.pids), n_boot, spec.BOOT_SEED)
    sc = {m: g.set_index("trace_id").score.reindex(t.trace_id).values
          for m, g in scores[scores.checkpoint == c].groupby("monitor")}
    if t.r.notna().all():
        sc["rollout r_hat (1 - r)"] = 1 - t.r.values
    out, boots = [], {}
    for name, s in sc.items():
        if np.isnan(s).any():
            continue
        w, _ = S.boot_within(pairs, s, draws)
        boots[name] = w
        est = pairs.auroc(s)["weighted"]
        lo, hi = S.ci(w)
        out.append({"population": label, "checkpoint": c, "score": name, "auroc": est, "ci_low": lo, "ci_high": hi,
                    "traces": len(t), "mixed_problems": len(pairs.pids), "pairs": int(pairs.pairs.sum())})
    for r in out:
        for ref in ("len", "rollout r_hat (1 - r)"):
            if ref in boots and r["score"] != ref:
                ests = {o["score"]: o["auroc"] for o in out}
                d = boots[r["score"]] - boots[ref]
                lo, hi = S.ci(d)
                r[f"minus_{ref.split()[0]}"] = ests[r["score"]] - ests[ref]
                r[f"minus_{ref.split()[0]}_ci"] = [lo, hi]
                r[f"minus_{ref.split()[0]}_p"] = wald_p(ests[r["score"]] - ests[ref], d)
    return out, boots


def mixed_count(rows):
    g = rows.groupby("problem_id").correct.agg(lambda v: 0 < v.sum() < len(v))
    return int(g.sum())


def populations(src, cuts):
    rows = []
    ck = sorted(cuts.checkpoint.unique())
    live_all = cuts[cuts.live].groupby("trace_id").checkpoint.nunique()
    matched = set(live_all.index[live_all == len(ck)])
    for c in ck:
        cc = cuts[cuts.checkpoint == c].merge(src[["trace_id", "problem_id", "correct"]], on="trace_id")
        lv = cc[cc.live]
        mixed = lv.groupby("problem_id").correct.agg(lambda v: 0 < v.sum() < len(v))
        rows.append({"checkpoint": c, "attempts": len(src), "problems": src.problem_id.nunique(),
                     "live": int(cc.live.sum()), "live_problems": lv.problem_id.nunique(),
                     "live_mixed_problems": int(mixed.sum()),
                     **{f"live_{k}": int((cc.live & cc.get(k, cc.pre_answer)).sum())
                        for k in ("pre_answer", "pre_answer_strict", "pre_answer_think_boxed")},
                     **{f"mixed_{k}": mixed_count(cc[cc.live & cc.get(k, cc.pre_answer)])
                        for k in ("pre_answer", "pre_answer_strict", "pre_answer_think_boxed")},
                     "matched_cohort": len(matched), "live_accuracy": float(lv.correct.mean()) if len(lv) else np.nan})
    return pd.DataFrame(rows), matched


def tests(vec, auc_boot, auc_est, margin, s7_mixed=None):
    # primary: both token-matched comparators; secondary: Holm over spec.SECONDARY. s7_mixed: mixed problems
    # among live pre-answer cuts at S7's checkpoint; below spec.S7_MIN_MIXED, S7 is estimate-only (no p, not in Holm)
    # verdict: superiority vs 0, both lower 95% bounds above 0 (intersection-union). rule-out, per comparator:
    # a gain >= margin is excluded when the one-sided 95% limit (95th bootstrap percentile) is below the margin.
    # this replaces the Tango score test, which needs one binary outcome per arm per problem
    out = {}
    pr = spec.PRIMARY
    for comp in ("random", f"aggregate{spec.AGG_EXTRA[0]}"):
        v = vec.get((pr["checkpoint"], pr["monitor"], pr["action"], comp))
        if v is None:
            out[f"primary vs {comp}"] = None
            continue
        est, lo, hi = summ(v)
        out[f"primary vs {comp}"] = {"delta": est, "ci": [lo, hi], "p": wald_p(est, v[1:]),
                                     "one_sided_95": float(np.nanpercentile(v[1:], 95))}
    ok = [out[k] for k in out if k.startswith("primary")]
    if margin is None:
        verdict = (f"not evaluated: team slots are {spec.SLOTS.get('status')}, not frozen"
                   if spec.SLOTS.get("status") != "frozen" else "not evaluated: margin slot is empty")
    elif any(o is None for o in ok):
        verdict = "not evaluable on this bundle"
    else:
        verdict = "met" if all(o["ci"][0] > 0 for o in ok) else "not met"
        for o in ok:
            o["rules_out_margin"] = o["one_sided_95"] < margin / 100
    out["primary verdict"] = verdict
    sec = []
    for s in spec.SECONDARY:
        if s.get("kind") == "auroc_pre_answer":
            key = (s["checkpoint"], "pre_answer")
            b = auc_boot.get(key, {})
            if s["monitor"] in b and s["vs"] in b:
                d = b[s["monitor"]] - b[s["vs"]]
                est = auc_est[key][s["monitor"]] - auc_est[key][s["vs"]]
                lo, hi = S.ci(d)
                if s7_mixed is None or s7_mixed < spec.S7_MIN_MIXED:
                    sec.append({**s, "delta": est, "ci": [lo, hi], "estimate_only": True, "mixed": s7_mixed})
                else:
                    sec.append({**s, "delta": est, "ci": [lo, hi], "p": wald_p(est, d), "mixed": s7_mixed})
            else:
                sec.append({**s, "delta": None})
            continue
        if s.get("kind") == "timeout":
            v = vec.get(("timeout", s["T"], s["R"]))
            if v is None or not np.isfinite(v).all():
                sec.append({**s, "delta": None})
                continue
            est, lo, hi = summ(v)
            sec.append({**s, "delta": est, "ci": [lo, hi], "p": wald_p(est, v[1:])})
            continue
        if s["vs"] == "random":
            v = vec.get((s["checkpoint"], s["monitor"], s["action"], "random"))
        else:  # value over the token-matched random line, this monitor minus the baseline monitor
            a = vec.get((s["checkpoint"], s["monitor"], s["action"], "random"))
            b = vec.get((s["checkpoint"], s["vs"], s["action"], "random"))
            v = None if a is None or b is None else a - b
        if v is None:
            sec.append({**s, "delta": None})
            continue
        est, lo, hi = summ(v)
        sec.append({**s, "delta": est, "ci": [lo, hi], "p": wald_p(est, v[1:])})
    have = [r for r in sec if r.get("delta") is not None and not r.get("estimate_only")]
    if have:
        for r, h in zip(have, S.holm([r["p"] for r in have])):
            r["p_holm"] = float(h)
    out["secondary"] = sec
    return out


def hull(x, y):
    # upper concave envelope of the points (x, y), as sorted vertex arrays
    pts = sorted(zip(x, y))
    up = []
    for p in pts:
        while len(up) >= 2 and (up[-1][0] - up[-2][0]) * (p[1] - up[-2][1]) - (up[-1][1] - up[-2][1]) * (p[0] - up[-2][0]) >= 0:
            up.pop()
        up.append(p)
    return np.array([u[0] for u in up]), np.array([u[1] for u in up])


def timeout_vec(src, votes, draws, grid=((4096, 1), (8192, 1), (8192, 5), (12288, 5))):
    # S8 and its descriptive grid: per attempt, timeout-restart accuracy and tokens, and the vote frontier (own
    # answer + m bank samples, exact over m-subsets; tokens n + m * mean bank length). per draw, the policy minus
    # the frontier interpolated at the policy's tokens (nan outside the frontier's range)
    sums = P.Sums(src.problem_id.values, draws)
    rng = np.random.default_rng(spec.BOOT_SEED)
    bank = {p: d.sort_values("sample_idx") for p, d in votes.groupby("problem_id")}
    orders = {p: [rng.permutation(len(d)) for _ in range(spec.TIMEOUT_ORDERS)] for p, d in bank.items()}
    ms, ks = range(0, 5), range(2, 6)
    fa = {m: np.zeros(len(src)) for m in ms}
    ft = {m: np.zeros(len(src)) for m in ms}
    sa = {k: np.zeros(len(src)) for k in ks}
    st = {k: np.zeros(len(src)) for k in ks}
    pol = {g: (np.zeros(len(src)), np.zeros(len(src))) for g in grid}
    for i, r in enumerate(src.itertuples()):
        b = bank[r.problem_id]
        keys, bn, bok = list(b.answer_key), b.n_gen.values.astype(float), b.correct.values.astype(float)
        for m in ms:
            fa[m][i] = P.aggregate_acc(r.answer_key, keys, m) if m else float(r.correct)
            ft[m][i] = r.n + m * bn.mean()
        for k in ks:
            sa[k][i], st[k][i] = P.short_first(float(r.n), float(r.correct), bn, bok, k, orders[r.problem_id], spec.MAX_NEW)
        for T, R in grid:
            pol[(T, R)][0][i], pol[(T, R)][1][i] = P.timeout_restart(float(r.n), float(r.correct), bn, bok, T, R,
                                                                     orders[r.problem_id])
    FA = np.stack([sums.mean(fa[m]) for m in ms], 1)
    FT = np.stack([sums.mean(ft[m]) for m in ms], 1)
    SA = np.stack([sums.mean(sa[k]) for k in ks], 1)
    ST = np.stack([sums.mean(st[k]) for k in ks], 1)

    def best(b, t):
        # best no-monitor baseline at t tokens: the upper concave envelope of every vote and short-1@k point, since
        # running one policy on a random share of attempts and another on the rest reaches any point between them
        return np.interp(t, *hull(np.r_[FT[b], ST[b]], np.r_[FA[b], SA[b]]), left=np.nan, right=np.nan)

    rows, vec = [], {}
    for (T, R), (a, t) in pol.items():
        A_, T_ = sums.mean(a), sums.mean(t)
        d = np.array([A_[b] - best(b, T_[b]) for b in range(len(A_))])
        vec[("timeout", T, R)] = d
        lo, hi = S.ci(d[1:]) if np.isfinite(d[1:]).mean() > 0.9 else (np.nan, np.nan)
        rows.append({"T": T, "R": R, "acc": A_[0], "tokens": T_[0], "vs_best_baseline": d[0], "lo": lo, "hi": hi})
    rows += [{"T": "-", "R": f"vote m={m}", "acc": FA[0, m], "tokens": FT[0, m], "vs_best_baseline": np.nan,
              "lo": np.nan, "hi": np.nan} for m in ms]
    rows += [{"T": "-", "R": f"short-1@{k}", "acc": SA[0, i], "tokens": ST[0, i], "vs_best_baseline": np.nan,
              "lo": np.nan, "hi": np.nan} for i, k in enumerate(ks)]
    return rows, vec


def wp_sums(g, x, y):
    # per-problem within-problem sum of products of x and y (each centered on its problem's mean)
    df = pd.DataFrame({"g": g, "x": np.asarray(x, float), "y": np.asarray(y, float)})
    df[["x", "y"]] -= df.groupby("g")[["x", "y"]].transform("mean")
    return (df.x * df.y).groupby(df.g).sum()


def capture(g, pids, x, a, b, draws):
    # share of the prefix-fixed variance x explains (one global slope): Cov(x, a) Cov(x, b) / (Var x Cov(a, b)),
    # a and b two rollout halves of the same prefix with independent noise
    sa, sb, sx, sab = (wp_sums(g, u, v).reindex(pids).fillna(0).values for u, v in ((x, a), (x, b), (x, x), (a, b)))
    with np.errstate(divide="ignore", invalid="ignore"):
        est = sa.sum() * sb.sum() / (sx.sum() * sab.sum()) if sx.sum() > 0 else 0.0
        boot = np.where(draws @ sx > 0, (draws @ sa) * (draws @ sb) / ((draws @ sx) * (draws @ sab)), 0.0)
    return est, boot  # a score with no within-problem variation explains none of D


def q1(cuts, bank, scores, n_boot=spec.N_BOOT, frozen=False, log=print):
    # question 1 family (spec Q1_*). population at each bank checkpoint: live traces with a bank there, in
    # problems with at least two such traces. D = within-problem share of one continuation's outcome variance
    # fixed by the prefix: split-half covariance of r over the problem's (n - 1)-weighted Bernoulli variance,
    # ratio of sums over problems, negatives kept
    rows, est = [], {}
    if not {"r_h0", "r_h1"} <= set(bank.columns):
        return {"rows": rows, "verdict": "not evaluable on this bundle"}
    for c in sorted(int(x) for x in set(bank.checkpoint) - {0}):
        live = set(cuts[(cuts.checkpoint == c) & cuts.live.astype(bool)].trace_id)
        b = bank[(bank.checkpoint == c) & bank.trace_id.isin(live) & (bank.k >= 2)].dropna(subset=["r_h0", "r_h1"])
        b = b[b.groupby("problem_id").trace_id.transform("size") >= 2].reset_index(drop=True)
        if b.empty:
            continue
        g = b.problem_id.values
        fix = wp_sums(g, b.r_h0, b.r_h1)
        pids = fix.index.values
        agg = pd.DataFrame({"g": g, "n": b.k.astype(float), "h": b.r * b.k, "m": 1.0}).groupby("g").sum().loc[pids]
        qp = agg.h / agg.n
        tot = ((agg.m - 1) * qp * (1 - qp) * agg.n / (agg.n - 1)).values
        draws = S.problem_draws(len(pids), n_boot, spec.BOOT_SEED)
        D, Db = fix.loc[pids].values.sum() / tot.sum(), (draws @ fix.loc[pids].values) / (draws @ tot)
        base = {"checkpoint": c, "traces": len(b), "problems": len(pids)}
        rows.append({**base, "what": "D", "est": D, "lo": S.ci(Db)[0], "hi": S.ci(Db)[1]})
        est[(c, "D")] = (D, Db)
        sc = scores[scores.checkpoint == c].pivot_table(index="trace_id", columns="monitor", values="score")
        for mon in sc.columns:
            x = sc[mon].reindex(b.trace_id).values
            if np.isnan(x).any():
                continue
            cap, capb = capture(g, pids, x, b.r_h0, b.r_h1, draws)
            rows.append({**base, "what": f"capture {mon}", "est": cap, "lo": S.ci(capb)[0], "hi": S.ci(capb)[1]})
            est[(c, mon)] = (cap, capb)
        # forward runs vs the primary monitor, both scored on the halves the forward monitor does not use
        if not {"m_r", "m_k", "r_q0", "r_q1"} <= set(b.columns) or spec.PRIMARY_MONITOR not in sc.columns:
            continue
        f = b[(b.m_k == spec.FWD_K)].dropna(subset=["r_q0", "r_q1"])
        f = f[f.groupby("problem_id").trace_id.transform("size") >= 2].reset_index(drop=True)
        if f.empty:
            continue
        gf = f.problem_id.values
        pf = np.unique(gf)
        df_ = S.problem_draws(len(pf), n_boot, spec.BOOT_SEED)
        cf, cfb = capture(gf, pf, f.m_r.values, f.r_q0, f.r_q1, df_)
        ca, cab = capture(gf, pf, f.m_r_all.values, f.r_q0, f.r_q1, df_)
        cp, cpb = capture(gf, pf, sc[spec.PRIMARY_MONITOR].reindex(f.trace_id).values, f.r_q0, f.r_q1, df_)
        fb = {"checkpoint": c, "traces": len(f), "problems": len(pf), "tokens": float(f.m_tok.mean())}
        with np.errstate(divide="ignore", invalid="ignore"):
            ret, retb = cf / ca, cfb / cab
        rows += [{**fb, "what": f"capture forward {spec.FWD_K} runs, cap {spec.FWD_CAP}", "est": cf,
                  "lo": S.ci(cfb)[0], "hi": S.ci(cfb)[1]},
                 {**fb, "what": f"capture forward {spec.FWD_K} runs, uncapped", "est": ca,
                  "lo": S.ci(cab)[0], "hi": S.ci(cab)[1]},
                 {**fb, "what": f"expected uncapped capture from D, D / (D + (1 - D) / {spec.FWD_K})",
                  "est": D / (D + (1 - D) / spec.FWD_K), "lo": np.nan, "hi": np.nan},
                 {**fb, "what": f"cap retention, capped / uncapped", "est": ret, "lo": S.ci(retb)[0], "hi": S.ci(retb)[1]},
                 {**fb, "what": f"forward (capped) minus {spec.PRIMARY_MONITOR}, follows from D", "est": cf - cp,
                  "lo": S.ci(cfb - cpb)[0], "hi": S.ci(cfb - cpb)[1]}]
        est[(c, "retention")] = (ret, retb)
    c = spec.Q1_CUT
    have = all(k in est for k in ((c, "D"), (c, spec.PRIMARY_MONITOR), (c, "retention")))
    tests = {}
    if have:
        tests = {"T1 D above D_MIN": S.ci(est[(c, "D")][1])[0] > spec.D_MIN,
                 "T2 primary capture below CAPTURE_MAX": S.ci(est[(c, spec.PRIMARY_MONITOR)][1])[1] < spec.CAPTURE_MAX,
                 "T3 cap retention above RETAIN_MIN": S.ci(est[(c, "retention")][1])[0] > spec.RETAIN_MIN}
    if not have:
        verdict = "not evaluable on this bundle"
    elif not frozen:
        verdict = f"not evaluated: team slots are {spec.SLOTS.get('status')}, not frozen"
    else:
        verdict = "met" if all(tests.values()) else "not met"
    return {"rows": rows, "tests": tests, "verdict": verdict}


def run(bundle, out, n_boot=spec.N_BOOT, log=print, keep="outcome"):
    meta, src, cuts, bank, votes, scores = load(bundle)
    dev = meta["kind"] != "confirmation"
    unbanked = check(meta, src, cuts, bank, votes, scores, log)
    pops, matched = populations(src, cuts)
    log(pops.to_string(index=False))
    ckpts = sorted(cuts.checkpoint.unique())
    draws = S.problem_draws(src.problem_id.nunique(), n_boot, spec.BOOT_SEED)
    pol_rows, vec, auc_rows, auc_boot, auc_est = [], {}, [], {}, {}
    for c in ckpts:
        t = attempts(src, cuts, bank, votes, c, unbanked, keep)
        if c in set(bank.checkpoint):
            rows, v = policies(t, c, scores, draws, log)
            pol_rows += rows
            vec.update({(c, *k): x for k, x in v.items()})
        pops_auc = [("live", t.live_all), ("live, pre-answer", t.live_all & t.pre_answer),
                    ("live, pre-answer (sensitivity: strict value match)", t.live_all & t.pre_answer_strict),
                    ("live, pre-answer (sensitivity: </think> and \\boxed only)", t.live_all & t.pre_answer_think_boxed),
                    ("matched cohort", t.trace_id.isin(matched) & t.live_all)]
        if t.unbanked.any() and c in set(bank.checkpoint):
            pops_auc.append(("live with a rollout bank (r_hat vs monitors)", t.live))
        for label, mask in pops_auc:
            r, b = aurocs(t[mask.values].reset_index(drop=True), scores, c, n_boot, label)
            auc_rows += r
            if label == "live, pre-answer":
                auc_boot[(c, "pre_answer")] = b
                auc_est[(c, "pre_answer")] = {x["score"]: x["auroc"] for x in r}
    t_rows, t_vec = timeout_vec(src, votes, draws)
    vec.update(t_vec)
    margin = spec.SLOTS["margin_pp"] if spec.SLOTS.get("status") == "frozen" else None
    s7c = next(s["checkpoint"] for s in spec.SECONDARY if s.get("kind") == "auroc_pre_answer")
    s7_mixed = pops.loc[pops.checkpoint == s7c, "mixed_pre_answer"]
    res = tests(vec, auc_boot, auc_est, margin, int(s7_mixed.iloc[0]) if len(s7_mixed) else None)
    qr = q1(cuts, bank, scores, n_boot, frozen=spec.SLOTS.get("status") == "frozen", log=log)
    res["q1"] = {"tests": qr["tests"], "verdict": qr["verdict"]}
    os.makedirs(out, exist_ok=True)
    pd.DataFrame(qr["rows"]).to_csv(os.path.join(out, "q1.csv"), index=False)
    pd.DataFrame(t_rows).to_csv(os.path.join(out, "timeout.csv"), index=False)
    pol = pd.DataFrame(pol_rows)
    pol.to_csv(os.path.join(out, "policies.csv"), index=False)
    pd.DataFrame(auc_rows).to_csv(os.path.join(out, "auroc.csv"), index=False)
    pops.to_csv(os.path.join(out, "populations.csv"), index=False)
    json.dump(res, open(os.path.join(out, "tests.json"), "w"), indent=1, default=float)
    md = report(meta, pops, pol, pd.DataFrame(auc_rows), res, dev, len(unbanked), pd.DataFrame(qr["rows"]),
                pd.DataFrame(t_rows))
    open(os.path.join(out, "report.md"), "w").write(md)
    log("\n" + md)
    return pol, pd.DataFrame(auc_rows), res


def pp(x):
    return "–" if x is None else f"{100 * x:+.1f}"


def ci3(v, scale=100, fmt="{:+.1f}"):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "–"
    e, lo, hi = v
    return f"{fmt.format(scale * e)} [{fmt.format(scale * lo)}, {fmt.format(scale * hi)}]"


def report(meta, pops, pol, auc, res, dev, n_unbanked, q1rows=None, trows=None):
    L = [f"# Confirmation analysis ({'DEVELOPMENT DRY RUN' if dev else 'confirmation data'})\n"]
    v = pol[pol.arm.str.startswith("resample validity check (every")] if len(pol) else pol
    for r in v.itertuples():
        bad = r.delta[1] > 0 or r.delta[2] < 0
        L.append(f"> Resample validity check at {r.checkpoint}: {ci3(r.delta)} pp"
                 + (" — **CI excludes 0: source outcomes and rollouts disagree; run the pre-specified sensitivity "
                    "analysis (--keep rollout) and report both.**" if bad else " (CI covers 0).") + "\n")
    if dev:
        L.append("> **Development dry run.** This checks that the analysis runs end to end; it is not evidence. "
                 "The monitors were fit on these traces (their training traces are scored out of fold, but the "
                 "thresholds and the design were chosen on them), the rollout bank covers only the dev ceiling cohort "
                 f"({n_unbanked} live cuts have no bank and are treated as not flaggable), and dev sources and "
                 "rollouts came from different engines. Nothing below may be cited as a result.\n")
    L.append(f"Team slots (hash {spec.slots_sha()[:16]}): {json.dumps(spec.SLOTS)}\n")
    L.append(f"Spec {spec.spec_sha()[:16]}; bundle: {json.dumps({k: v for k, v in meta.items() if k not in ('provenance', 'files')})}\n")
    L.append("## Populations\n")
    L.append("Accuracy endpoints use every attempt of every problem. *live* = still generating after c tokens and "
             "</think> not yet written; only live traces can be flagged. *matched cohort* = live at every checkpoint.\n")
    L.append("| " + " | ".join(pops.columns) + " |")
    L.append("|" + "---|" * len(pops.columns))
    for r in pops.itertuples(index=False):
        L.append("| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in r) + " |")
    L.append("\n## Within-problem AUROC, P(wrong) [95% CI]\n")
    L.append("Mixed problems among the rows scored. *rollout r_hat* is rollout-estimated from "
             f"{spec.K_PREFIX} continuations per cut: a noisy diagnostic of what the prefix determines, not a bound on any monitor.\n")
    if len(auc):
        L.append("| population | checkpoint | score | AUROC [95% CI] | minus len [95% CI] | traces / problems / pairs |")
        L.append("|---|---|---|---|---|---|")
        for r in auc.itertuples():
            d = getattr(r, "minus_len", None)
            dci = getattr(r, "minus_len_ci", None)
            ds = "–" if d is None or (isinstance(d, float) and np.isnan(d)) else f"{d:+.3f} [{dci[0]:+.3f}, {dci[1]:+.3f}]"
            L.append(f"| {r.population} | {r.checkpoint} | {r.score} | {r.auroc:.3f} [{r.ci_low:.3f}, {r.ci_high:.3f}] "
                     f"| {ds} | {r.traces} / {r.mixed_problems} / {r.pairs} |")
    L.append("\n## Policies: accuracy per attempt and total generated tokens per attempt\n")
    L.append("Tokens count everything generated: discarded prefixes, replacements, extra votes and, for the rollout-"
             "ranked diagnostic, the rollouts used to decide. Comparators marked *equal tokens* are funded to the "
             "policy's own total tokens in every bootstrap draw (pi = their rate). CIs: problem bootstrap.\n")
    if len(pol):
        L.append("| checkpoint | arm | comparator | accuracy % | tokens / attempt | Δ pp [95% CI] | flagged (of live) | pi |")
        L.append("|---|---|---|---|---|---|---|---|")
        for r in pol.itertuples():
            tok = f"{r.tokens[0]:,.0f} [{r.tokens[1]:,.0f}, {r.tokens[2]:,.0f}]"
            acc = f"{100 * r.acc[0]:.1f}"
            pi = "" if r.pi is None or (isinstance(r.pi, float) and np.isnan(r.pi)) else f"{r.pi:.2f}"
            st = getattr(r, "match_status", None)
            if st is not None and not (isinstance(st, float) and np.isnan(st)) and st != 0:
                pi += " (capped)" if st == 1 else " (cannot match; vs none)"
            L.append(f"| {r.checkpoint} | {r.arm} | {r.comparator} | {acc} | {tok} | {ci3(r.delta)} | "
                     f"{r.flag_share:.2f} | {pi} |")
    L.append("\n## Pre-registered tests\n")
    for k in ("primary vs random", f"primary vs aggregate{spec.AGG_EXTRA[0]}"):
        o = res.get(k)
        L.append(f"- {k}: " + ("not evaluable on this bundle" if o is None else
                                f"{pp(o['delta'])} pp [{pp(o['ci'][0])}, {pp(o['ci'][1])}]"))
    L.append(f"- primary verdict (both lower 95% bounds above 0): **{res['primary verdict']}**")
    for k in ("primary vs random", f"primary vs aggregate{spec.AGG_EXTRA[0]}"):
        o = res.get(k)
        if o is not None and "rules_out_margin" in o:
            L.append(f"- {k}, gain >= {spec.SLOTS['margin_pp']:g} pp ruled out (one-sided 95% limit "
                     f"{pp(o['one_sided_95'])} pp): **{'yes' if o['rules_out_margin'] else 'no'}**")
    for s in res["secondary"]:
        if s.get("delta") is None:
            L.append(f"- {s['id']}: not evaluable on this bundle")
            continue
        unit = "" if s.get("kind") == "auroc_pre_answer" else " pp"
        f = (lambda x: f"{x:+.3f}") if unit == "" else pp
        tail = (f"estimate only: {s['mixed']} mixed problems < {spec.S7_MIN_MIXED}, no claim, not in the Holm family"
                if s.get("estimate_only") else f"Holm p {s.get('p_holm', float('nan')):.3f}")
        what = (f"timeout restart T {s['T']}, R {s['R']}, vs {s['vs']}" if s.get("kind") == "timeout" else
                f"{s.get('kind', s.get('action'))} at {s['checkpoint']}, {s['monitor']} vs {s['vs']}")
        L.append(f"- {s['id']} ({what}): "
                 f"{f(s['delta'])}{unit} [{f(s['ci'][0])}, {f(s['ci'][1])}], {tail}")
    if trows is not None and len(trows):
        L.append("\n## Timeout restart vs the best no-monitor baseline at equal tokens (S8 and grid)\n")
        L.append("A source still running at T tokens is aborted and replaced by fresh t0-bank samples (at most R "
                 "restarts); every token charged. Compared, at the same tokens per attempt, with the better of the vote "
                 "frontier and the short-1@k frontier (first of k parallel samples to finish), mixtures included "
                 "(upper concave envelope).\n")
        L.append("| T | R | accuracy % | tokens / attempt | vs best baseline, pp [95% CI] |")
        L.append("|---|---|---|---|---|")
        for r in trows.itertuples():
            L.append(f"| {r.T} | {r.R} | {100 * r.acc:.1f} | {r.tokens:,.0f} | "
                     f"{pp(r.vs_best_baseline) if np.isfinite(r.vs_best_baseline) else '–'} "
                     f"[{pp(r.lo) if np.isfinite(r.lo) else '–'}, {pp(r.hi) if np.isfinite(r.hi) else '–'}] |")
    if q1rows is not None and len(q1rows):
        L.append("\n## Question 1: how much the prefix fixes, and what captures it\n")
        L.append("D = within-problem share of one continuation's outcome variance fixed by the prefix. capture = "
                 "share of D a score explains (one global slope), scored on rollout halves the score does not use.\n")
        L.append("| checkpoint | quantity | estimate [95% CI] | traces / problems |")
        L.append("|---|---|---|---|")
        for r in q1rows.itertuples():
            L.append(f"| {r.checkpoint} | {r.what} | {r.est:.3f} [{r.lo:.3f}, {r.hi:.3f}] | {r.traces} / {r.problems} |")
        for k, v in res["q1"]["tests"].items():
            L.append(f"- {k}: {'yes' if v else 'no'}")
        L.append(f"- question 1 verdict (T1, T2 and T3 all hold): **{res['q1']['verdict']}**")
    L.append("\nWording: report these as estimates on this cohort. A null here means the tested monitors have not "
             "shown intervention value at equal compute; it does not show that hidden states lack the information.\n")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=spec.N_BOOT)
    ap.add_argument("--keep", default="outcome", choices=["outcome", "rollout"],
                    help="value of a kept live trace: its outcome (primary) or its prefix's bank mean (sensitivity)")
    a = ap.parse_args()
    run(a.bundle, a.out, a.n_boot, keep=a.keep)


if __name__ == "__main__":
    main()
