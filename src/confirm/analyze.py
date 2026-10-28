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
    out = {}
    pr = spec.PRIMARY
    for comp in ("random", f"aggregate{spec.AGG_EXTRA[0]}"):
        v = vec.get((pr["checkpoint"], pr["monitor"], pr["action"], comp))
        if v is None:
            out[f"primary vs {comp}"] = None
            continue
        est, lo, hi = summ(v)
        out[f"primary vs {comp}"] = {"delta": est, "ci": [lo, hi], "p": wald_p(est, v[1:])}
    ok = [out[k] for k in out if k.startswith("primary")]
    if margin is None:
        verdict = (f"not evaluated: team slots are {spec.SLOTS.get('status')}, not frozen"
                   if spec.SLOTS.get("status") != "frozen" else "not evaluated: margin slot is empty")
    elif any(o is None for o in ok):
        verdict = "not evaluable on this bundle"
    else:
        verdict = "met" if all(o["ci"][0] > margin / 100 for o in ok) else "not met"
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
    margin = spec.SLOTS["margin_pp"] if spec.SLOTS.get("status") == "frozen" else None
    s7c = next(s["checkpoint"] for s in spec.SECONDARY if s.get("kind") == "auroc_pre_answer")
    s7_mixed = pops.loc[pops.checkpoint == s7c, "mixed_pre_answer"]
    res = tests(vec, auc_boot, auc_est, margin, int(s7_mixed.iloc[0]) if len(s7_mixed) else None)
    os.makedirs(out, exist_ok=True)
    pol = pd.DataFrame(pol_rows)
    pol.to_csv(os.path.join(out, "policies.csv"), index=False)
    pd.DataFrame(auc_rows).to_csv(os.path.join(out, "auroc.csv"), index=False)
    pops.to_csv(os.path.join(out, "populations.csv"), index=False)
    json.dump(res, open(os.path.join(out, "tests.json"), "w"), indent=1, default=float)
    md = report(meta, pops, pol, pd.DataFrame(auc_rows), res, dev, len(unbanked))
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


def report(meta, pops, pol, auc, res, dev, n_unbanked):
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
    L.append(f"- primary verdict (both lower bounds above the margin): **{res['primary verdict']}**")
    for s in res["secondary"]:
        if s.get("delta") is None:
            L.append(f"- {s['id']}: not evaluable on this bundle")
            continue
        unit = "" if s.get("kind") == "auroc_pre_answer" else " pp"
        f = (lambda x: f"{x:+.3f}") if unit == "" else pp
        tail = (f"estimate only: {s['mixed']} mixed problems < {spec.S7_MIN_MIXED}, no claim, not in the Holm family"
                if s.get("estimate_only") else f"Holm p {s.get('p_holm', float('nan')):.3f}")
        L.append(f"- {s['id']} ({s.get('kind', s.get('action'))} at {s['checkpoint']}, {s['monitor']} vs {s['vs']}): "
                 f"{f(s['delta'])}{unit} [{f(s['ci'][0])}, {f(s['ci'][1])}], {tail}")
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
