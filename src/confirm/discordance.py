# what the dev dry run implies for the paired planning design (one binary outcome per arm per problem):
# the effect d and the discordance q = P(the two arms disagree on a problem), for the primary policy
# (spec.PRIMARY at its frozen threshold) against token-matched aggregation m=2 and random restart,
# then the power of his two tests (power.py) at those values. per attempt, from the bundle's
# expectations: a flagged live trace is replaced by a fresh sample (correct with prob r0); the
# comparator acts on each eligible attempt with prob pi. two pairings:
#   shared source   both arms continue the same source trace and draw their replacements independently,
#                   so they disagree only where one of them acts
#   independent     each arm generates its own source; q = E_problem[pM (1 - pC) + pC (1 - pM)]
# the 87 dry-run problems are the ceiling cohort (mixed problems whose traces reach 4096), not an
# unfiltered population: --traces adds a diluted scenario that scales q and d by the share of dev
# problems in that cohort (both are ~0 on problems no policy can change; an approximation).
#   python src/confirm/discordance.py --bundle <dev_bundle> [--traces data/traces_B.jsonl] [--n 1000]
import argparse, json, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "probe_cmp"))
import spec  # noqa: E402
import policy as P  # noqa: E402
import analyze as A  # noqa: E402
import power as W  # noqa: E402


def implied(t, scores, c):
    """per comparator: d, q_shared, q_indep, pi, from the attempts table t at checkpoint c"""
    pr = spec.PRIMARY
    sums = P.Sums(t.problem_id.values, np.zeros((0, t.problem_id.nunique())))
    base_acc, base_tok = t.correct.values.astype(float), t.n.values.astype(float)
    gain, extra = P.arm_effect(t, pr["action"], c)
    g = scores[(scores.checkpoint == c) & (scores.monitor == pr["monitor"])]
    s = g.set_index("trace_id").score.reindex(t.trace_id).values
    w = P.flags(np.nan_to_num(s, nan=-np.inf), float(g.threshold.iloc[0]), t.live.values).astype(float)
    pM, tM = base_acc + w * gain, base_tok + w * extra
    m = spec.AGG_EXTRA[0]
    comps = {"random": (gain, extra), f"aggregate{m}": (t[f"A{m}"].values - base_acc, m * t.L0.values)}
    out = {}
    for name, (cg, ce) in comps.items():
        _, pi, status = P.matched(sums, base_acc, base_tok, pM, tM, cg, ce)
        pC = base_acc + pi[0] * cg
        acting = float(pi[0] if name.startswith("aggregate") else pi[0] * t.live.values.mean())
        q_shared = float(np.mean(pM * (1 - pC) + pC * (1 - pM)))
        pb = pd.DataFrame({"p": t.problem_id.values, "m": pM, "c": pC}).groupby("p").mean()
        q_indep = float(np.mean(pb.m * (1 - pb.c) + pb.c * (1 - pb.m)))
        out[name] = {"d": float(np.mean(pM - pC)), "q_shared": q_shared, "q_indep": q_indep, "pi": float(pi[0]),
                     "match_status": int(status[0]), "comparator_acting_share": acting,
                     "monitor_flag_share_of_attempts": float(w.mean())}
    return out


def dev_share(traces_path, n_cohort):
    tr = pd.read_json(traces_path, lines=True)
    tr["ok"] = tr.is_correct.astype(bool) & ~tr.trunc.astype(bool)
    k = tr.groupby("problem_id").ok.agg(["sum", "size"])
    return {"dev_problems": int(len(k)), "dev_mixed_problems": int(((k["sum"] > 0) & (k["sum"] < k["size"])).sum()),
            "cohort_problems": n_cohort, "cohort_share": n_cohort / len(k)}


def run(bundle, traces=None, n=1000, margin=None, log=print):
    meta, src, cuts, bank, votes, scores = A.load(bundle)
    unbanked = A.check(meta, src, cuts, bank, votes, scores, lambda *a: None)
    c = spec.PRIMARY["checkpoint"]
    t = A.attempts(src, cuts, bank, votes, c, unbanked)
    res = {"checkpoint": c, "problems": int(t.problem_id.nunique()), "attempts": len(t), "comparators": implied(t, scores, c)}
    if traces:
        res["dilution"] = dev_share(traces, res["problems"])
    margin = (spec.SLOTS["margin_pp"] or 5.0) / 100 if margin is None else margin
    rows = []
    for name, v in res["comparators"].items():
        scen = [("shared source", v["q_shared"], v["d"]), ("independent", v["q_indep"], v["d"])]
        if traces:
            sh = res["dilution"]["cohort_share"]
            scen += [("shared source, diluted", v["q_shared"] * sh, v["d"] * sh),
                     ("independent, diluted", v["q_indep"] * sh, v["d"] * sh)]
        for label, q, d in scen:
            d_use = float(np.clip(d, -q, q))
            rows.append({"comparator": name, "pairing": label, "n": n, "q": q, "d": d,
                         "mcnemar_detect": W.mcnemar_power(n, q, d_use),
                         "tango_rule_out": W.ruleout_power(n, q, d_use, margin)})
    rows.append({"comparator": "planning assumption", "pairing": "-", "n": n, "q": 0.25, "d": 0.05,
                 "mcnemar_detect": W.mcnemar_power(n, 0.25, 0.05), "tango_rule_out": W.ruleout_power(n, 0.25, 0.0, margin)})
    res["power"] = rows
    log(json.dumps({k: v for k, v in res.items() if k != "power"}, indent=1))
    log(f"\npower at N={n}, McNemar two-sided .05 against 0; Tango one-sided .05 against d >= {margin:.2f} "
        "(planning row: rule-out at true d = 0)")
    log(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--traces", default=None)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    res = run(a.bundle, a.traces, a.n)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
