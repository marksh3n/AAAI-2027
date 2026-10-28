# mixed-problem selection. the population keeps a problem only if its eligible traces (untruncated,
# n_gen < 32768, retokenized length >= 4096: build_prefixes.ceiling_population, the same set as
# probe_cmp's) hold a correct and a wrong one. with per-trace accuracy p among eligible traces and
# n of them, E[source accuracy | mixed] = (p - p^n) / (1 - p^n - (1-p)^n). p_hat per problem comes
# from the t0 rollouts, length-filtered like the source (lenfilter.py's r_len at t0).
#   python mixed_sel.py --prefixes <lenfilter_prefixes.csv> --traces data/traces_B.jsonl
import argparse, json, os, sys
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--prefixes", required=True, help="lenfilter.py's per-prefix csv")
ap.add_argument("--traces", required=True)
ap.add_argument("--repo", default=".")
ap.add_argument("--skip-rebuild", action="store_true", help="skip re-deriving the population (needs the tokenizer)")
ap.add_argument("--out", default=None, help="optional per-problem csv")
a = ap.parse_args()
sys.path.insert(0, os.path.join(a.repo, "src", "probe_cmp"))
import stats as S  # noqa: E402

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)
CUTS = ["p512", "p2048", "p4096"]
pref = pd.read_csv(a.prefixes)
traces = [json.loads(line) for line in open(a.traces)]
T = pd.DataFrame(traces).set_index("trace_id", drop=False).rename_axis(None)
pop_ids = set(pref.trace_id.dropna())
pop = T.loc[sorted(pop_ids)]
print(f"population from the rollouts: {len(pop)} traces, {pop.problem_id.nunique()} problems")

if not a.skip_rebuild:
    sys.path.insert(0, os.path.join(a.repo, "src", "recovery"))
    from build_prefixes import CEILING_POSITIONS, ceiling_population  # noqa: E402
    from common import MAX_NEW, encode_trace, load_tokenizer  # noqa: E402
    tok = load_tokenizer()
    # exactly build_ceiling's lens
    lens = {t["trace_id"]: len(encode_trace(tok, t["text"])[0]) if not t["trunc"] and t["n_gen"] < MAX_NEW else 0
            for t in traces}
    rebuilt = ceiling_population(traces, lens, max(CEILING_POSITIONS))
    got = {t["trace_id"] for ts in rebuilt.values() for t in ts}
    assert got == pop_ids, f"rebuilt population differs: {len(got ^ pop_ids)} traces"
    elig = T[[lens[t] >= max(CEILING_POSITIONS) for t in T.index]]
    g6 = T.groupby("problem_id").is_correct.agg(["min", "max"])
    mixed6 = set(g6.index[g6["min"] != g6["max"]])
    has_elig = set(elig.problem_id)
    print(f"ceiling_population rebuilt from traces_B gives the same {len(got)} traces: mixed is taken over "
          f"eligible traces only.\n  problems with >= 1 eligible trace: {len(has_elig)}; mixed over eligible: "
          f"{len(rebuilt)}; mixed over all 6 samples: {len(mixed6 & has_elig)} of them ({len(mixed6)} of 367 overall)")

n = pop.groupby("problem_id").size()
acc = pop.groupby("problem_id").is_correct.mean()
t0 = pref[pref.checkpoint == "t0"].set_index("problem_id").loc[n.index]
assert (t0.K_eff > 0).all()
p_len, p_unf = t0.k_eff / t0.K_eff, t0.k / t0.n


def f(p, n):
    # E[accuracy | 1 <= correct <= n-1]; limits at p = 0 (1/n) and p = 1 ((n-1)/n)
    p, n = np.asarray(p, float), np.asarray(n, float)
    den = 1 - p ** n - (1 - p) ** n
    with np.errstate(invalid="ignore", divide="ignore"):
        v = (p - p ** n) / den
    return np.where(p <= 0, 1 / n, np.where(p >= 1, (n - 1) / n, v))


def f_post(k, K, n, draws=4000, seed=0):
    # f averaged over a Jeffreys Beta(k + 1/2, K - k + 1/2) posterior for p: p_hat has K <= 8 samples
    rng = np.random.default_rng(seed)
    return np.array([f(rng.beta(ki + 0.5, Ki - ki + 0.5, draws), ni).mean() for ki, Ki, ni in zip(k, K, n)])


prob = pd.DataFrame({"n_eligible": n, "source_acc": acc, "p_hat": p_len, "K_eff_t0": t0.K_eff,
                     "pred": f(p_len, n), "pred_posterior": f_post(t0.k_eff, t0.K_eff, n),
                     "pred_unfiltered_p": f(p_unf, n), "pred_n6": f(p_len, 6)})
rr = (pref[pref.checkpoint.isin(CUTS)].pivot(index="trace_id", columns="checkpoint", values="r_len")
      .reindex(columns=CUTS))
tr = pop[["problem_id", "is_correct"]].join(rr)
for c in CUTS:
    prob[f"r_{c}"] = tr.groupby("problem_id")[c].mean()
print("n eligible per problem:", n.value_counts().sort_index().to_dict())


def wmean(col, sub):
    return float((sub[col] * sub.n_eligible).sum() / sub.n_eligible.sum())


def ci(sub, num):
    # problem-clustered bootstrap of a trace-weighted mean of per-problem values, 2000 draws, seed 0
    draws = S.problem_draws(len(sub), 2000, seed=0)
    w = sub.n_eligible.values
    return S.ci(draws @ (w * num) / (draws @ w))


def report(sub, name):
    obs, pred = wmean("source_acc", sub), wmean("pred", sub)
    lo, hi = ci(sub, (sub.source_acc - sub.pred).values)
    row = {"problems": f"{name}: {len(sub)}", "traces": int(sub.n_eligible.sum()), "source acc": obs,
           "predicted | mixed": pred, "obs - pred": obs - pred, "CI": f"[{lo:+.3f}, {hi:+.3f}]",
           "pred (posterior)": wmean("pred_posterior", sub), "pred (unfiltered p)": wmean("pred_unfiltered_p", sub),
           "pred (n=6)": wmean("pred_n6", sub), "mean p_hat (t0)": wmean("p_hat", sub)}
    for c in CUTS:
        sub_t = tr[tr.problem_id.isin(sub.index)]
        g = sub_t.assign(d=sub_t[c] - sub_t.is_correct.astype(float)).groupby("problem_id").d.mean()
        d_lo, d_hi = ci(sub, g.loc[sub.index].values)
        row[f"r {c[1:]}"] = wmean(f"r_{c}", sub)
        row[f"r {c[1:]} - acc"] = f"{wmean(f'r_{c}', sub) - obs:+.3f} [{d_lo:+.3f}, {d_hi:+.3f}]"
    return row


bands = [("all", prob), ("p_hat in [0.35, 0.65]", prob[(prob.p_hat >= 0.35) & (prob.p_hat <= 0.65)]),
         ("p_hat < 0.35", prob[prob.p_hat < 0.35]), ("p_hat > 0.65", prob[prob.p_hat > 0.65])]
rows = [report(sub, name) for name, sub in bands if len(sub)]
out = pd.DataFrame(rows).set_index("problems")
print("\n== source accuracy vs its prediction under mixed selection, and mean r_hat (trace-weighted) ==")
print(out[["traces", "source acc", "predicted | mixed", "obs - pred", "CI", "mean p_hat (t0)"]].round(3).to_string())
print("\nsensitivity of the prediction:")
print(out[["pred (posterior)", "pred (unfiltered p)", "pred (n=6)"]].round(3).to_string())
print("\nmean length-filtered r_hat at each cut, and r_hat - source acc (paired, problem bootstrap):")
print(out[[c for c in out.columns if c.startswith("r ")]].round(3).to_string())
al = out.iloc[0]
print(f"\nshare of the t0 gap (mean p_hat - source acc = {al['mean p_hat (t0)'] - al['source acc']:+.3f}) that "
      f"selection predicts: {(al['mean p_hat (t0)'] - al['predicted | mixed']) / (al['mean p_hat (t0)'] - al['source acc']):.0%}")
print("note: inside a p_hat band, mean p_hat (t0) is selected on its own noise; the r columns use other rollouts")
if a.out:
    prob.to_csv(a.out)
    print("wrote", a.out)
