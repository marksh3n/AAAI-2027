# length-filtered r_hat: apply the source traces' selection to every continuation. a
# continuation is kept only if it finished and cut + its length >= 4096, as the 448 source
# traces were (untruncated, >= 4096 tokens). r_hat per prefix over kept continuations; K_eff = 0
# prefixes are dropped. AUROC uses ceiling.compare, so the bootstrap is the one already reported.
#   python lenfilter.py --rollouts R --traces data/traces_B.jsonl --meta data/meta_B.parquet --preds P
import argparse, json, os, sys
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--rollouts", required=True)
ap.add_argument("--traces", required=True)
ap.add_argument("--meta", required=True)
ap.add_argument("--preds", required=True)
ap.add_argument("--repo", default=".")
ap.add_argument("--out", default=None, help="optional csv of the per-prefix table")
a = ap.parse_args()
sys.path.insert(0, os.path.join(a.repo, "src", "probe_cmp"))
import ceiling as C  # noqa: E402
import stats as S  # noqa: E402

MAX_NEW, MIN_LEN = 32768, 4096
ORDER = ["t0", "p512", "p2048", "p4096"]
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)

# one row per continuation, deduplicated exactly as ceiling.load_rollouts does
last, torn = {}, 0
for line in open(a.rollouts):
    if not line.strip():
        continue
    try:
        r = json.loads(line)
    except json.JSONDecodeError:
        torn += 1
        continue
    for s in r["samples"]:
        last[(r["prefix_id"], r["round"], s["sample_idx"])] = {
            "prefix_id": r["prefix_id"], "problem_id": r["problem_id"], "trace_id": r["trace_id"],
            "checkpoint": r["checkpoint"], "cut": r["cut_token_idx"], "n_gen": s["n_gen"],
            "finish": s["finish_reason"], "truncated": bool(s["truncated"]), "correct": bool(s["correct"]),
            "think": "</think>" in s["text"]}
cont = pd.DataFrame(list(last.values()))
print(f"{len(cont)} continuations over {cont.prefix_id.nunique()} prefixes; torn lines skipped {torn}")
print("finish_reason:", cont.finish.value_counts(dropna=False).to_dict())
assert not (cont.correct & cont.truncated).any(), "a truncated continuation is scored correct"

cont["finished"] = ~cont.truncated
cont["long_enough"] = cont.cut + cont.n_gen >= MIN_LEN
cont["keep"] = cont.finished & cont.long_enough
cont["k_keep"] = cont.keep & cont.correct

pref = cont.groupby("prefix_id").agg(
    problem_id=("problem_id", "first"), trace_id=("trace_id", "first"), checkpoint=("checkpoint", "first"),
    cut=("cut", "first"), n=("correct", "size"), k=("correct", "sum"), n_fin=("finished", "sum"),
    K_eff=("keep", "sum"), k_eff=("k_keep", "sum")).reset_index()
pref["r"] = pref.k / pref.n
pref["r_finished"] = (pref.k / pref.n_fin.where(pref.n_fin > 0)).fillna(pref.r)  # as ceiling.py
pref["r_len"] = pref.k_eff / pref.K_eff.where(pref.K_eff > 0)

traces = C.load_traces(a.traces)
has = pref.trace_id.notna()
pref["y"] = np.nan
pref.loc[has, "y"] = traces.loc[pref.trace_id[has], "y"].values
pref["label"] = np.where(~has, "t0", np.where(pref.y == 1, "wrong", "correct"))

# same per-prefix n, k and r_finished as the pipeline's own loader
base, _ = C.load_rollouts(a.rollouts)
chk = pref.merge(base[["prefix_id", "n", "k", "r_finished"]], on="prefix_id", suffixes=("", "_c"))
assert len(chk) == len(pref) == len(base), (len(chk), len(pref), len(base))
assert (chk.n == chk.n_c).all() and (chk.k == chk.k_c).all() and np.allclose(chk.r_finished, chk.r_finished_c)
print("per-prefix n, k, r_finished match ceiling.load_rollouts")

# where the other source conditions would bite: vLLM's n_gen counts the EOS on a stop while the
# source's >= 4096 used the retokenized length without it; the source had 32768 tokens in total
# (cut included) and a trace without </think> counted as truncated
meta = C.load_meta(a.meta)
pref["pre_think"] = [isinstance(t, str) and "</think>" in meta.at[(t, int(c)), "prefix"]
                     for t, c in zip(pref.trace_id, pref.cut)]
cont = cont.merge(pref[["prefix_id", "pre_think"]], on="prefix_id")
eos_flip = cont.finished & (cont.long_enough != (cont.cut + cont.n_gen - (cont.finish == "stop") >= MIN_LEN))
over_cap = cont.keep & (cont.cut + cont.n_gen >= MAX_NEW)
no_think = cont.keep & ~(cont.think | cont.pre_think)
print(f"kept continuations: {int(cont.keep.sum())}; whose keep flips if the EOS is not counted: "
      f"{int(eos_flip.sum())}")
print("kept but outside the source's other rules, by checkpoint:")
print(pd.DataFrame({"kept": cont.groupby("checkpoint").keep.sum(),
                    "cut + n_gen >= 32768": over_cap.groupby(cont.checkpoint).sum(),
                    "no </think> in prefix or continuation": no_think.groupby(cont.checkpoint).sum()})
      .reindex(ORDER).to_string())
cont["strict"] = cont.keep & ~over_cap & ~no_think
st = cont.groupby("prefix_id").agg(K_strict=("strict", "sum"),
                                   k_strict=("correct", lambda v: int((v & cont.loc[v.index, "strict"]).sum())))
pref = pref.merge(st, on="prefix_id")
pref["r_strict"] = pref.k_strict / pref.K_strict.where(pref.K_strict > 0)

print("\n== continuations removed, and prefixes dropped at K_eff = 0 ==")
cont["label"] = cont.prefix_id.map(pref.set_index("prefix_id").label)
rem = cont.groupby(["checkpoint", "label"]).agg(
    continuations=("keep", "size"), truncated=("truncated", "sum"),
    finished_but_short=("long_enough", lambda v: int((~v & cont.loc[v.index, "finished"]).sum())),
    kept=("keep", "sum"))
drop = pref.groupby(["checkpoint", "label"]).agg(
    prefixes=("K_eff", "size"), dropped=("K_eff", lambda v: int((v == 0).sum())),
    dropped_all_truncated=("n_fin", lambda v: int((v == 0).sum())),
    mean_K_eff=("K_eff", "mean"))
drop["dropped_finished_but_short"] = drop.dropped - drop.dropped_all_truncated
print(rem.join(drop).reindex(ORDER, level=0).round(2).to_string())

kept = pref[pref.K_eff > 0]
tr = kept[kept.label != "t0"]
print("\n== (a) mean r_hat by source label, kept prefixes ==")
print(tr.groupby(["checkpoint", "label"]).agg(
    prefixes=("r_len", "size"), r_len=("r_len", "mean"), r_finished_same_prefixes=("r_finished", "mean"),
    r_unfiltered_same_prefixes=("r", "mean")).reindex(ORDER[1:], level=0).round(3).to_string())

pop = sorted(pref.trace_id.dropna().unique())
acc_all = 1 - traces.loc[pop, "y"].mean()
print(f"\n== (b) mean r_hat over traces vs source accuracy ({len(pop)} traces, accuracy {acc_all:.3f}) ==")


def paired_ci(d, groups):
    # problem-clustered bootstrap of a per-trace mean, 2000 draws, seed 0
    g = pd.Series(d).groupby(np.asarray(groups))
    sums, cnts = g.sum().values, g.size().values
    draws = S.problem_draws(len(sums), 2000, seed=0)
    return S.ci(draws @ sums / (draws @ cnts))


rows = []
for c in ORDER[1:]:
    allc, g = pref[pref.checkpoint == c], tr[tr.checkpoint == c]
    d = g.r_len.values - (1 - g.y.values)
    lo, hi = paired_ci(d, g.problem_id.values)
    gs = allc[allc.K_strict > 0]
    rows.append({"checkpoint": c, "traces": len(allc), "kept": len(g), "mean r_len": g.r_len.mean(),
                 "source acc, kept": 1 - g.y.mean(), "r_len - acc (paired)": d.mean(), "CI low": lo, "CI high": hi,
                 "mean r_finished, all": allc.r_finished.mean(), "mean r, all": allc.r.mean(),
                 "mean r_strict": gs.r_strict.mean(), "source acc, strict kept": 1 - gs.y.mean()})
# t0 is per problem: weight each problem by its traces in the population
w = traces.loc[pop].groupby("problem_id").size()
t0 = pref[pref.checkpoint == "t0"].set_index("problem_id")
t0k = t0[t0.K_eff > 0]
acc_p = 1 - traces.loc[pop].groupby("problem_id").y.mean()
rows.append({"checkpoint": "t0 (trace-weighted)", "traces": int(w.loc[t0.index].sum()),
             "kept": int(w.loc[t0k.index].sum()),
             "mean r_len": float((t0k.r_len * w.loc[t0k.index]).sum() / w.loc[t0k.index].sum()),
             "source acc, kept": float((acc_p.loc[t0k.index] * w.loc[t0k.index]).sum() / w.loc[t0k.index].sum()),
             "mean r_finished, all": float((t0.r_finished * w.loc[t0.index]).sum() / w.loc[t0.index].sum()),
             "mean r, all": float((t0.r * w.loc[t0.index]).sum() / w.loc[t0.index].sum())})
print(pd.DataFrame(rows).round(3).to_string(index=False))

print("\n== (c) within-problem AUROC, P(wrong); r_hat score = 1 - r_hat; deltas paired on the same problem draws ==")
out = []
for rc, pc in C.CKPTS.items():
    g = tr[tr.checkpoint == rc][["trace_id", "problem_id", "y", "r", "r_finished", "r_len"]].copy()
    g["y"] = g.y.astype(int)
    preds = C.load_preds(os.path.join(a.preds, f"{pc}.parquet"))
    # compare() scores the columns r and r_finished: the first call puts the unfiltered r_hat in
    # the r_finished slot, the second the usual r_finished. every delta is length-filtered minus
    # the row's score, on the same traces and the same problem draws
    one, ties = C.compare(g.assign(r=g.r_len, r_finished=g.r), preds)
    two, _ = C.compare(g.assign(r=g.r_len), preds)
    rows_c = pd.concat([one[one.score == "r_hat"].assign(score="r_hat length-filtered"),
                        one[one.score == "r_hat finished"].assign(score="r_hat unfiltered"),
                        two[two.score == "r_hat finished"],
                        one[one.score.isin(["lr/raw L20", "len"])]])
    for r in rows_c.itertuples():
        out.append({"checkpoint": pc, "score": r.score, "AUROC": f"{r.auroc:.3f} [{r.ci_low:.3f}, {r.ci_high:.3f}]",
                    "length-filtered minus score": "–" if r.score == "r_hat length-filtered" else
                    f"{r.delta_r_hat_minus:+.3f} [{r.delta_ci_low:+.3f}, {r.delta_ci_high:+.3f}] p={r.delta_p:.3f}",
                    "traces/problems/pairs": f"{r.n_traces}/{r.n_problems}/{r.n_pairs}"})
    out.append({"checkpoint": pc, "score": f"(pairs tied under length-filtered r_hat: {ties:.2f})"})
print(pd.DataFrame(out).fillna("").to_string(index=False))

g4 = pref[pref.checkpoint == "p4096"]
k4 = g4[g4.K_eff > 0]
print(f"\n4096 check: length condition failed by {int((~cont[cont.checkpoint == 'p4096'].long_enough).sum())} "
      f"continuations; {int((g4.K_eff == 0).sum())} prefixes dropped; max |r_len - r_finished| on kept "
      f"{float((k4.r_len - k4.r_finished).abs().max()):.2e}")
if a.out:
    pref.to_csv(a.out, index=False)
    print("wrote", a.out)
