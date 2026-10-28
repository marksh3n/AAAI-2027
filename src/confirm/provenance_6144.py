# where the dev 6144 "top-neuron tree" number comes from. CPU, no model. three numbers circulate:
#   Ghait's memo (PR #5, analysis/E0_FINDINGS.md): 197 traces / 53 mixed problems, depth-3 tree on 50
#     training-selected layer-20 neurons, within-problem AUROC .642 / .679 / .642 over three grouped fold
#     assignments, observable-prefix baseline .553 / .572 / .553
#   Jaden's exports (jaden/confirmation-planning): e0_6144_preds_pop342.csv (342 / 74) tree .523,
#     len .554, tfidf .545; e0_6144_preds_cohort227.csv (227 / 59) tree .573, len .568, tfidf .440
# the tree is probe_cmp's hgb/topk50 (models.py): HistGradientBoosting, max_depth 3, 150 rounds, on the
# 50 columns with the largest within-problem |AUROC - .5| in the training fold. this script rebuilds
# the cohorts, searches rules that give 197 / 53, and refits tree and baselines per fold seed with
# probe_cmp's own folds (run.make_folds), so each number can be matched to a population.
#   python src/confirm/provenance_6144.py --data data --out data/confirm/provenance_6144
import argparse, json, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "probe_cmp"))
from monitors import oof_scores  # noqa: E402
import bundle_lib as B  # noqa: E402
import data as D  # noqa: E402
import run as R  # noqa: E402
import stats as S  # noqa: E402


ARMS = [("hgb", "topk50", 20), ("lr", "len", None), ("lr", "tfidf", None), ("lr", "raw", 20)]
TARGET = (197, 53)


def remix(store, ids):
    per = store.traces.loc[list(ids), ["trace_id", "problem_id", "y"]].reset_index(drop=True)
    per = per[per.problem_id.isin(D.mixed(per))].sort_values(["problem_id", "trace_id"])
    return per.reset_index(drop=True)


def pre_answer(store, ckpt, ids):
    texts = store.text(ckpt, list(ids))
    # the strict rule (value anywhere), as in the first provenance run
    return {i for i, t in zip(ids, texts) if B.cut_rules(t, store.traces.at[i, "answer"], True)["pre_answer_strict"]}


def no_think(store, ckpt, ids):
    return {i for i, t in zip(ids, store.text(ckpt, list(ids))) if "</think>" not in t}


def candidates(store):
    # probe_cmp's subset runs: every registered position up to the deepest one (128 ... 6144 / 8192)
    up_to = lambda c: [x for x in store.ckpts if x != D.END and int(x) <= c]
    p342 = D.population(store, up_to(6144))
    p227 = D.population(store, up_to(8192)) if "8192" in store.ckpts else p342.iloc[:0]
    at = lambda c: set(store.table[(store.table.ckpt == c) & store.table.eligible].trace_id)
    line = pd.Series(np.arange(len(store.traces)), index=store.traces.index)
    main = set(line.index[line >= 480])
    c8 = at("8192")
    rules = {
        "pop342: eligible, reach 6144, mixed": p342,
        "cohort227: eligible, reach 8192, mixed": p227,
        "reach 8192, no </think> at 8192": remix(store, no_think(store, "8192", c8)),
        "reach 8192, no </think> at 6144": remix(store, no_think(store, "6144", c8)),
        "reach 8192, pre-answer at 8192": remix(store, pre_answer(store, "8192", c8)),
        "reach 8192, pre-answer at 6144": remix(store, pre_answer(store, "6144", c8)),
        "reach 6144, pre-answer at 6144": remix(store, pre_answer(store, "6144", at("6144"))),
        "reach 6144, no </think> at 6144": remix(store, no_think(store, "6144", at("6144"))),
        "reach 8192, main block (two BOS)": remix(store, c8 & main),
        "reach 6144, main block (two BOS)": remix(store, at("6144") & main),
        "reach 8192, n_gen < 16384": remix(store, {t for t in c8 if store.traces.at[t, "n_gen"] < 16384}),
        "reach 8192, n_gen < 12288": remix(store, {t for t in c8 if store.traces.at[t, "n_gen"] < 12288}),
    }
    return rules


def auroc_per_seed(store, pop, ckpt, seeds=(0, 1, 2)):
    ids, y, g = pop.trace_id.values, pop.y.values, pop.problem_id.values
    texts = store.text(ckpt, ids)
    tokens = store.rows(ckpt, ids).tokens.values.astype(float)
    X = store.hidden(ckpt, ids, 20)
    fmap = R.make_folds(pop, list(seeds))
    pairs = S.Pairs(y, g)
    out = []
    for model, feat, layer in ARMS:
        ps = []
        for s in seeds:
            p, _ = oof_scores(model, feat, y, g, R.fold_indices(pop, fmap[str(s)]), s, X, texts, tokens)
            ps.append(p)
            out.append({"arm": f"{model}/{feat}", "seed": s, "auroc": pairs.auroc(p)["weighted"]})
        out.append({"arm": f"{model}/{feat}", "seed": "mean of seed predictions",
                    "auroc": pairs.auroc(np.mean(ps, 0))["weighted"]})
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--registry", default=D.REGISTRY)
    ap.add_argument("--out", required=True)
    ap.add_argument("--jaden", nargs="*", default=[], help="Jaden's e0_6144_preds_*.csv, to compare trace sets")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    store = D.Store(a.data, reg=D.load_registry(a.registry), names=["B_fixed"])
    rules = candidates(store)
    rows = []
    print("candidate cohorts (traces / mixed problems); Ghait's memo cites 197 / 53")
    for name, pop in rules.items():
        n, m = len(pop), pop.problem_id.nunique()
        rows.append({"rule": name, "traces": n, "mixed_problems": m, "matches_197_53": (n, m) == TARGET})
        print(f"  {n:4d} / {m:3d}  {'<== 197 / 53' if (n, m) == TARGET else '':12s} {name}")
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "cohorts.csv"), index=False)
    for path in a.jaden:
        j = pd.read_csv(path)
        for name, pop in rules.items():
            if set(pop.trace_id) == set(j.trace_id):
                print(f"  {os.path.basename(path)} has exactly the traces of: {name}")
    res = []
    todo = {k: v for k, v in rules.items() if k.startswith(("pop342", "cohort227")) or (len(v), v.problem_id.nunique()) == TARGET}
    for name, pop in todo.items():
        if pop.problem_id.nunique() < R.N_SPLITS:
            print(f"\n{name}: {pop.problem_id.nunique()} mixed problems, too few to refit")
            continue
        for ckpt in ("4096", "6144", "8192"):
            if not set(pop.trace_id) <= set(store.table[store.table.ckpt == ckpt].trace_id):
                continue
            r = auroc_per_seed(store, pop, ckpt).assign(cohort=name, checkpoint=ckpt)
            res.append(r)
            piv = r.pivot(index="arm", columns="seed", values="auroc")
            print(f"\n{name} @ {ckpt}: within-problem AUROC by fold seed\n{piv.round(3).to_string()}")
    if res:
        pd.concat(res).to_csv(os.path.join(a.out, "auroc_by_seed.csv"), index=False)
    json.dump({"memo": "PR #5 analysis/E0_FINDINGS.md: 197/53, tree .642/.679/.642, baseline .553/.572/.553",
               "jaden": {"pop342": {"tree": .523, "len": .554, "tfidf": .545},
                         "cohort227": {"tree": .573, "len": .568, "tfidf": .440}}},
              open(os.path.join(a.out, "cited.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
