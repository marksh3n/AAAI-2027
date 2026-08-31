# within-problem probe at fixed generation positions.
# constants duplicated, not imported: keeps this torch-free
import argparse
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

LAYERS = [-1, 0, 4, 8, 12, 16, 20, 24, 27]
POSITIONS = [128, 256, 512, 1024, 2048, 4096]
N_OUTER = 5
N_INNER = 3
CS = [0.001, 0.01, 0.1, 1.0]
MIN_PAIRS = 20
SEED = 0


def folds(ids, k):
    ids = np.asarray(ids)
    return [
        (np.concatenate([ids[j::k] for j in range(k) if j != i]), ids[i::k]) for i in range(k)
    ]


def pair_counts(df):
    # (wrong, correct) pairs within one problem: difficulty and length cancel
    out = []
    for _, g in df.groupby("problem_id"):
        a, b = g.p[g.y == 1].values, g.p[g.y == 0].values
        if len(a) and len(b):
            d = a[:, None] - b[None, :]
            out.append(((d > 0).sum() + 0.5 * (d == 0).sum(), d.size))
    return np.array(out, dtype=float).reshape(-1, 2)


def auroc(counts):
    return counts[:, 0].sum() / counts[:, 1].sum() if len(counts) else np.nan


def paired_d(df):
    diffs = np.array(
        [
            g.p[g.y == 1].mean() - g.p[g.y == 0].mean()
            for _, g in df.groupby("problem_id")
            if g.y.nunique() > 1
        ]
    )
    if len(diffs) < 2:
        return np.nan
    sd = diffs.std(ddof=1)
    return diffs.mean() / sd if sd > 1e-9 else np.nan


def boot_ci(counts, n=2000):
    # resample problems: pairs within a problem are dependent
    rng = np.random.default_rng(SEED)
    vals = [auroc(counts[rng.integers(0, len(counts), len(counts))]) for _ in range(n)]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def fit_score(x_tr, y_tr, x_te, C=1.0, svd=0):
    # svd is fitted inside the fold, never on the test problems
    steps = [StandardScaler(with_mean=False)]
    if svd:
        steps.append(TruncatedSVD(n_components=svd, random_state=SEED))
    steps.append(LogisticRegression(C=C, max_iter=2000))
    return make_pipeline(*steps).fit(x_tr, y_tr).predict_proba(x_te)[:, 1]


def scores(cell, feats, tr_ids, te_ids, C=1.0, svd=0):
    tr, te = cell[cell.problem_id.isin(tr_ids)], cell[cell.problem_id.isin(te_ids)]
    if tr.y.nunique() < 2 or not len(te):
        return None
    return te.assign(p=fit_score(feats[tr.row.values], tr.y.values, feats[te.row.values], C, svd))


def tfidf(cell, tr_ids, te_ids):
    tr, te = cell[cell.problem_id.isin(tr_ids)], cell[cell.problem_id.isin(te_ids)]
    if tr.y.nunique() < 2 or not len(te):
        return None
    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    return te.assign(p=fit_score(vec.fit_transform(tr.prefix), tr.y.values, vec.transform(te.prefix)))


def nested(cell, feats_by_layer, outer, mixed, svd=0):
    # inner folds pick the layer, outer folds score it. with ~38 mixed problems a single
    # held-out split leaves too few to read, so every problem serves as test once.
    parts, picked = [], []
    for tr, te in outer:
        te_mixed = [p for p in te if p in mixed]
        if not te_mixed:
            continue
        best, best_cv = None, -1
        # 1536 dims against ~2k rows, so C matters more than the layer does
        for layer, feats in zip(LAYERS, feats_by_layer):
            for C in CS:
                inner = []
                for i_tr, i_te in folds(tr, N_INNER):
                    s = scores(cell, feats, i_tr, [p for p in i_te if p in mixed], C, svd)
                    if s is not None:
                        inner.append(auroc(pair_counts(s)))
                cv = np.nanmean(inner) if inner and not np.all(np.isnan(inner)) else np.nan
                if not np.isnan(cv) and cv > best_cv:
                    best, best_cv = (layer, C), cv
        if best is None:
            continue
        s = scores(cell, feats_by_layer[LAYERS.index(best[0])], tr, te_mixed, best[1], svd)
        if s is not None:
            parts.append(s)
            picked.append(best)
    return (pd.concat(parts) if parts else None), picked


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="B")
    parser.add_argument("--svd", type=int, default=0)
    args = parser.parse_args()

    s = f"_{args.tag}" if args.tag else ""
    meta = pd.read_parquet(f"data/meta{s}.parquet")
    acts = np.load(f"data/acts{s}.npy", mmap_mode="r")
    meta["y"] = (~meta.is_correct.astype(bool)).astype(int)
    feats_by_layer = [np.asarray(acts[:, i, :], dtype=np.float32) for i in range(len(LAYERS))]

    per_trace = meta.groupby("trace_id").agg(y=("y", "first"), pid=("problem_id", "first"))
    mixed = {p for p, g in per_trace.groupby("pid") if g.y.nunique() > 1}
    print(f"{len(mixed)} mixed-outcome problems of {meta.problem_id.nunique()}")
    if len(mixed) < 30:
        print("WARNING: under 30 mixed problems, the within-problem test is underpowered")

    ids = np.array(sorted(meta.problem_id.unique()))
    np.random.default_rng(SEED).shuffle(ids)
    outer = folds(ids, N_OUTER)
    print(f"{N_OUTER}-fold outer CV, {N_INNER}-fold inner over layer x C"
          + (f", SVD to {args.svd} components\n" if args.svd else "\n"))

    print(f"{'pos':>5}{'pairs':>7}{'(layer, C)':>34}{'auroc':>7}{'95% CI':>16}{'d':>7}"
          f"{'tfidf':>8}{'restart':>8}{'n_gen':>8}{'shuf':>7}")
    out = []
    for pos in POSITIONS:
        cell = meta[meta.pos == pos]
        if not len(cell):
            continue
        scored, picked = nested(cell, feats_by_layer, outer, mixed, args.svd)
        if scored is None:
            print(f"{pos:>5}  skipped: no fold had both outcomes")
            continue
        counts = pair_counts(scored)
        if counts[:, 1].sum() < MIN_PAIRS:
            print(f"{pos:>5}  skipped: {int(counts[:, 1].sum())} pairs, too few to read")
            continue

        # baselines run through the same outer folds; n_gen is an oracle a live monitor
        # never has, restarts and prefix text are what it does have
        tf = pd.concat([r for r in (tfidf(cell, tr, [p for p in te if p in mixed])
                                    for tr, te in outer) if r is not None])
        te_all = cell[cell.problem_id.isin(mixed)]
        top = max(set(picked), key=picked.count)
        shuf = cell.copy()
        lab = shuf.groupby("trace_id").y.first()
        perm = np.random.default_rng(SEED + 1).permutation(lab.values)
        shuf["y"] = shuf.trace_id.map(dict(zip(lab.index, perm)))
        sh = pd.concat([r for r in (scores(shuf, feats_by_layer[LAYERS.index(top[0])], tr,
                                           [p for p in te if p in mixed], top[1], args.svd)
                                    for tr, te in outer) if r is not None])

        row = {
            "pos": pos,
            "pairs": int(counts[:, 1].sum()),
            "n_problems": len(counts),
            "layers": picked,
            "auroc": auroc(counts),
            "ci": boot_ci(counts),
            "d": paired_d(scored),
            "tfidf": auroc(pair_counts(tf)),
            "restarts": auroc(pair_counts(te_all.assign(p=te_all.restarts))),
            "n_gen_oracle": auroc(pair_counts(te_all.assign(p=te_all.n_gen))),
            "shuffled": auroc(pair_counts(sh)),
        }
        out.append(row)
        ci = row["ci"]
        print(f"{pos:>5}{row['pairs']:>7}{str(picked):>34}{row['auroc']:>7.3f}"
              f"{f'[{ci[0]:.2f}, {ci[1]:.2f}]':>16}{row['d']:>7.2f}"
              f"{row['tfidf']:>8.3f}{row['restarts']:>8.3f}"
              f"{row['n_gen_oracle']:>8.3f}{row['shuffled']:>7.3f}", flush=True)

    pd.DataFrame(out).to_json(f"data/fixed_summary{s}.json", orient="records", indent=2)
    print(f"\nwrote data/fixed_summary{s}.json")


if __name__ == "__main__":
    main()
