# within-problem probe at fixed generation positions.
# constants duplicated, not imported: keeps this torch-free
import argparse
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

LAYERS = [-1, 0, 4, 8, 12, 16, 20, 24, 27]
POSITIONS = [128, 256, 512, 1024, 2048, 4096]
K_FOLDS = 5
MIN_PAIRS = 20
SEED = 0


def splits(problem_ids):
    ids = np.array(sorted(set(problem_ids)))
    np.random.default_rng(SEED).shuffle(ids)
    cut = int(0.8 * len(ids))
    return ids[:cut], ids[cut:]


def folds(dev):
    return [
        (np.concatenate([dev[j::K_FOLDS] for j in range(K_FOLDS) if j != i]), dev[i::K_FOLDS])
        for i in range(K_FOLDS)
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
    # resample problems: samples within a problem are dependent
    rng = np.random.default_rng(SEED)
    vals = [auroc(counts[rng.integers(0, len(counts), len(counts))]) for _ in range(n)]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def fit_score(x_tr, y_tr, x_te, C=1.0):
    model = make_pipeline(StandardScaler(with_mean=False), LogisticRegression(C=C, max_iter=2000))
    return model.fit(x_tr, y_tr).predict_proba(x_te)[:, 1]


def probe_scores(cell, feats, tr_ids, te_ids):
    tr, te = cell[cell.problem_id.isin(tr_ids)], cell[cell.problem_id.isin(te_ids)]
    if tr.y.nunique() < 2 or not len(te):
        return None
    return te.assign(p=fit_score(feats[tr.row.values], tr.y.values, feats[te.row.values]))


def tfidf_scores(cell, tr_ids, te_ids):
    tr, te = cell[cell.problem_id.isin(tr_ids)], cell[cell.problem_id.isin(te_ids)]
    if tr.y.nunique() < 2 or not len(te):
        return None
    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    return te.assign(p=fit_score(vec.fit_transform(tr.prefix), tr.y.values, vec.transform(te.prefix)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="B")
    args = parser.parse_args()

    s = f"_{args.tag}" if args.tag else ""
    meta = pd.read_parquet(f"data/meta{s}.parquet")
    acts = np.load(f"data/acts{s}.npy", mmap_mode="r")
    meta["y"] = (~meta.is_correct.astype(bool)).astype(int)

    per_trace = meta.groupby("trace_id").agg(y=("y", "first"), pid=("problem_id", "first"))
    mixed = {p for p, g in per_trace.groupby("pid") if g.y.nunique() > 1}
    print(f"{len(mixed)} mixed-outcome problems of {meta.problem_id.nunique()}")
    if len(mixed) < 30:
        print("WARNING: under 30 mixed problems, the within-problem test is underpowered")

    dev, test = splits(meta.problem_id)
    dev_mixed = np.array([p for p in dev if p in mixed])
    test_mixed = np.array([p for p in test if p in mixed])
    print(f"{len(dev)} dev / {len(test)} test problems "
          f"({len(dev_mixed)} / {len(test_mixed)} mixed)\n")

    header = f"{'pos':>5}{'n':>5}{'layer':>7}{'cv':>7}{'test':>7}{'95% CI':>16}{'d':>7}"
    print(header + f"{'tfidf':>8}{'restart':>8}{'n_gen':>8}{'shuf':>7}")
    out = []
    for pos in POSITIONS:
        cell = meta[meta.pos == pos]
        if not len(cell) or cell[cell.problem_id.isin(dev)].y.nunique() < 2:
            continue

        # select layer on dev, read test once
        best, best_cv = None, -1
        for layer_pos, layer in enumerate(LAYERS):
            feats = np.asarray(acts[:, layer_pos, :], dtype=np.float32)
            scores = []
            for tr_ids, va_ids in folds(dev):
                scored = probe_scores(cell, feats, tr_ids, [v for v in va_ids if v in mixed])
                if scored is not None:
                    scores.append(auroc(pair_counts(scored)))
            cv = np.nanmean(scores) if scores and not np.all(np.isnan(scores)) else np.nan
            if not np.isnan(cv) and cv > best_cv:
                best, best_cv = layer, cv
        if best is None:
            print(f"{pos:>5}  skipped: no dev fold had both outcomes")
            continue

        feats = np.asarray(acts[:, LAYERS.index(best), :], dtype=np.float32)
        scored = probe_scores(cell, feats, dev, test_mixed)
        counts = pair_counts(scored)
        if counts[:, 1].sum() < MIN_PAIRS:
            print(f"{pos:>5}  skipped: {int(counts[:, 1].sum())} test pairs, too few to read")
            continue
        ci = boot_ci(counts)

        # baselines share the probe's test problems
        tf = tfidf_scores(cell, dev, test_mixed)
        te = cell[cell.problem_id.isin(test_mixed)]
        rng = np.random.default_rng(SEED + 1)
        shuf = cell.copy()
        labels = shuf.groupby("trace_id").y.first()
        shuf["y"] = shuf.trace_id.map(dict(zip(labels.index, rng.permutation(labels.values))))

        row = {
            "pos": pos,
            "n_test_pairs": int(counts[:, 1].sum()),
            "layer": best,
            "cv": best_cv,
            "test": auroc(counts),
            "ci": ci,
            "d": paired_d(scored),
            "tfidf": auroc(pair_counts(tf)) if tf is not None else np.nan,
            # n_gen is an oracle; restarts and prefix are online
            "restarts": auroc(pair_counts(te.assign(p=te.restarts))),
            "n_gen_oracle": auroc(pair_counts(te.assign(p=te.n_gen))),
            "shuffled": auroc(pair_counts(probe_scores(shuf, feats, dev, test_mixed))),
        }
        out.append(row)
        print(f"{pos:>5}{row['n_test_pairs']:>5}{best:>7}{best_cv:>7.3f}{row['test']:>7.3f}"
              f"{f'[{ci[0]:.2f}, {ci[1]:.2f}]':>16}{row['d']:>7.2f}"
              f"{row['tfidf']:>8.3f}{row['restarts']:>8.3f}"
              f"{row['n_gen_oracle']:>8.3f}{row['shuffled']:>7.3f}", flush=True)

    pd.DataFrame(out).to_json(f"data/fixed_summary{s}.json", orient="records", indent=2)
    print(f"\nwrote data/fixed_summary{s}.json")


if __name__ == "__main__":
    main()
