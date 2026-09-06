import json, os
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cache import StaleCache, check_dataset_manifest, short
from extract import ACTS_PATH, LAYERS, META_PATH, TRACES_PATH
from taxonomy import RESTART

RESULTS_PATH = "data/results.parquet"
SUMMARY_PATH = "data/summary.json"
N_BINS = 8
CS = [0.01, 0.1, 1.0]
K_FOLDS = 5
SEED = 0


def make_splits(problem_ids):
    ids = np.array(sorted(set(problem_ids)))
    np.random.default_rng(SEED).shuffle(ids)
    cut = int(0.8 * len(ids))
    return ids[:cut], ids[cut:]


def folds(dev):
    # dev is already shuffled, so striding gives random folds
    return [
        (np.concatenate([dev[j::K_FOLDS] for j in range(K_FOLDS) if j != i]), dev[i::K_FOLDS])
        for i in range(K_FOLDS)
    ]


def trace_level(meta, probs):
    df = pd.DataFrame({"trace_id": meta.trace_id.values, "y": meta.y.values, "p": probs})
    return df.groupby("trace_id").agg(y=("y", "first"), p=("p", "mean"))


def auroc(g):
    return roc_auc_score(g.y, g.p) if g.y.nunique() > 1 else np.nan


def fit(X, y, C):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000)).fit(X, y)


def run_grid(meta, acts, dev, bins=range(N_BINS)):
    # K-fold CV over dev for cell selection: one val split has too few errors to pick
    # the best of ~200 cells without landing on noise
    splits = folds(dev)
    # neither the bin nor the fold membership depends on layer or C, so slice once
    # instead of re-filtering for all 27 layer x C combinations
    cells = {}
    for b in bins:
        cell = meta[meta.bin == b]
        cells[b] = [(cell[cell.problem_id.isin(tr_ids)], cell[cell.problem_id.isin(va_ids)])
                    for tr_ids, va_ids in splits]
    out = []
    for layer_pos, layer in enumerate(LAYERS):
        feats = np.asarray(acts[:, layer_pos, :], dtype=np.float32)
        print(f"  layer {layer}", flush=True)
        for b in bins:
            for C in CS:
                scores = []
                for tr, va in cells[b]:
                    if tr.y.nunique() < 2 or va.y.nunique() < 2:
                        continue
                    model = fit(feats[tr.row.values], tr.y.values, C)
                    g = trace_level(va, model.predict_proba(feats[va.row.values])[:, 1])
                    scores.append(auroc(g))
                if scores:
                    out.append(
                        {
                            "layer": layer,
                            "bin": b,
                            "C": C,
                            "cv_auroc": float(np.nanmean(scores)),
                            "cv_std": float(np.nanstd(scores)),
                            "n_folds": len(scores),
                        }
                    )
    return pd.DataFrame(out)


def evaluate(meta, acts, dev, test, layer, b, C, bins=None):
    feats = np.asarray(acts[:, LAYERS.index(layer), :], dtype=np.float32)
    cell = meta[meta.bin == b] if bins is None else meta[meta.bin.isin(bins)]
    tr = cell[cell.problem_id.isin(dev)]
    te = cell[cell.problem_id.isin(test)]
    model = fit(feats[tr.row.values], tr.y.values, C)
    g = trace_level(te, model.predict_proba(feats[te.row.values])[:, 1])
    return auroc(g), g


def bootstrap_ci(g, n=1000):
    rng = np.random.default_rng(SEED)
    y, p = g.y.values, g.p.values
    scores = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) > 1:
            scores.append(roc_auc_score(y[i], p[i]))
    return [float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))]


def select_and_test(meta, acts, dev, test, tag, bins=range(N_BINS)):
    cached = f"data/grid_{tag}.parquet"
    if os.path.exists(cached):
        grid = pd.read_parquet(cached)
    else:
        grid = run_grid(meta, acts, dev, bins)
        grid["condition"] = tag
        grid.to_parquet(cached)
    best = grid.loc[grid.cv_auroc.idxmax()]
    test_auroc, g = evaluate(meta, acts, dev, test, int(best.layer), int(best.bin), float(best.C))
    return grid, {
        "layer": int(best.layer),
        "bin": int(best.bin),
        "C": float(best.C),
        "cv_auroc": float(best.cv_auroc),
        "cv_std": float(best.cv_std),
        "test_auroc": float(test_auroc),
        "test_ci": bootstrap_ci(g),
        "n_test_traces": int(len(g)),
        "n_test_wrong": int(g.y.sum()),
    }


def length_matched(meta):
    per_trace = meta.groupby("trace_id").agg(y=("y", "first"), n_steps=("n_steps", "first"))
    q = pd.qcut(per_trace.n_steps, 5, labels=False, duplicates="drop")
    rng = np.random.default_rng(SEED)
    keep = []
    for b in np.unique(q):
        grp = per_trace[q == b]
        wrong, right = grp[grp.y == 1], grp[grp.y == 0]
        k = min(len(wrong), len(right))
        if k:
            keep += list(rng.choice(wrong.index, k, replace=False))
            keep += list(rng.choice(right.index, k, replace=False))
    return meta[meta.trace_id.isin(keep)]


def shuffled(meta):
    labels = meta.groupby("trace_id").y.first()
    perm = np.random.default_rng(SEED + 1).permutation(labels.values)
    swap = dict(zip(labels.index, perm))
    out = meta.copy()
    out["y"] = out.trace_id.map(swap)
    return out


def main():
    meta = pd.read_parquet(META_PATH)
    acts = np.load(ACTS_PATH, mmap_mode="r")
    # rows index acts positionally, so prove the two files came from one extraction
    dm = check_dataset_manifest(META_PATH, ACTS_PATH, meta.trace_id.tolist(), acts.shape)
    if not (meta.row.values == np.arange(len(meta))).all():
        raise StaleCache("meta.row is not 0..n-1, so it cannot index acts.npy")
    if dm["config"].get("layers") != LAYERS:
        raise StaleCache(f"LAYERS = {LAYERS} but acts.npy was extracted with "
                         f"{dm['config'].get('layers')}")
    print(f"dataset config {short(dm['config_hash'])}, extracted {dm['written']}")
    meta["y"] = (~meta.is_correct.astype(bool)).astype(int)
    meta["rel"] = meta.step_idx / meta.n_steps
    meta["bin"] = np.clip((meta.rel * N_BINS).astype(int), 0, N_BINS - 1)

    dev, test = make_splits(meta.problem_id)
    assert not set(dev) & set(test)
    print(f"{len(dev)} dev / {len(test)} test problems, {K_FOLDS}-fold CV for selection")

    grids, summary = [], {}
    for tag, m in [
        ("main", meta),
        ("shuffled", shuffled(meta)),
        ("length_matched", length_matched(meta)),
    ]:
        print(tag, flush=True)
        grid, best = select_and_test(m, acts, dev, test, tag)
        grids.append(grid)
        summary[tag] = best
        print(tag, best["layer"], best["bin"], f"test {best['test_auroc']:.3f}")

    early = meta[meta.rel < 0.6]
    grid, best = select_and_test(early, acts, dev, test, "first60", range(5))
    grids.append(grid)
    summary["first60"] = best
    print("first60", best["layer"], best["bin"], f"test {best['test_auroc']:.3f}")

    summary["selectivity_gap"] = summary["main"]["test_auroc"] - summary["shuffled"]["test_auroc"]

    traces = [json.loads(line) for line in open(TRACES_PATH)]
    summary["truncation_rate"] = sum(t["trunc"] for t in traces) / len(traces)
    for t in traces:
        if t["trunc"]:
            t["is_correct"] = False
    summary["accuracy"] = sum(t["is_correct"] for t in traces) / len(traces)
    summary["self_correction_rate"] = sum(
        bool(RESTART.search(t["text"].split("</think>")[0])) for t in traces
    ) / len(traces)

    by_level = Counter((t["level"], t["is_correct"]) for t in traces)
    summary["by_level"] = {
        str(l): {"n": by_level[(l, True)] + by_level[(l, False)], "wrong": by_level[(l, False)]}
        for l in (3, 4, 5)
    }

    pd.concat(grids).to_parquet(RESULTS_PATH)
    json.dump(summary, open(SUMMARY_PATH, "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
