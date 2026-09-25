# probe comparison across checkpoints on one fixed trace population. fits everything and
# saves out-of-fold predictions; report.py turns those into every table, CI and the plot
# without refitting. CPU only.
#   python src/probe_cmp/run.py --quick     # 1 seed, 50 permutations, layer 20
#   python src/probe_cmp/run.py             # 3 seeds, 200 permutations, layers 20 and 16
import argparse, hashlib, json, os, sys, time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from sklearn.model_selection import StratifiedGroupKFold

import data as D
import models as M
import report
from stats import Pairs, permute_within

PRIMARY_LAYER = 20
LAYERS = [20, 16]
SEEDS = [0, 1, 2]
N_SPLITS = 5
N_PERM = 200
MIN_PROBLEMS = 40
NO_LAYER = -999  # len and tfidf do not read hidden states
HIDDEN_ARMS = [("lr", "raw"), ("lr", "svd50"), ("lr", "topk50"), ("hgb", "svd50"), ("hgb", "topk50"),
               ("mlp", "svd50"), ("mlp", "topk50"), ("lr", "svd50+len")]
SHARED_ARMS = [("lr", "len"), ("lr", "tfidf")]
PERM_ARMS = [("lr", "svd50"), ("hgb", "svd50")]
VERSION = 1  # bump when fitting changes; saved predictions from another version are refused


def sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def make_folds(pop, seeds):
    # problem -> fold, per seed. stratified on the trace label, grouped by problem
    out = {}
    for s in seeds:
        fold = np.empty(len(pop), int)
        cv = StratifiedGroupKFold(N_SPLITS, shuffle=True, random_state=s)
        for k, (_, te) in enumerate(cv.split(pop, pop.y, pop.problem_id)):
            fold[te] = k
        out[str(s)] = {str(p): int(k) for p, k in zip(pop.problem_id, fold)}
    return out


def fold_indices(pop, fmap):
    f = pop.problem_id.astype(str).map(fmap).values
    return [(np.flatnonzero(f != k), np.flatnonzero(f == k)) for k in range(N_SPLITS)]


def load_or_make_folds(path, pop, pop_sha):
    if os.path.exists(path):
        saved = json.load(open(path))
        if saved.get("population_sha") == pop_sha:
            return saved["seeds"]
        print(f"  {path} belongs to another population; rebuilding it", flush=True)
    seeds = make_folds(pop, SEEDS)  # always all 3 seeds, so --quick and full share seed 0
    json.dump({"population_sha": pop_sha, "n_splits": N_SPLITS, "grouped_by": "problem_id",
               "stratified_on": "y (1 = wrong)", "seeds": seeds}, open(path, "w"), indent=1)
    return seeds


def perm_scores(build, y, groups, folds, rep):
    # one within-problem permutation, fixed hyperparameters, seed-0 folds
    yp = permute_within(y, groups, np.random.default_rng(10_000 + rep)) if rep >= 0 else y
    pairs = Pairs(yp, groups)
    out = {}
    for model, feat in PERM_ARMS:
        p, _ = M.oof(build, feat, model, yp, groups, folds, seed=0, C=M.PERM_C if model == "lr" else None)
        out[f"{model}/{feat}"] = pairs.auroc(p)["weighted"]
    return out


def saved_ok(path, cfg_sha):
    if not os.path.exists(path):
        return False
    col = pd.read_parquet(path, columns=["cfg_sha"]).cfg_sha
    return len(col) > 0 and (col == cfg_sha).all()


def fmt(sec):
    return f"{sec / 60:.1f} min" if sec >= 60 else f"{sec:.0f} s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--registry", default=D.REGISTRY)
    ap.add_argument("--out", default=None, help="default data/probe_cmp; --quick writes to <out>/quick")
    ap.add_argument("--quick", action="store_true", help="1 seed, 50 permutations, layer 20 only")
    ap.add_argument("--checkpoints", default="", help="comma-separated subset; default every registered one")
    ap.add_argument("--sources", default="", help="comma-separated registry names; default all present")
    ap.add_argument("--jobs", type=int, default=-1, help="workers for the permutation null")
    ap.add_argument("--min-problems", type=int, default=MIN_PROBLEMS)
    ap.add_argument("--perms", type=int, default=None, help="override the permutation count")
    a = ap.parse_args()
    seeds, n_perm, layers = ([0], 50, [PRIMARY_LAYER]) if a.quick else (SEEDS, N_PERM, LAYERS)
    n_perm = n_perm if a.perms is None else a.perms
    base = a.out or "data/probe_cmp"
    out = os.path.join(base, "quick") if a.quick else base  # a quick run never overwrites a full one
    os.makedirs(os.path.join(out, "preds"), exist_ok=True)
    t_start = time.time()

    print("loading and checking every registered acts/meta pair", flush=True)
    store = D.Store(a.data, reg=D.load_registry(a.registry),
                    names=[s for s in a.sources.split(",") if s] or None)
    missing = [l for l in layers if l not in store.layers()]
    if missing:
        sys.exit(f"layers {missing} are not in every source (common layers: {store.layers()})")
    asked = [c for c in a.checkpoints.split(",") if c]
    unknown = [c for c in asked if c not in store.ckpts]
    if unknown:
        sys.exit(f"checkpoints {unknown} not registered; available {store.ckpts}")
    ckpts = [c for c in store.ckpts if not asked or c in asked]  # always in token order

    if a.checkpoints:
        # a subset is a different population: keep its files away from the full run's
        base = os.path.join(base, "subset_" + "_".join(ckpts))
        out = os.path.join(base, "quick") if a.quick else base
        os.makedirs(os.path.join(out, "preds"), exist_ok=True)
    pop = D.population(store, ckpts)
    n_trunc = int((~store.traces.eligible).sum())
    n_prob = pop.problem_id.nunique()
    print(f"excluded {n_trunc} truncated traces (trunc or n_gen >= {D.MAX_NEW})")
    print(f"fixed population over {ckpts}: {len(pop)} traces, {n_prob} mixed problems, "
          f"{int(pop.y.sum())} wrong, {D.pair_total(pop)} correct x wrong pairs", flush=True)
    varying = D.varying_counts(store)
    print("secondary, varying population per checkpoint (counts only):")
    print(varying.to_string(index=False), flush=True)
    if n_prob < a.min_problems:
        sys.exit(f"only {n_prob} mixed problems survive the fixed population (< {a.min_problems}); "
                 "stopping rather than changing the design. see validate.py for candidate sets.")

    pop_sha = sha({"trace_ids": pop.trace_id.tolist(), "ckpts": ckpts})
    json.dump({"checkpoints": ckpts, "population_sha": pop_sha, "excluded_truncated": n_trunc,
               "excluded_trace_ids": store.traces.index[~store.traces.eligible].tolist(),
               "n_traces": len(pop), "n_problems": n_prob, "n_wrong": int(pop.y.sum()),
               "n_pairs": D.pair_total(pop), "trace_ids": pop.trace_id.tolist(),
               "problem_ids": pop.problem_id.tolist(), "y_wrong": pop.y.tolist(),
               "varying_secondary": varying.to_dict("records")},
              open(os.path.join(out, "population.json"), "w"), indent=1)
    # one folds.json per population, shared by --quick and the full run
    folds_by_seed = load_or_make_folds(os.path.join(base, "folds.json"), pop, pop_sha)
    folds = {s: fold_indices(pop, folds_by_seed[str(s)]) for s in seeds}

    cfg = {"version": VERSION, "population_sha": pop_sha, "seeds": seeds, "n_perm": n_perm,
           "layers": layers, "cs": M.CS, "svd_k": M.SVD_K, "top_k": M.TOP_K, "perm_c": M.PERM_C}
    cfg_sha = sha(cfg)
    ids, y, groups = pop.trace_id.values, pop.y.values, pop.problem_id.values
    print(f"\nrun config {cfg_sha}: {len(seeds)} seed(s), {n_perm} permutations, layers {layers}, "
          f"{N_SPLITS}-fold grouped CV, inner {M.N_INNER}-fold grouped CV for C in {M.CS}", flush=True)

    # tf-idf cost grows with text length, the hidden-state fits do not: project with both
    chars = {c: sum(map(len, store.text(c, ids))) for c in ckpts}
    paths = {c: (os.path.join(out, "preds", f"{c}.parquet"), os.path.join(out, "preds", f"{c}.perm.parquet"))
             for c in ckpts}
    # resume after a disconnect: a checkpoint is done when its files carry this config
    done = {c for c in ckpts if saved_ok(paths[c][0], cfg_sha) and (not n_perm or saved_ok(paths[c][1], cfg_sha))}
    todo = [c for c in ckpts if c not in done]
    first = None
    for i, ckpt in enumerate(ckpts):
        path, perm_path = paths[ckpt]
        if ckpt in done:
            print(f"[{ckpt}] predictions for this config already saved, skipping", flush=True)
            continue
        t0 = time.time()
        times = {}
        texts = store.text(ckpt, ids)
        E = store.len_features(ckpt, ids, texts)
        rows, perm_rows = [], []

        def keep(layer, model, feat, seed, p, cs):
            rows.append(pd.DataFrame({"ckpt": ckpt, "layer": layer, "model": model, "feature": feat,
                                      "seed": seed, "trace_id": ids, "problem_id": groups, "y": y,
                                      "p": p, "C": json.dumps(cs), "cfg_sha": cfg_sha}))

        tb = time.time()
        builders = {"len": M.Plain(E), "tfidf": M.Tfidf(texts)}
        for seed in seeds:
            for model, feat in SHARED_ARMS:
                p, cs = M.oof(builders[feat], feat, model, y, groups, folds[seed], seed)
                keep(NO_LAYER, model, feat, seed, p, cs)
        times["len+tfidf"] = time.time() - tb

        for layer in layers:
            X = store.hidden(ckpt, ids, layer)
            tb = time.time()
            for seed in seeds:
                build = M.Hidden(X, groups, extra=E, seed=seed)
                for model, feat in HIDDEN_ARMS:
                    p, cs = M.oof(build, feat, model, y, groups, folds[seed], seed)
                    keep(layer, model, feat, seed, p, cs)
            times[f"L{layer} fits"] = time.time() - tb
            if layer == PRIMARY_LAYER and n_perm:
                tb = time.time()
                build = M.Hidden(X, groups, extra=E, seed=0)
                obs = perm_scores(build, y, groups, folds[0], -1)  # also warms the SVD cache
                # one thread per worker: HGB's own OpenMP threads would oversubscribe 2 vCPUs
                with parallel_config(backend="loky", inner_max_num_threads=1):
                    null = Parallel(n_jobs=a.jobs)(
                        delayed(perm_scores)(build, y, groups, folds[0], r) for r in range(n_perm))
                for arm in obs:
                    perm_rows.append(pd.DataFrame({"ckpt": ckpt, "layer": layer, "arm": arm,
                                                   "rep": np.arange(-1, n_perm),
                                                   "score": [obs[arm]] + [n[arm] for n in null],
                                                   "cfg_sha": cfg_sha}))
                times[f"L{layer} perm x{n_perm}"] = time.time() - tb

        pd.concat(rows).to_parquet(path)
        if perm_rows:
            pd.concat(perm_rows).to_parquet(perm_path)
        el = time.time() - t0
        print(f"[{ckpt}] {fmt(el)}: " + ", ".join(f"{k} {fmt(v)}" for k, v in times.items()), flush=True)
        if first is None:
            first = ckpt
            text_t = times["len+tfidf"]
            fixed_t = el - text_t
            rest = [c for c in todo if c != ckpt]
            proj = time.time() - t_start + sum(fixed_t + text_t * chars[c] / max(chars[ckpt], 1) for c in rest)
            print(f"  projected total ~{fmt(proj)} for {len(todo)} checkpoint(s) to fit "
                  f"(tf-idf scaled by text length; budget 2 h: {'ok' if proj <= 7200 else 'OVER'})", flush=True)

    print(f"\nfitting done in {fmt(time.time() - t_start)}; building report", flush=True)
    report.build(out, store, pop, ckpts, layers, seeds, n_perm)
    print(f"total {fmt(time.time() - t_start)}")


if __name__ == "__main__":
    main()
