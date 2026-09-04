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
from scipy.sparse import csr_matrix, hstack

from cache import StaleCache, check_dataset_manifest, short

LAYERS = [-1, 0, 4, 8, 12, 16, 20, 24, 27]
POSITIONS = [128, 256, 512, 1024, 2048, 4096]
N_OUTER = 5
N_INNER = 3
CS = [0.001, 0.01, 0.1, 1.0]
MIN_PAIRS = 20
SEED = 0
# the ablation compares two arms on identical folds:
#   TR  = prefix text (tf-idf) + restart density
#   TRH = the same, plus hidden states
# n_gen is deliberately absent from both: it is the FINAL trace length, which a monitor
# reading a prefix at token 128 cannot know. it was an oracle, not a baseline.
ARMS = ("TR", "TRH")
ABLATION_LAYER = 16
ABLATION_SVD = 50
N_SEEDS = 5
NULL_REPS = 10


def folds(ids, k):
    ids = np.asarray(ids)
    return [
        (np.concatenate([ids[j::k] for j in range(k) if j != i]), ids[i::k]) for i in range(k)
    ]


def pair_counts(df, with_ids=False):
    # (wrong, correct) pairs within one problem: difficulty and length cancel
    out, pids = [], []
    for pid, g in df.groupby("problem_id"):
        a, b = g.p[g.y == 1].values, g.p[g.y == 0].values
        if len(a) and len(b):
            d = a[:, None] - b[None, :]
            out.append(((d > 0).sum() + 0.5 * (d == 0).sum(), d.size))
            pids.append(pid)
    counts = np.array(out, dtype=float).reshape(-1, 2)
    return (counts, pids) if with_ids else counts


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


def yield_table(meta):
    # later positions are length-selected subsamples: a trace only has a row at 4096 if it
    # ran that long, and wrong traces are far longer here. the denominators are not
    # comparable across positions, so print them next to every AUROC.
    per = meta.groupby(["pos", "trace_id"], as_index=False).agg(
        y=("y", "first"), pid=("problem_id", "first")
    )
    rows = []
    for pos, g in per.groupby("pos"):
        rows.append(
            {
                "pos": pos,
                "traces": len(g),
                "correct": int((g.y == 0).sum()),
                "wrong": int((g.y == 1).sum()),
                "problems": g.pid.nunique(),
                "mixed": int((g.groupby("pid").y.nunique() > 1).sum()),
            }
        )
    return pd.DataFrame(rows)


def split_features(cell, hidden, tr_ids, te_ids, svd):
    # every transform is fitted on the training problems only: the tf-idf vocabulary and
    # the SVD basis never see a test problem. built once per split and shared by both arms
    # so they differ in features alone, never in the split or the text representation.
    tr = cell[cell.problem_id.isin(tr_ids)]
    te = cell[cell.problem_id.isin(te_ids)]
    if tr.y.nunique() < 2 or not len(te):
        return None
    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    text_tr, text_te = vec.fit_transform(tr.prefix), vec.transform(te.prefix)
    rest_tr = csr_matrix(tr.restarts.values.reshape(-1, 1).astype(np.float64))
    rest_te = csr_matrix(te.restarts.values.reshape(-1, 1).astype(np.float64))
    tr_X = hstack([text_tr, rest_tr]).tocsr()
    te_X = hstack([text_te, rest_te]).tocsr()
    h_tr, h_te = hidden[tr.row.values], hidden[te.row.values]
    if svd:
        k = min(svd, h_tr.shape[1] - 1, max(h_tr.shape[0] - 1, 1))
        red = TruncatedSVD(n_components=k, random_state=SEED).fit(h_tr)
        h_tr, h_te = red.transform(h_tr), red.transform(h_te)
    return (
        {
            "TR": (tr_X, te_X),
            "TRH": (
                hstack([tr_X, csr_matrix(h_tr)]).tocsr(),
                hstack([te_X, csr_matrix(h_te)]).tocsr(),
            ),
        },
        tr,
        te,
    )


def score_arm(X_tr, X_te, tr, te, C):
    return te.assign(p=fit_score(X_tr, tr.y.values, X_te, C))


def pick_cs(cell, hidden, tr_ids, mixed, svd):
    # C is chosen inside the training problems only, per arm, on identical inner splits
    per = {(a, C): [] for a in ARMS for C in CS}
    for i_tr, i_te in folds(tr_ids, N_INNER):
        built = split_features(cell, hidden, i_tr, [p for p in i_te if p in mixed], svd)
        if built is None:
            continue
        mats, tr, te = built
        for a in ARMS:
            for C in CS:
                per[(a, C)].append(auroc(pair_counts(score_arm(*mats[a], tr, te, C))))
    out = {}
    for a in ARMS:
        means = {
            C: np.nanmean(per[(a, C)])
            for C in CS
            if per[(a, C)] and not np.all(np.isnan(per[(a, C)]))
        }
        out[a] = max(means, key=means.get) if means else None
    return out


def run_position(cell, hidden, outer, mixed, svd):
    got = {a: [] for a in ARMS}
    for tr_ids, te_ids in outer:
        te_mixed = [p for p in te_ids if p in mixed]
        if not te_mixed:
            continue
        cs = pick_cs(cell, hidden, tr_ids, mixed, svd)
        built = split_features(cell, hidden, tr_ids, te_mixed, svd)
        if built is None:
            continue
        mats, tr, te = built
        for a in ARMS:
            if cs[a] is not None:
                got[a].append(score_arm(*mats[a], tr, te, cs[a]))
    return {a: (pd.concat(v) if v else None) for a, v in got.items()}


def paired_delta(a_scored, b_scored, n=2000):
    # resamples the SAME problems for both arms, so the difference is paired and the
    # between-problem variance that both arms share cancels instead of inflating the CI
    ca, pa = pair_counts(a_scored, True)
    cb, pb = pair_counts(b_scored, True)
    ib = {q: i for i, q in enumerate(pb)}
    common = [q for q in pa if q in ib]
    if len(common) < 2:
        return np.nan, [np.nan, np.nan]
    A = ca[[pa.index(q) for q in common]]
    B = cb[[ib[q] for q in common]]
    rng = np.random.default_rng(SEED)
    diffs = []
    for _ in range(n):
        i = rng.integers(0, len(common), len(common))
        diffs.append(auroc(B[i]) - auroc(A[i]))
    return float(auroc(B) - auroc(A)), [
        float(np.percentile(diffs, 2.5)),
        float(np.percentile(diffs, 97.5)),
    ]


def permute_within_problem(hidden, cell, rng):
    # destroys the within-problem association between activation and trace while keeping
    # the feature distribution, the dimensionality and any between-problem structure.
    # this is the null the within-problem AUROC is actually measured against.
    out = hidden.copy()
    for _, g in cell.groupby("problem_id"):
        rows = g.row.values
        out[rows] = hidden[rng.permutation(rows)]
    return out


def null_deltas(cell, hidden, ids, mixed, svd, reps, n_seeds):
    # concatenating 50 uninformative columns still moves the fit: at this sample size that
    # is worth several AUROC points, and the paired bootstrap over problems cannot see it
    # because it resamples problems, not model fits. so measure it directly.
    out = []
    for rep in range(reps):
        h = permute_within_problem(hidden, cell, np.random.default_rng(1000 + rep))
        order = ids.copy()
        np.random.default_rng(SEED + (rep % max(n_seeds, 1))).shuffle(order)
        sc = run_position(cell, h, folds(order, N_OUTER), mixed, svd)
        if any(sc[a] is None for a in ARMS):
            continue
        out.append(auroc(pair_counts(sc["TRH"])) - auroc(pair_counts(sc["TR"])))
    return out


def ablation(meta, hidden, mixed, svd, n_seeds, null_reps=0):
    # the whole outer split is redrawn per seed: with 38 mixed problems over 5 folds the
    # fold assignment is itself a large source of variance, which one seed cannot show.
    ids = np.array(sorted(meta.problem_id.unique()))
    out = []
    print(f"{'pos':>5}{'pairs':>7}{'probs':>6}"
          f"{'TR':>16}{'TRH':>16}{'delta':>16}{'null delta':>16}{'z':>7}{'CI':>20}",
          flush=True)
    for pos in POSITIONS:
        cell = meta[meta.pos == pos]
        if not len(cell):
            continue
        per_seed = {a: [] for a in ARMS}
        kept = {a: [] for a in ARMS}
        for seed in range(n_seeds):
            order = ids.copy()
            np.random.default_rng(SEED + seed).shuffle(order)
            scored = run_position(cell, hidden, folds(order, N_OUTER), mixed, svd)
            if any(scored[a] is None for a in ARMS):
                continue
            for a in ARMS:
                per_seed[a].append(auroc(pair_counts(scored[a])))
                kept[a].append(scored[a])
        if not per_seed["TR"]:
            print(f"{pos:>5}  skipped: no fold had both outcomes", flush=True)
            continue
        # one score per trace, averaged over seeds, then bootstrapped over problems
        pool = {
            a: pd.concat(kept[a])
            .groupby(["problem_id", "trace_id"], as_index=False)
            .agg(y=("y", "first"), p=("p", "mean"))
            for a in ARMS
        }
        counts = pair_counts(pool["TRH"])
        if counts[:, 1].sum() < MIN_PAIRS:
            print(f"{pos:>5}  skipped: {int(counts[:, 1].sum())} pairs, too few to read", flush=True)
            continue
        d_pool, ci = paired_delta(pool["TR"], pool["TRH"])
        null = null_deltas(cell, hidden, ids, mixed, svd, null_reps, n_seeds)
        sd = lambda v: float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
        gap = np.array(per_seed["TRH"]) - np.array(per_seed["TR"])
        row = {
            "pos": pos,
            "pairs": int(counts[:, 1].sum()),
            "n_problems": len(counts),
            "n_seeds": len(per_seed["TR"]),
            "TR_mean": float(np.mean(per_seed["TR"])),
            "TR_sd": sd(per_seed["TR"]),
            "TRH_mean": float(np.mean(per_seed["TRH"])),
            "TRH_sd": sd(per_seed["TRH"]),
            "delta_mean": float(gap.mean()),
            "delta_sd": sd(gap),
            "TR_pooled": auroc(pair_counts(pool["TR"])),
            "TRH_pooled": auroc(pair_counts(pool["TRH"])),
            "delta_pooled": d_pool,
            "delta_ci": ci,
            "null_deltas": null,
            "null_mean": float(np.mean(null)) if null else np.nan,
            "null_sd": sd(null),
            "per_seed_TR": per_seed["TR"],
            "per_seed_TRH": per_seed["TRH"],
        }
        # how far the observed gap sits from the gap permuted activations produce, in
        # units of the null's own spread. a percentile band needs far more reps than this
        # to be stable; the standardized distance is usable from ~10. it is a guide, not a
        # formal test -- the null sd is itself estimated from `null_reps` draws.
        row["null_z"] = (
            (row["delta_mean"] - row["null_mean"]) / row["null_sd"]
            if len(null) >= 2 and row["null_sd"] > 1e-9 else np.nan
        )
        row["beats_null"] = bool(abs(row["null_z"]) >= 2) if np.isfinite(row["null_z"]) else None
        out.append(row)
        cells = [
            f"{row['TR_mean']:.3f} +-{row['TR_sd']:.3f}",
            f"{row['TRH_mean']:.3f} +-{row['TRH_sd']:.3f}",
            f"{row['delta_mean']:+.3f} +-{row['delta_sd']:.3f}",
            f"{row['null_mean']:+.3f} +-{row['null_sd']:.3f}" if null else "not run",
            f"{row['null_z']:+.1f}" if np.isfinite(row["null_z"]) else "-",
            f"[{ci[0]:+.3f}, {ci[1]:+.3f}]",
        ]
        print(f"{pos:>5}{row['pairs']:>7}{row['n_problems']:>6}"
              f"{cells[0]:>16}{cells[1]:>16}{cells[2]:>16}{cells[3]:>16}"
              f"{cells[4]:>7}{cells[5]:>20}", flush=True)
    return out


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
    parser.add_argument("--strict", action="store_true",
                        help="also re-hash acts.npy against the dataset manifest")
    parser.add_argument("--ablation", action="store_true",
                        help="run TR vs TRH on matched folds over several seeds "
                             "instead of the hidden-only layer sweep")
    parser.add_argument("--layer", type=int, default=ABLATION_LAYER,
                        help="single layer for the ablation; no inner layer sweep, because "
                             "38 mixed problems cannot support nested selection")
    parser.add_argument("--svd-hidden", type=int, default=ABLATION_SVD,
                        help="reduce the hidden block to this many components before "
                             "concatenating (0 to keep all 1536); fitted per training fold")
    parser.add_argument("--seeds", type=int, default=N_SEEDS,
                        help="number of outer-fold assignments to average over")
    parser.add_argument("--null-reps", type=int, default=NULL_REPS,
                        help="permutation reps calibrating how large a delta uninformative "
                             "hidden columns produce on their own; 0 to skip")
    args = parser.parse_args()

    s = f"_{args.tag}" if args.tag else ""
    meta_path, acts_path = f"data/meta{s}.parquet", f"data/acts{s}.npy"
    meta = pd.read_parquet(meta_path)
    acts = np.load(acts_path, mmap_mode="r")
    # the labels here are joined to the activations by row index alone, so establish first
    # that this meta and this acts were written by one extraction under one config
    dm = check_dataset_manifest(meta_path, acts_path, meta.trace_id.tolist(), acts.shape,
                                strict=args.strict)
    if not (meta.row.values == np.arange(len(meta))).all():
        raise StaleCache("meta.row is not 0..n-1, so it cannot index acts.npy")
    # LAYERS and POSITIONS are duplicated here to stay torch-free; check the copies agree
    # with the ones the activations were actually extracted at
    for name, mine in [("layers", LAYERS), ("positions", POSITIONS)]:
        theirs = dm["config"].get(name)
        if theirs != mine:
            raise StaleCache(f"probe_fixed {name.upper()} = {mine} but acts{s}.npy was "
                             f"extracted with {theirs}")
    print(f"dataset config {short(dm['config_hash'])}, extracted {dm['written']}")
    meta["y"] = (~meta.is_correct.astype(bool)).astype(int)
    feats_by_layer = [np.asarray(acts[:, i, :], dtype=np.float32) for i in range(len(LAYERS))]

    per_trace = meta.groupby("trace_id").agg(y=("y", "first"), pid=("problem_id", "first"))
    mixed = {p for p, g in per_trace.groupby("pid") if g.y.nunique() > 1}
    print(f"{len(mixed)} mixed-outcome problems of {meta.problem_id.nunique()}")
    if len(mixed) < 30:
        print("WARNING: under 30 mixed problems, the within-problem test is underpowered")

    print("\nyield by position (later positions are length-selected subsamples)")
    print(yield_table(meta).to_string(index=False))

    if args.ablation:
        if args.layer not in LAYERS:
            raise SystemExit(f"--layer {args.layer} not extracted; available: {LAYERS}")
        hidden = feats_by_layer[LAYERS.index(args.layer)]
        print(f"\nablation: TR (prefix tf-idf + restart density) vs TRH (+ hidden states)")
        print(f"layer {args.layer}, hidden SVD {args.svd_hidden or 'off'}, "
              f"{args.seeds} fold seeds, C chosen per arm by {N_INNER}-fold inner CV.")
        print("n_gen is excluded: it is the final trace length, unknown to an online monitor.")
        print(f"null delta: {args.null_reps} reps with activations permuted within problem, "
              "i.e. what the gap\nlooks like when the hidden block carries nothing. judge "
              "delta by z (distance from that\nnull in its own sd), not by the CI -- the CI "
              "resamples problems and is blind to the\nfitting noise that 50 extra columns "
              "introduce.")
        if 0 < args.null_reps < 10:
            print(f"WARNING: {args.null_reps} null reps is too few to estimate the null sd; "
                  "z will be unstable and can\nread large by accident. use at least 10.")
        print(flush=True)
        rows = ablation(meta, hidden, mixed, args.svd_hidden, args.seeds, args.null_reps)
        pd.DataFrame(rows).to_json(f"data/ablation{s}.json", orient="records", indent=2)
        print(f"\nwrote data/ablation{s}.json")
        return

    ids = np.array(sorted(meta.problem_id.unique()))
    np.random.default_rng(SEED).shuffle(ids)
    outer = folds(ids, N_OUTER)
    print(f"{N_OUTER}-fold outer CV, {N_INNER}-fold inner over layer x C"
          + (f", SVD to {args.svd} components\n" if args.svd else "\n"))

    print(f"{'pos':>5}{'pairs':>7}{'(layer, C)':>34}{'auroc':>7}{'95% CI':>16}{'d':>7}"
          f"{'tfidf':>8}{'restart':>8}{'shuf':>7}")
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
            "shuffled": auroc(pair_counts(sh)),
        }
        out.append(row)
        ci = row["ci"]
        print(f"{pos:>5}{row['pairs']:>7}{str(picked):>34}{row['auroc']:>7.3f}"
              f"{f'[{ci[0]:.2f}, {ci[1]:.2f}]':>16}{row['d']:>7.2f}"
              f"{row['tfidf']:>8.3f}{row['restarts']:>8.3f}"
              f"{row['shuffled']:>7.3f}", flush=True)

    pd.DataFrame(out).to_json(f"data/fixed_summary{s}.json", orient="records", indent=2)
    print(f"\nwrote data/fixed_summary{s}.json")


if __name__ == "__main__":
    main()
