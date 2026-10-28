# fit the frozen monitors on ALL development data and pick their thresholds on development data
# only. CPU. one model per (monitor, checkpoint) in spec.MONITORS.
#   training rows at checkpoint c: probe_cmp's population at c alone, i.e. untruncated traces that
#     reach c, in problems with a correct and a wrong such trace (at 6144 this is the 342 / 74 cohort)
#   C (lr): probe_cmp's grouped inner CV, run once on all training rows; HGB is fixed (models.py)
#   threshold: flag a share spec.Q_DEV of dev LIVE traces at c (still generating, </think> not yet
#     written), all problems. training traces enter with their out-of-fold score (5 grouped folds,
#     seed spec.FIT_SEED), the rest with the final model's score; they are not in its training set.
# writes <out>/<monitor>@<c>.joblib and <out>/manifest.json (C, threshold, dev OOF AUROC, every hash)
#   python src/confirm/fit_monitors.py --data data --out data/confirm/monitors
import argparse, hashlib, json, os, sys, time
import joblib
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "probe_cmp"))
import spec  # noqa: E402
from monitors import Monitor, oof_scores, threshold  # noqa: E402
import data as D  # noqa: E402
import run as R  # noqa: E402
import stats as S  # noqa: E402


def inputs(store, ckpt, ids, layer):
    rows = store.rows(ckpt, ids)
    texts = store.text(ckpt, ids)
    X = store.hidden(ckpt, ids, layer) if layer is not None else None
    return X, texts, rows.tokens.values.astype(float)


def live_rows(store, ckpt):
    # every trace with a row at ckpt (all problems, truncated ones too) whose prefix has not closed
    # its thinking; rows exist only for pos <= retokenized length, so the trace is still generating
    tab = store.table[store.table.ckpt == ckpt]
    ids = tab.trace_id.tolist()
    texts = store.text(ckpt, ids)
    keep = [spec.THINK_END not in t for t in texts]
    return [i for i, k in zip(ids, keep) if k]


def auroc_ci(y, g, s, n_boot=spec.N_BOOT):
    pairs = S.Pairs(y, g)
    w, _ = S.boot_within(pairs, s, S.problem_draws(len(pairs.pids), n_boot, spec.BOOT_SEED))
    lo, hi = S.ci(w)
    return pairs.auroc(s)["weighted"], lo, hi


def fit_one(store, name, ckpt, log=print):
    model, feature, layer, _ = spec.MONITORS[name]
    pop = D.population(store, [ckpt])
    ids, y, g = pop.trace_id.values, pop.y.values, pop.problem_id.values
    X, texts, tokens = inputs(store, ckpt, ids, layer)
    folds = R.fold_indices(pop, R.make_folds(pop, [spec.FIT_SEED])[str(spec.FIT_SEED)])
    p_oof, fold_C = oof_scores(model, feature, y, g, folds, spec.FIT_SEED, X, texts, tokens)
    final = Monitor(model, feature, seed=spec.FIT_SEED).fit(y, g, X, texts, tokens)
    auc, lo, hi = auroc_ci(y, g, p_oof)

    live = live_rows(store, ckpt)
    oof_map = dict(zip(ids, p_oof))
    rest = [t for t in live if t not in oof_map]
    s_rest = final.score(*inputs(store, ckpt, rest, layer)) if rest else np.array([])
    s_live = np.array([oof_map[t] for t in live if t in oof_map] + list(s_rest))
    tau = threshold(s_live, spec.Q_DEV)
    log(f"  {name}@{ckpt}: train {len(ids)} traces / {len(np.unique(g))} problems, C {final.C}, "
        f"dev OOF within-problem AUROC {auc:.3f} [{lo:.3f}, {hi:.3f}]; live {len(live)} "
        f"({len(live) - len(rest)} OOF + {len(rest)} out-of-sample), threshold {tau:.4f} "
        f"flags {np.mean(s_live > tau):.3f}")
    info = {"monitor": name, "model": model, "feature": feature, "layer": layer, "checkpoint": int(ckpt),
            "C": final.C, "fold_C": fold_C, "train_traces": int(len(ids)), "train_problems": int(len(np.unique(g))),
            "train_wrong": int(y.sum()), "train_trace_ids_sha": hashlib.sha256(json.dumps(list(ids)).encode()).hexdigest(),
            "dev_oof_auroc": auc, "dev_oof_ci": [lo, hi], "dev_live_traces": len(live),
            "threshold": tau, "q_dev": spec.Q_DEV, "dev_live_flag_share": float(np.mean(s_live > tau))}
    oof = pd.DataFrame({"trace_id": ids, "problem_id": g, "y": y, "p_oof": p_oof, "monitor": name, "checkpoint": int(ckpt)})
    return final, info, oof


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data", help="traces_B.jsonl, acts_B.npy (p32k), meta_B.parquet")
    ap.add_argument("--registry", default=D.REGISTRY)
    ap.add_argument("--out", default="data/confirm/monitors")
    ap.add_argument("--only", nargs="*", default=None, help="monitor names (default: all in spec)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    store = D.Store(a.data, reg=D.load_registry(a.registry), names=["B_fixed"])
    missing = [str(c) for c in spec.CHECKPOINTS if str(c) not in store.ckpts]
    if missing:
        sys.exit(f"checkpoints {missing} not in the dev activations (need the p32k acts_B / meta_B)")
    src, meta = store.sources["B_fixed"]
    files = {"traces": os.path.join(a.data, store.reg["traces"]), "acts": os.path.join(a.data, src["acts"]),
             "meta": os.path.join(a.data, src["meta"])}
    manifest = {"spec_sha256": spec.spec_sha(), "monitor_spec_sha256": spec.monitor_sha(), "written": time.strftime("%Y-%m-%d %H:%M:%S"),
                "inputs": {k: {"path": os.path.basename(v), "sha256": spec.file_sha(v)} for k, v in files.items()},
                "code": spec.code_sha(), "versions": _versions(), "monitors": {}}
    oofs = []
    for name, (_, _, _, ckpts) in spec.MONITORS.items():
        if a.only and name not in a.only:
            continue
        for c in ckpts:
            final, info, oof = fit_one(store, name, str(c))
            path = os.path.join(a.out, f"{name}@{c}.joblib")
            joblib.dump(final, path)
            info["file"] = os.path.basename(path)
            info["sha256"] = spec.file_sha(path)
            manifest["monitors"][f"{name}@{c}"] = info
            oofs.append(oof)
    pd.concat(oofs).to_parquet(os.path.join(a.out, "dev_oof.parquet"))
    json.dump(manifest, open(os.path.join(a.out, "manifest.json"), "w"), indent=1, default=float)
    print(f"wrote {len(manifest['monitors'])} monitors and manifest.json to {a.out} in {(time.time() - t0) / 60:.1f} min; "
          f"spec {manifest['spec_sha256'][:16]}")


def _versions():
    import scipy, sklearn
    return {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
            "sklearn": sklearn.__version__, "scipy": scipy.__version__, "joblib": joblib.__version__}


if __name__ == "__main__":
    main()
