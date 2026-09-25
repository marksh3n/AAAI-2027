# Step 7: the iteration-1 "0.887 vs 0.854" check. BETWEEN-problem (one trace per problem):
# 367 traces of 367 problems, so this AUROC mostly measures difficulty. the within-problem
# version of the same comparison is the end row of run.py's report, and that is the primary one.
# 0.887 = Colab cell 41: layer 16, StandardScaler + LR(C=0.01), StratifiedKFold(5, shuffle, 0),
#         cross_val_predict over all 367 last-step rows. 0.854 = cell 26: raw n_steps, unfitted.
#   Colab: %run src/probe_cmp/fetch.py --iter1, then python src/probe_cmp/iter1.py --out <dir>
#   (or --acts /path/to/iteration-1/acts.npy; the 367 rows are sliced and cached in data/iter1/)
import argparse, json, os, sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from stats import boot_p, ci, delong

LAYERS = [-1, 0, 4, 8, 12, 16, 20, 24, 27]  # iteration-1 extraction (steps mode)
LAYER, C = 16, 0.01
EXPECT = {"probe": "0.887", "length": "0.854"}
LABEL = "between-problem (one trace per problem)"
N_BOOT, N_SEEDS = 2000, 10


def last_rows(meta):
    if not (meta.row.values == np.arange(len(meta))).all():
        sys.exit("iteration-1 meta.row is not 0..n-1")
    last = meta[meta.step_idx == meta.n_steps - 1]
    if len(last) != 367 or last.trace_id.duplicated().any() or last.problem_id.duplicated().any():
        sys.exit(f"expected 367 last-step rows, one per problem; got {len(last)}")
    return last


def load_x(a, last):
    cache = os.path.join(a.dir, f"last_L{LAYER}.npy")
    ids = os.path.join(a.dir, f"last_L{LAYER}.trace_ids.json")
    if a.acts:
        acts = np.load(a.acts, mmap_mode="r")
        if acts.shape != (len(pd.read_parquet(a.meta)), len(LAYERS), 1536):
            sys.exit(f"acts {acts.shape} does not match iteration-1 meta")
        # same indexing as cell 41: acts[meta.index[last]], then the layer slot
        x = np.asarray(acts[last.row.values])[:, LAYERS.index(LAYER), :]
        np.save(cache, x)
        json.dump(last.trace_id.tolist(), open(ids, "w"))
        print(f"sliced {x.shape} from {a.acts} -> {cache}")
    elif os.path.exists(cache):
        if json.load(open(ids)) != last.trace_id.tolist():
            sys.exit(f"{cache} rows are not in iteration-1 last-step order")
        x = np.load(cache)
    else:
        sys.exit(f"no {cache}; pass --acts <iteration-1 acts.npy> once to slice it")
    return x.astype(np.float32)


def probe_oof(x, correct, seed):
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    pipe = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))
    return cross_val_predict(pipe, x, correct, cv=cv, method="predict_proba")[:, 1]


def boot_traces(y, a, b, seed=0):
    # paired, class-stratified bootstrap over traces (as cell 43)
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    d = np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        d[i] = roc_auc_score(y[idx], a[idx]) - roc_auc_score(y[idx], b[idx])
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/iter1", help="holds meta.parquet and the cached slice")
    ap.add_argument("--meta", default=None)
    ap.add_argument("--acts", default=None, help="iteration-1 acts.npy (58813, 9, 1536); needed once")
    ap.add_argument("--out", default="data/probe_cmp")
    a = ap.parse_args()
    a.meta = a.meta or os.path.join(a.dir, "meta.parquet")
    if a.acts is None and os.path.exists(os.path.join(a.dir, "acts.npy")):  # put there by fetch.py --iter1
        a.acts = os.path.join(a.dir, "acts.npy")
    os.makedirs(a.out, exist_ok=True)
    last = last_rows(pd.read_parquet(a.meta))
    x = load_x(a, last)
    correct = last.is_correct.astype(int).values
    wrong = 1 - correct
    steps = last.n_steps.values.astype(float)

    # 1. reproduce both published numbers before testing anything
    p = probe_oof(x, correct, 0)
    got = {"probe": f"{roc_auc_score(correct, p):.3f}", "length": f"{roc_auc_score(wrong, steps):.3f}"}
    print(f"reproduce: probe {got['probe']} (expect {EXPECT['probe']}), "
          f"length {got['length']} (expect {EXPECT['length']})")
    result = {"label": LABEL, "n_traces": len(last), "n_wrong": int(wrong.sum()), "layer": LAYER, "C": C,
              "expected": EXPECT, "reproduced_values": got, "reproduced": got == EXPECT}
    path = os.path.join(a.out, "delong.json")
    if got != EXPECT:
        result["plain"] = [f"NOT REPRODUCED: got probe {got['probe']} / length {got['length']}, expected "
                           f"{EXPECT['probe']} / {EXPECT['length']}. Row alignment of the iteration-1 slice is "
                           "suspect; nothing below was run."]
        json.dump(result, open(path, "w"), indent=2)
        sys.exit(result["plain"][0])

    # 2. DeLong on the seed-0 OOF scores, both oriented to P(wrong)
    probe_wrong = 1 - p
    dl = delong(wrong, probe_wrong, steps)
    # 3. paired bootstrap over traces
    bd = boot_traces(wrong, probe_wrong, steps)
    lo, hi = ci(bd)
    # 4. fold-seed spread: DeLong and the trace bootstrap treat the predictions as fixed, so
    # they miss the variance from refitting; redrawing the folds shows it
    len_auc = roc_auc_score(wrong, steps)
    per_seed = []
    for s in range(N_SEEDS):
        ps = 1 - probe_oof(x, correct, s)
        per_seed.append({"seed": s, "probe": roc_auc_score(wrong, ps), "delta": roc_auc_score(wrong, ps) - len_auc,
                         "delong_p": delong(wrong, ps, steps)["p"]})
    dels = np.array([r["delta"] for r in per_seed])
    result.update({
        "delong": dl,
        "bootstrap_traces": {"delta": float(np.mean(bd)), "ci": [lo, hi], "p": boot_p(bd), "n_boot": N_BOOT},
        "fold_seeds": {"n": N_SEEDS, "delta_mean": float(dels.mean()), "delta_min": float(dels.min()),
                       "delta_max": float(dels.max()), "frac_hidden_gt_length": float((dels > 0).mean()),
                       "per_seed": per_seed},
    })
    sig = dl["ci"][0] > 0 and lo > 0 and dels.min() > 0
    result["plain"] = [
        f"Label: {LABEL}. {len(last)} iteration-1 traces, {int(wrong.sum())} wrong; probe = layer {LAYER}, C={C}, "
        "5-fold OOF. Layer and C were picked on these same predictions (cell 41 sweep), so the probe side is optimistic.",
        f"Reproduced exactly: probe {got['probe']}, step count {got['length']}.",
        f"DeLong: Δ = {dl['delta']:+.3f} [{dl['ci'][0]:+.3f}, {dl['ci'][1]:+.3f}], p = {dl['p']:.3f} "
        "(treats the OOF predictions as fixed, so this interval is too narrow).",
        f"Paired bootstrap over traces: Δ [{lo:+.3f}, {hi:+.3f}], p = {boot_p(bd):.3f}.",
        f"Across {N_SEEDS} fold seeds: Δ mean {dels.mean():+.3f}, range [{dels.min():+.3f}, {dels.max():+.3f}]; "
        f"hidden > length in {int((dels > 0).sum())}/{N_SEEDS}.",
        ("In plain words: between problems, the probe beats counting steps by a small but consistent margin."
         if sig else
         "In plain words: between problems, the probe is not reliably better than counting steps; the 0.033 gap "
         "is within the noise of one test set and one fold split."),
    ]
    json.dump(result, open(path, "w"), indent=2, default=float)
    print("\n".join(result["plain"]))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
