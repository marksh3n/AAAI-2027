"""E0 full results table: every extracted layer x every checkpoint, with baselines side by side.

Adapted from the drafted layer_sweep.py to the real files:
  acts_full.npy / meta_full.parquet / full_config.json   extract_full.py (BOS matched to generation)
  traces_B.jsonl                                          labels and line numbers
  recovery_ceiling                                        the Modal ceiling rollouts (row format 3)

Population per checkpoint (as probe_cmp): untruncated traces that reach it, in problems with both a
correct and a wrong such trace. Later checkpoints are length-selected; counts are in the table.
Every score is out-of-fold (GroupKFold by problem) with a fixed C, so no cell is tuned on test data;
the grid is descriptive and a headline number still needs nested selection. Metric: within-problem
pair AUROC, the share of (correct, wrong) pairs of one problem where the correct attempt scores higher
(numerically the same as the earlier P(wrong) orientation). CIs: problem bootstrap.

Ceiling block: the r(t) rollouts exist at 512 / 2048 / 4096 only, and only the main-run problems
(traces_B lines 480+) were rolled out from the prompt their source traces were generated with, so the
ceiling and every other arm are scored there on the same traces (52 problems).
"""
import argparse, json, os
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True, help="dir with acts_full.npy, meta_full.parquet, full_config.json")
ap.add_argument("--traces", required=True)
ap.add_argument("--rollouts", required=True, help="recovery_ceiling")
ap.add_argument("--out", required=True)
ap.add_argument("--pilot-lines", type=int, default=480)
ap.add_argument("--min-problems", type=int, default=10, help="skip a checkpoint with fewer mixed problems")
ap.add_argument("--layers", default="", help="comma list to run a subset, e.g. 16,20")
ap.add_argument("--jobs", type=int, default=-1)
a = ap.parse_args()

C, K, NB, NDRAW, SEED = 0.1, 5, 1000, 20, 0
CEIL = {"p512": "512", "p2048": "2048", "p4096": "4096"}
cfg = json.load(open(os.path.join(a.data, "full_config.json")))
LAYERS = cfg["layers"]
RUN_LAYERS = [int(v) for v in a.layers.split(",")] if a.layers else LAYERS


def pair_stats(pid, y, s):
    w, n = [], []
    for p in np.unique(pid):
        m = pid == p
        c, x = s[m & (y == 1)], s[m & (y == 0)]
        if len(c) and len(x):
            d = c[:, None] - x[None, :]
            w.append((d > 0).sum() + 0.5 * (d == 0).sum())
            n.append(d.size)
    return np.array(w, float), np.array(n, float)


def summarize(pid, y, s, seed=SEED):
    ok = ~np.isnan(s)
    w, n = pair_stats(pid[ok], y[ok], s[ok])
    if n.sum() == 0:
        return dict(auroc=np.nan, lo=np.nan, hi=np.nan, pairs=0, n_problems=0)
    idx = np.random.default_rng(seed).integers(0, len(w), (NB, len(w)))  # same draws for every arm
    bs = w[idx].sum(1) / n[idx].sum(1)
    return dict(auroc=w.sum() / n.sum(), lo=np.quantile(bs, .025), hi=np.quantile(bs, .975),
                pairs=int(n.sum()), n_problems=len(w))


def oof(X, t, pid, make, proba=True):
    s = np.full(len(t), np.nan)
    for tr, te in GroupKFold(K).split(X, t, pid):
        if proba and len(np.unique(t[tr])) < 2:
            continue
        m = make().fit(X[tr], t[tr])
        s[te] = m.predict_proba(X[te])[:, 1] if proba else m.predict(X[te])
    return s


def shuffle_within(y, pid, seed):
    rng, ys = np.random.default_rng(seed), y.copy()
    for p in np.unique(pid):
        i = np.where(pid == p)[0]
        ys[i] = rng.permutation(ys[i])
    return ys


probe = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))
soft = lambda: make_pipeline(StandardScaler(), Ridge(alpha=1 / C))
tfidf = lambda: make_pipeline(TfidfVectorizer(min_df=2, ngram_range=(1, 2), max_features=50000),
                              LogisticRegression(C=1.0, max_iter=3000))
lenlr = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000))


def paragraphs(text):
    # \n\n paragraphs inside <think>, plus one for the answer once </think> has appeared (probe_cmp's len)
    cut = text.find("</think>")
    think = text if cut < 0 else text[:cut]
    return sum(1 for c in think.split("\n\n") if c.strip()) + (cut >= 0)


def load_ceiling(path):
    # (trace_id, checkpoint) -> 0/1 outcome per continuation; rounds pool, a rewritten sample keeps the last
    last = {}
    for line in open(path):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue  # a torn last line
        if r.get("trace_id") is None or r["checkpoint"] not in CEIL:
            continue
        for s in r["samples"]:
            last[(r["trace_id"], CEIL[r["checkpoint"]], r["round"], s["sample_idx"])] = float(bool(s["correct"]))
    out = {}
    for (tid, c, _, _), v in last.items():
        out.setdefault((tid, c), []).append(v)
    return {k: np.array(v) for k, v in out.items()}


def layer_cells(li, L, X, y, pid, ckpt, sub, rr):
    # one layer: probe and a one-shuffle sanity probe on the checkpoint population; on the ceiling
    # subset, the same out-of-fold probe scores plus a ridge probe fitted to r_hat inside the subset
    s = oof(X, y, pid, probe)
    ys = shuffle_within(y, pid, SEED + li)
    rows = [dict(block="grid", ckpt=ckpt, arm="probe", layer=L, **summarize(pid, y, s)),
            dict(block="grid", ckpt=ckpt, arm="probe_shuffled_once", layer=L, **summarize(pid, ys, oof(X, ys, pid, probe)))]
    if sub is not None:
        rows.append(dict(block="ceiling", ckpt=ckpt, arm="probe", layer=L, **summarize(pid[sub], y[sub], s[sub])))
        k = sub & ~np.isnan(rr)
        sr = np.full(len(y), np.nan)
        sr[k] = oof(X[k], rr[k], pid[k], soft, proba=False)
        rows.append(dict(block="ceiling", ckpt=ckpt, arm="probe_fit_to_r_hat", layer=L, **summarize(pid[k], y[k], sr[k])))
    return rows


def main():
    os.makedirs(a.out, exist_ok=True)
    meta = pd.read_parquet(os.path.join(a.data, "meta_full.parquet"))
    acts = np.load(os.path.join(a.data, "acts_full.npy"), mmap_mode="r")
    assert len(acts) == len(meta) and (meta.row.values == np.arange(len(meta))).all()
    assert acts.shape[1] == len(LAYERS), (acts.shape, LAYERS)
    traces = {json.loads(l)["trace_id"]: json.loads(l) for l in open(a.traces)}
    got = meta.groupby("trace_id").is_correct.first()
    want = pd.Series({t: bool(v["is_correct"]) and not v["trunc"] for t, v in traces.items()}).reindex(got.index)
    assert (got == want).all(), f"{int((got != want).sum())} labels disagree with traces_B"
    print("BOS by block:", meta.groupby(meta.line >= a.pilot_lines).n_bos.agg(["min", "max"]).to_dict(), flush=True)
    rec = load_ceiling(a.rollouts)
    ceil_traces = {t for t, _ in rec}
    line = meta.groupby("trace_id").line.first()
    ceil_main = {t for t in ceil_traces if line[t] >= a.pilot_lines}
    print(f"ceiling rollouts: {len(ceil_traces)} traces, {len(ceil_main)} from the main run", flush=True)

    ckpts = sorted((c for c in meta.ckpt.unique() if c != "end"), key=int) + ["end"]
    out, counts = [], []
    for ckpt in ckpts:
        d = meta[(meta.ckpt == ckpt) & ~meta.trunc & (meta.n_gen < 32768)]
        g = d.groupby("problem_id").is_correct.agg(["min", "max"])
        d = d[d.problem_id.isin(g.index[g["min"] != g["max"]])].reset_index(drop=True)
        y, pid = d.is_correct.values.astype(int), d.problem_id.values
        npb = len(np.unique(pid))
        counts.append({"ckpt": ckpt, "traces": len(d), "problems": npb, "wrong": int((y == 0).sum())})
        if npb < max(a.min_problems, K):
            print(f"{ckpt}: {npb} mixed problems, skipped", flush=True)
            continue
        sub, rr = None, np.full(len(d), np.nan)
        if ckpt in CEIL.values():
            sub = d.trace_id.isin(ceil_main).values
            rr = np.array([rec[(t, ckpt)].mean() if (t, ckpt) in rec else np.nan for t in d.trace_id])
            gone = sub & np.isnan(rr)
            if gone.any():
                # a prefix with no rollout at this cut (ceiling.py had 447 of 448 at 4096): every arm of
                # the ceiling block drops the same traces, so the comparison stays on identical traces
                print(f"{ckpt}: {int(gone.sum())} ceiling trace(s) without rollouts here, left out: "
                      f"{list(d.trace_id[gone])}", flush=True)
                sub = sub & ~gone
        print(f"{ckpt}: {len(d)} traces / {npb} problems" + (f"; ceiling subset {int(sub.sum())} traces / "
              f"{len(np.unique(pid[sub]))} problems" if sub is not None else ""), flush=True)
        A = acts[d.row.values]  # this checkpoint's rows, all layers, in memory
        cells = Parallel(n_jobs=a.jobs)(
            delayed(layer_cells)(LAYERS.index(L), L, np.asarray(A[:, LAYERS.index(L), :], np.float32), y, pid, ckpt, sub, rr)
            for L in RUN_LAYERS)
        out += [r for c in cells for r in c]

        T = d.prefix.fillna("").values.astype(object)
        s_text = oof(T, y, pid, tfidf)
        para = np.log1p([paragraphs(x) for x in T])
        feats = np.column_stack([para, np.log1p(d.pos.values)]) if ckpt == "end" else para[:, None]
        s_len = oof(feats, y, pid, lenlr)  # what an online monitor sees: paragraphs (and tokens, at end)
        s_oracle = -d.n_gen.values.astype(float)  # final length: not observable before the end
        base = [("text_tfidf", s_text), ("len_observed", s_len), ("len_final_oracle", s_oracle)]
        out += [dict(block="grid", ckpt=ckpt, arm=n, layer=None, **summarize(pid, y, s)) for n, s in base]
        if sub is not None:
            out += [dict(block="ceiling", ckpt=ckpt, arm=n, layer=None, **summarize(pid[sub], y[sub], s[sub]))
                    for n, s in base]
            out.append(dict(block="ceiling", ckpt=ckpt, arm="r_hat_all8", layer=None, **summarize(pid[sub], y[sub], rr[sub])))
            keys = list(d.trace_id[sub])
            for k in (1, 2, 4):
                rng = np.random.default_rng(SEED + k)
                vals = [summarize(pid[sub], y[sub], np.array([rng.choice(rec[(t, ckpt)], k, replace=False).mean()
                                                              for t in keys]))["auroc"] for _ in range(NDRAW)]
                out.append(dict(block="ceiling", ckpt=ckpt, arm=f"r_hat_k{k}", layer=None, auroc=float(np.mean(vals)),
                                lo=np.nan, hi=np.nan, pairs=np.nan, n_problems=np.nan))
        pd.DataFrame(out).to_csv(os.path.join(a.out, "layer_sweep_long.csv"), index=False)  # partial, per ckpt

    long = pd.DataFrame(out)
    long.to_csv(os.path.join(a.out, "layer_sweep_long.csv"), index=False)
    pd.DataFrame(counts).to_csv(os.path.join(a.out, "layer_sweep_counts.csv"), index=False)
    cell = lambda r: "" if pd.isna(r.auroc) else (f"{r.auroc:.3f}" if pd.isna(r.lo) else
                                                  f"{r.auroc:.3f} [{r.lo:.2f}, {r.hi:.2f}]")
    long["cell"] = long.apply(cell, axis=1)
    long["row"] = [arm if pd.isna(L) else f"{arm} L{int(L)}" for arm, L in zip(long.arm, long.layer)]
    for blk, name in (("grid", "layer_sweep.csv"), ("ceiling", "layer_sweep_ceiling.csv")):
        g = long[long.block == blk]
        if not len(g):
            continue
        piv = g.pivot(index="row", columns="ckpt", values="cell").reindex(
            index=list(dict.fromkeys(g.row)), columns=[c for c in ckpts if c in set(g.ckpt)])
        ref = g[g.arm == ("probe" if blk == "grid" else "r_hat_all8")].groupby("ckpt")[["pairs", "n_problems"]].first()
        piv.loc["(pairs / problems)"] = [f"{int(ref.at[c, 'pairs'])} / {int(ref.at[c, 'n_problems'])}"
                                         if c in ref.index else "" for c in piv.columns]
        piv.to_csv(os.path.join(a.out, name))
        with pd.option_context("display.width", 250, "display.max_rows", 500, "display.max_columns", 30):
            print(f"\n== {name} ==\n{piv}", flush=True)
    print("\ncheckpoint populations:\n" + pd.DataFrame(counts).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
