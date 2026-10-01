# every number in the outputs comes from the saved out-of-fold predictions: seed-averaged per
# trace, then a problem-clustered bootstrap whose 2000 draws are shared by every arm, so each
# delta is paired. nothing here refits a model.
#   python src/probe_cmp/report.py [--quick]      # rebuild outputs from data/probe_cmp/preds
import argparse, glob, json, os
import numpy as np
import pandas as pd

import stats as S

NO_LAYER = -999
N_BOOT = 2000
SHORT = {"lr/len": "len", "lr/tfidf": "tfidf", "lr/svd50+len": "svd50+len"}
HIDDEN = ["lr/raw", "lr/svd50", "lr/topk50", "hgb/svd50", "hgb/topk50", "mlp/svd50", "mlp/topk50", "svd50+len"]
SHARED = ["len", "tfidf"]
TABLE = ["lr/raw", "lr/svd50", "hgb/svd50", "mlp/svd50", "len", "tfidf", "svd50+len"]
PRIMARY = ("lr/raw", "len", 20)


def arm_of(model, feature):
    k = f"{model}/{feature}"
    return SHORT.get(k, k)


def res_key(c, l, arm):
    return (c, NO_LAYER if arm in SHARED else l, arm)


def f3(v):
    return "–" if v is None or not np.isfinite(v) else f"{v:.3f}"


def fci(v, lo, hi, signed=False):
    if v is None or not np.isfinite(v):
        return "–"
    f = "{:+.3f}" if signed else "{:.3f}"
    return f"{f.format(v)} [{f.format(lo)}, {f.format(hi)}]"


def load(out, ckpts):
    preds, perm = [], []
    for c in ckpts:
        preds.append(pd.read_parquet(os.path.join(out, "preds", f"{c}.parquet")))
        pp = os.path.join(out, "preds", f"{c}.perm.parquet")
        if os.path.exists(pp):
            perm.append(pd.read_parquet(pp))
    preds = pd.concat(preds, ignore_index=True)
    preds["arm"] = [arm_of(m, f) for m, f in zip(preds.model, preds.feature)]
    return preds, (pd.concat(perm, ignore_index=True) if perm else None)


def find(out, name):
    # delong.json and reextract_check.json sit in the base output dir; a --quick or subset
    # report lives one or two levels below it
    d = out.rstrip("/")
    for _ in range(3):
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
        d = os.path.dirname(d)
    return None


def build(out, store, pop, ckpts, layers, seeds):
    preds, perm = load(out, ckpts)
    y, groups = pop.y.values, pop.problem_id.values
    order = {t: i for i, t in enumerate(pop.trace_id)}
    pairs = S.Pairs(y, groups)
    draws = S.problem_draws(len(pairs.pids), N_BOOT, seed=0)
    n_tr, n_pb, n_pairs = len(pop), len(pairs.pids), int(pairs.pairs.sum())

    def aligned(g):
        v = np.full(len(pop), np.nan)
        v[g.trace_id.map(order).values] = g.p.values
        if np.isnan(v).any():
            raise ValueError("saved predictions do not cover the population")
        return v

    # point estimates and bootstrap draws per (ckpt, layer, arm); layer-free arms under NO_LAYER
    res, boots, rows = {}, {}, []
    for (c, l, arm), g in preds.groupby(["ckpt", "layer", "arm"], sort=False):
        per_seed = {s: aligned(gs) for s, gs in g.groupby("seed")}
        s_avg = np.mean(list(per_seed.values()), axis=0)
        w = pairs.auroc(s_avg)
        bw, bu = S.boot_within(pairs, s_avg, draws)
        bp = S.boot_pooled(y, s_avg, groups, pairs.pids, draws)
        pooled = S.pooled_auroc(y, s_avg)
        seed_w = [pairs.auroc(v)["weighted"] for v in per_seed.values()]
        seed_u = [pairs.auroc(v)["unweighted"] for v in per_seed.values()]
        seed_p = [S.pooled_auroc(y, v) for v in per_seed.values()]
        res[(c, l, arm)] = {"within": w["weighted"], "ci": S.ci(bw), "unweighted": w["unweighted"]}
        boots[(c, l, arm)] = bw
        model, feat = g.model.iat[0], g.feature.iat[0]
        role = "primary" if (arm in PRIMARY[:2] and l in (PRIMARY[2], NO_LAYER)) else "exploratory"
        for metric, val, bs, spread in [("within_weighted", w["weighted"], bw, seed_w),
                                        ("within_unweighted", w["unweighted"], bu, seed_u),
                                        ("pooled", pooled, bp, seed_p)]:
            lo, hi = S.ci(bs)
            rows.append({"layer": "none" if l == NO_LAYER else l, "checkpoint": c, "feature": feat,
                         "model": model, "arm": arm, "metric": metric, "value": val, "ci_low": lo,
                         "ci_high": hi, "n_traces": n_tr, "n_problems": n_pb,
                         "n_discordant_pairs": n_pairs, "seed_min": min(spread), "seed_max": max(spread),
                         "n_seeds": len(spread), "role": role,
                         "chosen_C": ";".join(sorted(set(g.C))) if model == "lr" else ""})
    long = pd.DataFrame(rows)
    long.to_csv(os.path.join(out, "results_long.csv"), index=False)

    # paired deltas, same draws for both arms
    drows = []
    for l in layers:
        for c in ckpts:
            comps = [(a, "len") for a in HIDDEN] + [(a, "tfidf") for a in HIDDEN] + \
                    [(a, "lr/raw") for a in HIDDEN if a != "lr/raw"] + \
                    [(f"{m}/{f}", f"lr/{f}") for m in ("hgb", "mlp") for f in ("svd50", "topk50")]
            for a, b in comps:
                ka, kb = res_key(c, l, a), res_key(c, l, b)
                if ka not in res or kb not in res:
                    continue
                d = res[ka]["within"] - res[kb]["within"]
                bd = boots[ka] - boots[kb]
                lo, hi = S.ci(bd)
                drows.append({"layer": l, "checkpoint": c, "arm_a": a, "arm_b": b, "metric": "within_weighted",
                              "delta": d, "ci_low": lo, "ci_high": hi, "p_boot": S.boot_p(bd),
                              "role": "primary" if (a, b, l) == PRIMARY else "exploratory"})
    deltas = pd.DataFrame(drows)
    prim = deltas.role == "primary"
    deltas["p_holm_primary"] = np.nan
    if prim.any():
        deltas.loc[prim, "p_holm_primary"] = S.holm(deltas.loc[prim, "p_boot"].values)
    deltas.to_csv(os.path.join(out, "paired_deltas.csv"), index=False)

    # permutation null
    prow = []
    if perm is not None:
        for (c, l, arm), g in perm.groupby(["ckpt", "layer", "arm"], sort=False):
            obs = float(g.score[g.rep == -1].iat[0])
            null = g.score[g.rep >= 0].values
            main = res.get((c, l, arm), {}).get("within", np.nan)
            prow.append({"layer": l, "checkpoint": c, "arm": arm, "n_perm": len(null),
                         "null_mean": null.mean(), "null_sd": null.std(ddof=1),
                         "null_p95": np.percentile(null, 95), "observed_seed0_fixed": obs,
                         "p_value": (1 + (null >= obs).sum()) / (1 + len(null)),
                         "main_score_seedavg_innerC": main})
    permdf = pd.DataFrame(prow)
    permdf.to_csv(os.path.join(out, "perm_null.csv"), index=False)

    delong_path = find(out, "delong.json")
    check_path = find(out, "reextract_check.json")
    write_summary(out, store, pop, ckpts, layers, seeds, res, deltas, permdf, n_pairs,
                  delong_path, check_path)
    plot(out, ckpts, res, permdf, layer=PRIMARY[2] if PRIMARY[2] in layers else layers[0])


def perm_p95(permdf, c, l):
    if permdf.empty:
        return np.nan
    g = permdf[(permdf.checkpoint == c) & (permdf.layer == l)]
    return float(g.null_p95.max()) if len(g) else np.nan


def delta_row(deltas, c, l, a, b):
    g = deltas[(deltas.checkpoint == c) & (deltas.layer == l) & (deltas.arm_a == a) & (deltas.arm_b == b)]
    return g.iloc[0] if len(g) else None


def fdelta(r):
    return "–" if r is None else fci(r.delta, r.ci_low, r.ci_high, signed=True)


def reextract_status(path):
    if path is None:
        return "NOT RUN: row-level alignment is unverified (run reextract_check.py on a GPU runtime)"
    r = json.load(open(path))
    return ("passed" if r.get("passed") else "FAILED") + f": {r.get('summary', '')}"


def write_summary(out, store, pop, ckpts, layers, seeds, res, deltas, permdf, n_pairs,
                  delong_path, check_path):
    L = []
    w = L.append
    n_expl = int((deltas.role == "exploratory").sum())
    popj = json.load(open(os.path.join(out, "population.json")))
    w("# Probe comparison across checkpoints (E0)\n")
    if popj.get("exploratory"):
        w("> **Exploratory run.** The checkpoint set or layers were chosen after seeing earlier results. "
          "Labels below such as *primary* and *Holm* describe the pipeline's layout, not a pre-declared "
          "test: read every number here as exploratory.\n")
    w("- **Question:** within a problem, do hidden states separate correct from wrong traces beyond "
      "visible features? Primary metric: within-problem AUROC, problems weighted by their "
      "correct×wrong pairs.")
    w("- **Orientation:** every score is P(wrong); AUROC = P(score of a wrong trace > score of a "
      "correct trace of the same problem), ties ½.")
    w(f"- **Population (fixed across checkpoints):** {len(pop)} traces, {pop.problem_id.nunique()} "
      f"mixed-outcome problems, {int(pop.y.sum())} wrong, {n_pairs} pairs. Truncated traces excluded: "
      f"{int((~store.traces.eligible).sum())}. Checkpoints: {', '.join(ckpts)}.")
    rule = {"pos-1": "absolute count of generated tokens observed, state at trace token pos−1",
            "answer_end": "`end`, the last token of the answer after `</think>`"}
    w("- **Checkpoint sources:** " + "; ".join(
        f"`{name}` ({src['acts']}): {rule.get(src['token_index'], src['token_index'])}"
        for name, (src, _) in store.sources.items()) + ".")
    w(f"- **Layer convention:** layer *l* = HF `output_hidden_states[l+1]` "
      f"({store.convention}); l = −1 is the embedding output, 27 is after the final RMSNorm. "
      f"Layers {', '.join(map(str, sorted(layers)))} therefore mean the residual stream after those decoder "
      f"blocks (0-indexed). "
      f"Every source is registered with this convention; per-layer norm profiles agree across sources, and "
      f"the GPU re-extraction check below tests it row by row.")
    if any(src.get("name_col") for src, _ in store.sources.values()):
        w("- **Relative checkpoints present:** where a checkpoint is a fraction of the trace, tokens observed "
          "= fraction × final length, so `len` there carries total length by construction.")
    w("- **len:** tokens and `\\n\\n` paragraphs observed up to the checkpoint (log1p, LR learns the sign). "
      "At a fixed absolute position every trace has the same token count, so there `len` is the paragraph "
      "count alone; at `end` it is total length.")
    w(f"- **Seeds / CV:** {len(seeds)} fold seed(s) × StratifiedGroupKFold(5) by problem; predictions averaged "
      f"per trace across seeds; C for every LR chosen by grouped 3-fold inner CV. CIs: problem-clustered "
      f"bootstrap, {N_BOOT} draws shared by all arms.")
    w(f"- **Primary comparison (pre-declared):** `lr/raw` vs `len`, layer 20, one per checkpoint "
      f"({len(ckpts)} tests, Holm-adjusted). Everything else is **exploratory**: {n_expl} further paired "
      f"comparisons with no multiplicity correction.")
    w(f"- **Row-level re-extraction check:** {reextract_status(check_path)}\n")

    for l in layers:
        w(f"## Layer {l}: within-problem AUROC [95% CI]\n")
        cols = TABLE + ["perm p95*", "Δ lr/raw − len", "Δ svd50+len − len"]
        w("| checkpoint | " + " | ".join(cols) + " |")
        w("|" + "---|" * (len(cols) + 1))
        for c in ckpts:
            cells = []
            for a in TABLE:
                r = res.get(res_key(c, l, a))
                cells.append(fci(r["within"], *r["ci"]) if r else "–")
            cells.append(f3(perm_p95(permdf, c, l)))
            cells.append(fdelta(delta_row(deltas, c, l, "lr/raw", "len")))
            cells.append(fdelta(delta_row(deltas, c, l, "svd50+len", "len")))
            label = f"**{c}**" if c == "end" else c
            w(f"| {label} | " + " | ".join(cells) + " |")
        w("\n\\* larger of the lr/svd50 and hgb/svd50 null 95th percentiles (layer 20 only).\n")
        if "end" not in ckpts:
            continue
        end = "end"
        w(f"**`{end}`, the checkpoint where length is strongest (layer {l}):**\n")
        w("| lr/raw | svd50+len | len | Δ lr/raw − len | Δ svd50+len − len |")
        w("|---|---|---|---|---|")
        g = lambda a: res.get(res_key(end, l, a))
        w("| " + " | ".join(fci(g(a)["within"], *g(a)["ci"]) if g(a) else "–"
                            for a in ("lr/raw", "svd50+len", "len")) +
          f" | {fdelta(delta_row(deltas, end, l, 'lr/raw', 'len'))} | "
          f"{fdelta(delta_row(deltas, end, l, 'svd50+len', 'len'))} |\n")

    pr = deltas[deltas.role == "primary"]
    if len(pr):
        w("### Primary comparisons (layer 20, lr/raw − len)\n")
        w("| checkpoint | Δ [95% CI] | bootstrap p | Holm p |")
        w("|---|---|---|---|")
        for r in pr.itertuples():
            w(f"| {r.checkpoint} | {fci(r.delta, r.ci_low, r.ci_high, True)} | {r.p_boot:.3f} | {r.p_holm_primary:.3f} |")
        w("")

    if not permdf.empty:
        w("### Within-problem permutation null (layer 20, seed-0 folds, fixed hyperparameters)\n")
        w("| checkpoint | arm | observed | null mean ± sd | null p95 | p |")
        w("|---|---|---|---|---|---|")
        for r in permdf.itertuples():
            w(f"| {r.checkpoint} | {r.arm} | {r.observed_seed0_fixed:.3f} | {r.null_mean:.3f} ± {r.null_sd:.3f} "
              f"| {r.null_p95:.3f} | {r.p_value:.3f} |")
        w("")

    if delong_path:
        dj = json.load(open(delong_path))
        w("## Iteration-1 check: 0.887 vs 0.854 — between-problem (one trace per problem)\n")
        for line in dj.get("plain", []):
            w(f"- {line}")
        w("")

    w("## Plain-language summary\n")
    for line in plain_lines(ckpts, res, deltas, permdf, layers):
        w(f"- {line}")
    w("\n## Secondary: varying population (counts only; later checkpoints are length-selected)\n")
    vc = pd.DataFrame(popj["varying_secondary"])
    w("| " + " | ".join(vc.columns) + " |")
    w("|" + "---|" * len(vc.columns))
    for r in vc.itertuples(index=False):
        w("| " + " | ".join(str(v) for v in r) + " |")
    open(os.path.join(out, "summary.md"), "w").write("\n".join(L) + "\n")


def plain_lines(ckpts, res, deltas, permdf, layers):
    l = 20 if 20 in layers else layers[0]
    get = lambda c, a: res.get(res_key(c, l, a))
    last = ckpts[-1]
    out = []
    r, n = get(last, "lr/raw"), get(last, "len")
    d = delta_row(deltas, last, l, "lr/raw", "len")
    if r and n and d is not None:
        beats = d.p_holm_primary < 0.05
        out.append(f"Primary test at `{last}` (layer {l}): the hidden-state probe scores {fci(r['within'], *r['ci'])} "
                   f"within problem against {fci(n['within'], *n['ci'])} for length, Δ {fdelta(d)}, Holm p = "
                   f"{d.p_holm_primary:.2f}; hidden states "
                   + ("separate correct from wrong attempts beyond what length gives." if beats else
                      "do not separate correct from wrong attempts detectably beyond length."))
    early = [c for c in ckpts if c != last]
    if early:
        vals = [get(c, "lr/raw") for c in early if get(c, "lr/raw")]
        above = sum(v["ci"][0] > 0.5 for v in vals)
        pe = deltas[(deltas.role == "primary") & deltas.checkpoint.isin(early)]
        sig = pe.p_holm_primary < 0.05
        up, down = pe[sig & (pe.delta > 0)], pe[sig & (pe.delta < 0)]
        verdict = (f"hidden states beat length early, at {', '.join(up.checkpoint)}." if len(up)
                   else "no early signal beyond length.")
        if len(down):
            verdict += (f" At {', '.join(down.checkpoint)} the probe is significantly *below* length, "
                        "likely a small-sample CV artifact, not an inverse signal.")
        out.append(f"Before `{last}`, lr/raw ranges {min(v['within'] for v in vals):.3f}–"
                   f"{max(v['within'] for v in vals):.3f} over {', '.join(early)}; {above} of {len(vals)} "
                   f"uncorrected CIs sit above 0.5, and the smallest primary p against length is "
                   f"{pe.p_boot.min():.2f} (Holm {pe.p_holm_primary.min():.2f}): " + verdict)
    d2 = delta_row(deltas, last, l, "svd50+len", "len")
    if d2 is not None:
        others = [f"layer {o}: {fdelta(delta_row(deltas, last, o, 'svd50+len', 'len'))}" for o in layers if o != l]
        if d2.ci_low > 0.01:
            verdict = "hidden states add information over length (exploratory, uncorrected)."
        elif d2.ci_low > 0:
            verdict = ("borderline: the CI only just clears 0 in one of the exploratory, uncorrected comparisons, "
                       "so at most a small increment over length.")
        else:
            verdict = "no detectable information over length."
        out.append(f"Incremental test at `{last}` (exploratory): adding hidden states (SVD-50) to length changes AUROC "
                   f"by {fdelta(d2)}" + (f" ({'; '.join(others)})" if others else "") + f" — {verdict}")
    nl = [(a, get(last, a)) for a in ("hgb/svd50", "hgb/topk50", "mlp/svd50", "mlp/topk50") if get(last, a)]
    if nl:
        a, best = max(nl, key=lambda t: t[1]["within"])
        twin = "lr/" + a.split("/")[1]  # same features, linear model: isolates non-linearity
        dn = delta_row(deltas, last, l, a, twin)
        out.append(f"Best non-linear model at `{last}` is {a} at {best['within']:.3f}; against {twin} on the same "
                   f"features that is {fdelta(dn)}, so " + ("non-linearity helps there." if dn is not None
                                                           and dn.ci_low > 0 else "non-linearity adds nothing detectable."))
    t, dt = get(last, "tfidf"), delta_row(deltas, last, l, "lr/raw", "tfidf")
    if t and dt is not None:
        out.append(f"Text alone (TF-IDF) scores {t['within']:.3f} at `{last}`; lr/raw − tfidf = {fdelta(dt)}.")
    return out


def plot(out, ckpts, res, permdf, layer):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # reference palette, fixed slot per arm in both panels; len and tfidf dashed
    color = {"lr/raw": "#2a78d6", "lr/svd50": "#eb6834", "hgb/svd50": "#1baf7a", "mlp/svd50": "#eda100",
             "svd50+len": "#e87ba4", "len": "#008300", "tfidf": "#4a3aa7"}
    panels = [("key comparisons", ["lr/raw", "svd50+len", "len", "tfidf"]),
              ("model variants (SVD-50)", ["lr/raw", "lr/svd50", "hgb/svd50", "mlp/svd50"])]
    ink, muted, grid, surface = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
    x = np.arange(len(ckpts))
    p95 = np.array([perm_p95(permdf, c, layer) for c in ckpts])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), dpi=150, sharey=True)
    fig.patch.set_facecolor(surface)
    for ax, (title, arms) in zip(axes, panels):
        ax.set_facecolor(surface)
        if np.isfinite(p95).any():
            ax.fill_between(x, 0.3, p95, color="#d8d7d1", lw=0, zorder=0,
                            label="permutation null, 95th pct")
            ax.plot(x, p95, color="#8a8984", lw=1, zorder=1)
        ax.axhline(0.5, color="#8a8984", lw=1, ls=":", zorder=1)
        for arm in arms:
            pts = [res.get(res_key(c, layer, arm)) for c in ckpts]
            if not any(pts):
                continue
            v = np.array([p["within"] if p else np.nan for p in pts])
            lo = np.array([p["ci"][0] if p else np.nan for p in pts])
            hi = np.array([p["ci"][1] if p else np.nan for p in pts])
            col, ls = color[arm], "--" if arm in SHARED else "-"
            ax.fill_between(x, lo, hi, color=col, alpha=0.13, lw=0, zorder=2)
            ax.plot(x, v, ls, color=col, lw=2, marker="o", ms=5, label=arm, zorder=3)
        ax.set_xticks(x, ckpts, rotation=45 if len(ckpts) > 7 else 0)
        ax.set_title(title, loc="left", fontsize=10, color=ink)
        ax.set_xlabel("checkpoint (generated tokens observed; end = last token)", color=muted, fontsize=9)
        ax.set_ylim(0.3, 1.0)
        ax.grid(axis="y", color=grid, lw=0.8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color("#b9b8b2")
        ax.tick_params(colors=muted, labelsize=9)
        ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.17))
    axes[0].set_ylabel("within-problem AUROC, P(wrong)", color=muted)
    fig.suptitle(f"Within-problem AUROC by checkpoint, layer {layer} (fixed population, 95% CI bands)",
                 x=0.01, ha="left", fontsize=11, color=ink)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "within_auroc_by_checkpoint.png"), facecolor=surface, bbox_inches="tight")
    plt.close(fig)


def main():
    import data as D
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--registry", default=D.REGISTRY)
    ap.add_argument("--out", default="data/probe_cmp")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    out = os.path.join(a.out, "quick") if a.quick else a.out
    popj = json.load(open(os.path.join(out, "population.json")))
    store = D.Store(a.data, reg=D.load_registry(a.registry), log=lambda *_: None)
    pop = D.population(store, popj["checkpoints"])
    if pop.trace_id.tolist() != popj["trace_ids"]:
        raise SystemExit("population.json does not match the population rebuilt from the data")
    preds = pd.concat(pd.read_parquet(p, columns=["layer", "seed"])
                      for p in glob.glob(os.path.join(out, "preds", "*.parquet"))
                      if not p.endswith(".perm.parquet"))
    layers = sorted({int(v) for v in preds.layer.unique() if v != NO_LAYER}, key=lambda v: (v != 20, -v))
    seeds = sorted(preds.seed.unique())
    build(out, store, pop, popj["checkpoints"], layers, seeds)


if __name__ == "__main__":
    main()
