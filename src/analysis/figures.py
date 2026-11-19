# paper figures from the CSVs the analysis scripts write (layout in README):
#   F1  D by cut (explore / holdout) and capture at 4,096 on the holdout half   determination.py, forward_capture.py
#   F2  re-run AUROC against extra tokens, holdout                              cheap_rollouts.py
#   F3  accuracy against tokens per attempt, restart policies, holdout         allocation.py
#   F4  D and capture at 4,096, three models, cohort rule                       replicate.py
#   P1  D by position of the cut                                                position.py
#   python src/analysis/figures.py --results results --out figures
import argparse, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.patches import Patch

ap = argparse.ArgumentParser()
ap.add_argument("--results", default="results")
ap.add_argument("--out", default="figures")
a = ap.parse_args()
R, OUT = a.results, a.out
DS = f"{R}/deepseek"
os.makedirs(OUT, exist_ok=True)

C = ["#2a78d6", "#eb6834", "#1baf7a", "#e34948", "#6250d6"]
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e6e5e0"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "legend.frameon": False, "savefig.dpi": 220, "font.family": "DejaVu Sans"})


def save(fig, name):
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}/{name}.{ext}", bbox_inches="tight")
    plt.close(fig)


FREE = [("length", "capture:len"), ("restart count", "capture:restarts"), ("answer written", "capture:committed"),
        ("probe on outcome", "capture:probe_y"), ("probe on r̂", "capture:probe_r"),
        ("probe on remaining length", "capture:probe_len")]
ORACLE = ("true remaining length (oracle)", "capture:oracle_remaining")
RERUNS = [("2 re-runs: finished within 4,096", "2 runs, finished within 4096"),
          ("2 re-runs: correct within 4,096", "2 runs, correct within 4096"),
          ("2 re-runs: correct, uncapped", "2 runs, correct within 32768")]

# ---- F1
x1 = {h: pd.read_csv(f"{DS}/x1_{h}.csv") for h in ("explore", "holdout")}
fw = pd.read_csv(f"{DS}/forward_holdout.csv").query("cut == 4096").set_index("monitor")
fig, (ax, bx) = plt.subplots(1, 2, figsize=(10.2, 3.3), gridspec_kw={"width_ratios": [1, 1.45]})
cuts = [512, 2048, 4096]
for j, h in enumerate(("explore", "holdout")):
    d = x1[h][x1[h].what == "determined"].set_index("cut").loc[cuts]
    xs = np.log2(cuts) + (-0.05, 0.05)[j]
    ax.errorbar(xs, d.est, yerr=[d.est - d.lo, d.hi - d.est], fmt="o-", color=C[j], lw=2, ms=6,
                label=f"{h} ({int(d.problems.iloc[0])} problems)")
ax.set_xticks(np.log2(cuts), [f"{c:,}" for c in cuts])
ax.set_ylim(0, 0.6)
ax.set_xlabel("cut (tokens of the trace kept)")
ax.set_ylabel("D: share of outcome variance\nexplained by the prefix")
ax.set_title("A  Outcome determination by cut", loc="left", fontsize=9.5, color=INK)
ax.legend(loc="upper left")
h4 = x1["holdout"].query("cut == 4096").set_index("what")
rows = ([(n, h4.loc[w, ["est", "lo", "hi"]].values, C[0]) for n, w in FREE]
        + [(ORACLE[0], h4.loc[ORACLE[1], ["est", "lo", "hi"]].values, MUTED)]
        + [(n, fw.loc[m, ["capture", "lo", "hi"]].values, C[1]) for n, m in RERUNS])
ys = np.arange(len(rows))[::-1]
for y, (name, (est, lo, hi), col) in zip(ys, rows):
    bx.barh(y, est, height=0.6, color=col)
    bx.plot([lo, hi], [y, y], color=INK, lw=0.9)
    bx.text(hi + 0.015, y, f"{est:.3f}", va="center", fontsize=8, color=INK)
bx.set_yticks(ys, [r[0] for r in rows])
bx.axvline(0, color=MUTED, lw=0.8)
bx.set_xlim(-0.05, 1.0)
bx.set_xlabel("capture: share of D explained (holdout, cut 4,096, 95% CI)")
bx.set_title("B  Capture by free readouts and re-runs", loc="left", fontsize=9.5, color=INK)
bx.legend(handles=[Patch(color=C[0], label="free readouts"), Patch(color=MUTED, label="oracle"),
                   Patch(color=C[1], label="re-runs (extra tokens)")], loc="upper right", fontsize=8)
save(fig, "F1_determination_capture")

# ---- F2
x2 = pd.read_csv(f"{DS}/x2_holdout.csv").query("cut == 4096 and score == 'correct'")
fig, ax = plt.subplots(figsize=(6.6, 3.8))
for (L, lab), col in zip(((2048, "each re-run capped at 2,048"), (4096, "capped at 4,096"), (32768, "uncapped")), C):
    g = x2[x2.L == L].sort_values("K")
    ax.fill_between(g.tokens_per_trace, g.lo, g.hi, color=col, alpha=0.12, lw=0)
    ax.plot(g.tokens_per_trace, g.auroc, "o-", color=col, lw=2.2, ms=6, mec="white", label=lab)
    if L == 4096:
        for r in g.itertuples():
            ax.annotate(f"K={r.K}", (r.tokens_per_trace, r.auroc), xytext=(8, -14), textcoords="offset points",
                        fontsize=8, color=MUTED)
src_tokens = x2.mean_source_tokens.iloc[0]
ax.axvline(src_tokens, color=MUTED, ls=":", lw=1)
ax.text(src_tokens * 1.03, 0.52, f"one source attempt\n({src_tokens / 1000:.1f}k tokens)", fontsize=8, color=MUTED)
ax.set_xscale("log", base=2)
ax.set_xticks([2048, 4096, 8192, 16384, 32768, 65536], ["2k", "4k", "8k", "16k", "32k", "64k"])
ax.set_ylim(0.45, 0.92)
ax.set_xlabel("extra tokens per trace (log scale)")
ax.set_ylabel("within-problem AUROC")
ax.set_title("K re-runs from the 4,096 cut (holdout)", loc="left", fontsize=9.5, color=INK)
ax.legend(loc="upper left")
save(fig, "F2_rerun_cost_curve")

# ---- F3
x4 = pd.read_csv(f"{DS}/x4_holdout.csv").set_index("policy")
env = x4.loc[[f"maj@{k}" for k in range(1, 6)] + [f"short-1@{k}" for k in range(2, 7)], ["tokens", "acc"]]
pts = env.sort_values("tokens").values
hull = []
for p in pts:  # upper concave envelope: mixtures of two baselines lie on the segment between them
    while len(hull) >= 2 and ((hull[-1] - hull[-2])[0] * (p - hull[-2])[1]
                              - (hull[-1] - hull[-2])[1] * (p - hull[-2])[0]) >= 0:
        hull.pop()
    hull.append(p)
hull = np.array(hull)
hull = hull[: hull[:, 1].argmax() + 1]  # past the most accurate baseline, spending more buys nothing
fig, ax = plt.subplots(figsize=(6.8, 4.0))
ax.plot(hull[:, 0], hull[:, 1], color=MUTED, lw=2, label="best baseline without a monitor (envelope)")
sets = [([f"maj@{k}" for k in range(1, 6)], "o", C[0], "majority vote, 1-5 attempts", False),
        ([f"short-1@{k}" for k in range(2, 7)], "s", C[2], "first of k to finish, k = 2-6", False),
        ([f"restart(T8192,R{r})" for r in (1, 2, 5)], "D", C[1], "timeout at 8,192, at most 1 / 2 / 5 restarts", True),
        (["restart(T4096,R1)"], "v", C[3], "restart every trace at 4,096", False),
        (["probe(q0.25)", "probe(q0.5)"], "^", C[4], "probe picks 25% / 50% to restart", False),
        (["random(q0.25)", "random(q0.5)"], "x", INK, "random 25% / 50% restarted", False)]
for names, m, col, lab, line in sets:
    g = x4.loc[names]
    ax.plot(g.tokens, g.acc, m + ("-" if line else ""), color=col, ms=7 if m != "x" else 8, mew=2 if m == "x" else 1,
            mec="white" if m != "x" else col, lw=2, label=lab)
ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:.0f}k"))
ax.set_xlim(6000, 21000)
ax.set_xlabel("generated tokens per answer")
ax.set_ylabel("final-answer accuracy")
ax.set_title("Accuracy at equal compute (holdout, source traces)", loc="left", fontsize=9.5, color=INK)
ax.legend(loc="lower right", fontsize=8)
save(fig, "F3_equal_compute")

# ---- F4
M = [("deepseek", "DeepSeek-R1-Distill-Qwen-1.5B"), ("qwen3", "Qwen3-1.7B"), ("ministral", "Ministral-3-3B-Reasoning")]
x = {m: pd.read_csv(f"{R}/models/x1_{m}_pairs.csv").query("cut == 4096").set_index("what") for m, _ in M}
f4 = {m: pd.read_csv(f"{R}/models/forward_{m}_pairs.csv").query("cut == 4096").set_index("monitor") for m, _ in M}
rows = ([("D (share of outcome variance explained)", "determined", None)] + [(n, w, None) for n, w in FREE]
        + [(ORACLE[0], ORACLE[1], None)] + [(n, None, f) for n, f in RERUNS[1:]])
fig, ax = plt.subplots(figsize=(7.2, 5.4))
y = np.arange(len(rows))[::-1]
for j, (m, lab) in enumerate(M):
    n = int(x[m].loc["determined", "problems"])
    for yi, (_, w, f) in zip(y, rows):
        est, lo, hi = x[m].loc[w, ["est", "lo", "hi"]] if w else f4[m].loc[f, ["capture", "lo", "hi"]]
        yy = yi + (0.27, 0.0, -0.27)[j]
        ax.barh(yy, est, height=0.25, color=C[j], label=f"{lab} ({n} problems)" if yi == y[0] else None)
        ax.plot([max(lo, -0.05), hi], [yy, yy], color=INK, lw=0.9)
ax.set_yticks(y, [r[0] for r in rows])
ax.axhline(y[0] - 0.5, color=MUTED, lw=0.8)
ax.axvline(0, color=MUTED, lw=0.8)
ax.set_xlim(-0.05, 0.8)
ax.set_xlabel("D (top row) or capture: share of D explained (cut 4,096, 95% CI)")
ax.legend(loc="center right", fontsize=8)
save(fig, "F4_models")

# ---- P1
bins = pd.read_csv(f"{R}/position_bins.csv").query("position_def == 'eventual' and traces >= 20")
fig, ax = plt.subplots(figsize=(6.4, 3.8))
for (_, lab), col in zip(M, C):
    g = bins[bins.model == lab].sort_values("mean_position")
    ax.errorbar(g.mean_position, g.D, yerr=[g.D - g.lo, g.hi - g.D], fmt="o-", color=col, capsize=2, lw=1.6, ms=5,
                label=lab)
ax.axhline(0, color=GRID, lw=1)
ax.set_xlabel("fraction of the eventual trace at the cut (uses the trace's future)")
ax.set_ylabel("D (share of outcome variance explained)")
ax.set_title("D by position of the cut (bins with at least 20 traces, 95% CI)", loc="left", fontsize=9)
ax.legend(fontsize=7.5)
save(fig, "P1_d_vs_position")
print("->", OUT)
