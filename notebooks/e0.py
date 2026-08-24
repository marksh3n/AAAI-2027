import marimo

app = marimo.App(width="medium")


@app.cell
def _():
    import json
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    results = pd.read_parquet("data/results.parquet")
    summary = json.load(open("data/summary.json"))
    return json, mo, np, pd, plt, results, summary


@app.cell
def _(mo, summary):
    _main, _shuf = summary["main"], summary["shuffled"]
    _lm, _early = summary["length_matched"], summary["first60"]
    _ci = _main["test_ci"]
    _lv = " / ".join(f"L{k}: {v['wrong']}/{v['n']}" for k, v in summary["by_level"].items())
    mo.md(f"""
    ## E0 summary

    | | |
    |---|---|
    | selected cell | layer {_main['layer']}, bin {_main['bin']} (C={_main['C']}) |
    | test AUROC | **{_main['test_auroc']:.3f}**  [{_ci[0]:.3f}, {_ci[1]:.3f}] |
    | shuffled | {_shuf['test_auroc']:.3f} |
    | selectivity gap | **{summary['selectivity_gap']:.3f}** |
    | length-matched | {_lm['test_auroc']:.3f} (layer {_lm['layer']}, bin {_lm['bin']}) |
    | first 60% | **{_early['test_auroc']:.3f}** (layer {_early['layer']}, bin {_early['bin']}) |
    | test traces | {_main['n_test_traces']} ({_main['n_test_wrong']} wrong) |
    | accuracy | {summary['accuracy']:.1%} |
    | truncation rate | {summary['truncation_rate']:.1%} |
    | self-correction rate | {summary['self_correction_rate']:.1%} |
    | errors by level | {_lv} |

    Go/no-go: test AUROC >= 0.65, selectivity gap > 0.10, surviving length matching.
    """)
    return


@app.cell
def _(np, pd, plt, results):
    def _heatmap(ax, condition, title):
        # best C per cell, so the grid is layer x bin
        sub = results[results.condition == condition]
        grid = sub.pivot_table(index="layer", columns="bin", values="val_auroc", aggfunc="max")
        im = ax.imshow(grid.values, vmin=0.35, vmax=0.85, cmap="RdBu_r", aspect="auto")
        ax.set_xticks(range(grid.shape[1]), [f"{b / 8:.2f}" for b in grid.columns])
        ax.set_yticks(range(grid.shape[0]), grid.index)
        ax.set_xlabel("relative position")
        ax.set_ylabel("layer")
        ax.set_title(title)
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                v = grid.values[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7)
        return im

    _fig, _axes = plt.subplots(1, 3, figsize=(16, 5))
    for _ax, (_cond, _title) in zip(
        _axes,
        [
            ("main", "val AUROC"),
            ("shuffled", "shuffled labels"),
            ("length_matched", "length-matched"),
        ],
    ):
        _im = _heatmap(_ax, _cond, _title)
    _fig.colorbar(_im, ax=_axes, shrink=0.8)
    _fig
    return


@app.cell
def _(plt, results, summary):
    _layer = summary["main"]["layer"]
    _curve = (
        results[(results.condition == "main") & (results.layer == _layer)]
        .groupby("bin")
        .val_auroc.max()
    )
    _shufc = (
        results[(results.condition == "shuffled") & (results.layer == _layer)]
        .groupby("bin")
        .val_auroc.max()
    )

    _fig2, _ax2 = plt.subplots(figsize=(7, 4.5))
    _ax2.plot(_curve.index / 8 + 1 / 16, _curve.values, "o-", label=f"layer {_layer}")
    _ax2.plot(_shufc.index / 8 + 1 / 16, _shufc.values, "o--", color="gray", label="shuffled")
    _ax2.axhline(0.5, color="k", lw=0.8)
    _ax2.axvline(0.6, color="crimson", ls=":", label="60% of trace")
    _ax2.set_xlabel("relative position in trace")
    _ax2.set_ylabel("val AUROC")
    _ax2.set_title(f"error signal vs step position, layer {_layer}")
    _ax2.legend()
    _fig2
    return


if __name__ == "__main__":
    app.run()
