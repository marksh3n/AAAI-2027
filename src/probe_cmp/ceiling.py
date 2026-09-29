# recoverability next to the probes, on identical traces. r(t) = P(final answer correct | prefix)
# under the model's own sampling; the true r(t) is the theoretical ceiling for any prefix-only
# predictor. r_hat, the share of K sampled continuations that end correct, is only a noisy
# estimate of it: with K=8, ties and binomial noise pull its AUROC toward 0.5.
# also: answer agreement (continuations reaching the source trace's own final answer), whether each
# cut already holds the answer (from meta's stored prefix text), and a raw restart-count baseline.
# every AUROC uses probe_cmp's own stats (pair-weighted, problem-clustered bootstrap, draws shared
# by all scores at a checkpoint, so every r_hat - probe delta is paired).
#   python src/probe_cmp/ceiling.py --rollouts <rollouts.jsonl> --preds <probe_cmp out>/preds \
#       --meta data/meta_B.parquet --out <dir>
import argparse, json, os, re
import numpy as np
import pandas as pd

import stats as S

MAX_NEW = 32768
CKPTS = {"p512": "512", "p2048": "2048", "p4096": "4096"}  # rollout checkpoint -> probe_cmp checkpoint
SHORT = {"lr/len": "len", "lr/tfidf": "tfidf", "lr/svd50+len": "svd50+len"}
NO_LAYER = -999
ARMS = [("lr/raw", 20), ("lr/raw", 16), ("lr/svd50", 20), ("hgb/svd50", 20), ("svd50+len", 20),
        ("len", NO_LAYER), ("tfidf", NO_LAYER)]
N_BOOT = 2000


def load_rollouts(path):
    # one row per (prefix, round) holding its samples; rounds pool, a rewritten sample keeps the last
    last, torn = {}, 0
    for line in open(path):
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            torn += 1  # a half-written last line from an interrupted run
            continue
        for s in r["samples"]:
            last[(r["prefix_id"], r["round"], s["sample_idx"])] = (r, s)
    rows = {}
    for (pid, _, _), (r, s) in last.items():
        d = rows.setdefault(pid, {"prefix_id": pid, "problem_id": r["problem_id"], "trace_id": r["trace_id"],
                                  "checkpoint": r["checkpoint"], "n": 0, "k": 0, "n_trunc": 0,
                                  "n_from_prefix": 0, "answers": []})
        d["n"] += 1
        d["k"] += bool(s["correct"])
        d["n_trunc"] += bool(s["truncated"])
        d["n_from_prefix"] += bool(s.get("answer_from_prefix"))
        d["answers"].append(s.get("answer"))
    df = pd.DataFrame(list(rows.values()))
    df["r"] = df.k / df.n
    # truncated continuations are never correct; this drops them from the denominator instead
    df["r_finished"] = (df.k / (df.n - df.n_trunc).replace(0, np.nan)).fillna(df.r)
    return df, torn


def load_traces(path):
    t = pd.DataFrame(json.loads(line) for line in open(path)).set_index("trace_id", drop=False)
    t["y"] = (~t.is_correct.astype(bool)).astype(int)  # 1 = wrong, as in probe_cmp
    return t


def norm(a):
    a = re.sub(r"\\(left|right|!|,|;|:)|\s", "", str(a)).replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    return a.rstrip(".").strip("$")


_SAME = {}


def same(a, b):
    # answer equivalence as the rollouts were scored (math_verify), string fallback without it
    if a is None or b is None:
        return False
    if (a, b) not in _SAME:
        try:
            from math_verify import parse, verify
            _SAME[(a, b)] = norm(a) == norm(b) or bool(verify(parse("$" + b + "$"), parse("$" + a + "$")))
        except Exception:
            _SAME[(a, b)] = norm(a) == norm(b)
    return _SAME[(a, b)]


def answer_flags(prefix, answer):
    # committed: the cut already holds the end of thinking or a boxed answer. written: the final
    # answer's value already appears in the prefix as a standalone token (over-flags short numbers)
    committed = "</think>" in prefix or "\\boxed" in prefix
    written = False
    if answer is not None and norm(answer):
        # keep the prefix's spaces (they are the word boundaries); a trailing period is punctuation,
        # but "5" inside 15, 7.5 or 5.2 is not the answer
        text = re.sub(r"\\(left|right)", "", prefix).replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
        a = re.escape(norm(answer))
        written = bool(re.search(rf"(?<![\w.]){a}(?!\w|\.\d)", text))
    return committed, committed or written


def load_meta(path):
    m = pd.read_parquet(path, columns=["trace_id", "pos", "prefix", "restarts"])
    m["pos"] = m.pos.astype(int)
    return m.set_index(["trace_id", "pos"])


def drift_ids(traces, ids):
    # traces whose retokenized length disagrees with the sampled count (vLLM counts the EOS)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    return {i for i in ids
            if traces.at[i, "n_gen"] != len(tok(traces.at[i, "text"], add_special_tokens=False)["input_ids"]) + 1}


def load_preds(path):
    p = pd.read_parquet(path)
    p["arm"] = [SHORT.get(f"{m}/{f}", f"{m}/{f}") for m, f in zip(p.model, p.feature)]
    # seed-averaged per trace, as report.py does
    return p.groupby(["layer", "arm", "trace_id"], as_index=False).agg(p=("p", "mean"), y=("y", "first"))


def compare(sub, preds, n_boot=N_BOOT):
    # sub: rollout rows of one checkpoint with y; preds: seed-averaged probe predictions there
    sub = sub.sort_values(["problem_id", "trace_id"]).reset_index(drop=True)
    ids, y, g = sub.trace_id.values, sub.y.values, sub.problem_id.values
    pairs = S.Pairs(y, g)
    draws = S.problem_draws(len(pairs.pids), n_boot, seed=0)
    scores = {"r_hat": 1 - sub.r.values, "r_hat finished": 1 - sub.r_finished.values}
    if "restarts" in sub:
        scores["restart density (raw)"] = sub.restarts.values.astype(float)  # more restarts -> P(wrong)
    for arm, layer in ARMS:
        a = preds[(preds.arm == arm) & (preds.layer == layer)].set_index("trace_id")
        if not len(a):
            continue
        missing = [i for i in ids if i not in a.index]
        if missing:
            raise SystemExit(f"{arm} L{layer}: {len(missing)} rollout traces have no probe prediction "
                             f"(e.g. {missing[:3]}); the ceiling and the probe must share traces")
        if (a.loc[ids, "y"].values != y).any():
            raise SystemExit(f"{arm} L{layer}: probe labels disagree with the traces' outcomes")
        scores[f"{arm} L{layer}" if layer != NO_LAYER else arm] = a.loc[ids, "p"].values
    rows, boots = [], {}
    for name, s in scores.items():
        w = pairs.auroc(s)
        bw, _ = S.boot_within(pairs, s, draws)
        boots[name] = bw
        rows.append({"score": name, "auroc": w["weighted"], "ci_low": S.ci(bw)[0], "ci_high": S.ci(bw)[1],
                     "n_traces": len(sub), "n_problems": w["n_problems"], "n_pairs": w["n_pairs"]})
    out = pd.DataFrame(rows)
    ref = boots["r_hat"]
    d = [(r.score, r.auroc) for r in out.itertuples()]
    base = out.set_index("score").auroc["r_hat"]
    out["delta_r_hat_minus"] = [base - v for _, v in d]
    out["delta_ci_low"] = [S.ci(ref - boots[n])[0] for n, _ in d]
    out["delta_ci_high"] = [S.ci(ref - boots[n])[1] for n, _ in d]
    out["delta_p"] = [S.boot_p(ref - boots[n]) if n != "r_hat" else np.nan for n, _ in d]
    # tied pairs under r_hat: how much K=8 discreteness caps the ceiling
    dd = (1 - sub.r.values)[pairs.wrong] - (1 - sub.r.values)[pairs.correct]
    return out, float((dd == 0).mean()) if len(dd) else np.nan


def describe(r, traces):
    # mean r by the source trace's outcome, and how often a prefix is already decided
    rows = []
    for c, g in r.groupby("checkpoint"):
        lab = np.where(g.trace_id.isna(), "t0", np.where(g.y == 1, "wrong", "correct"))
        for name, h in g.groupby(lab):
            row = {"checkpoint": c, "source trace": name, "prefixes": len(h), "mean r": h.r.mean(),
                   "r = 0": (h.r == 0).mean(), "r = 1": (h.r == 1).mean(),
                   "mean agreement": h.agree.mean() if "agree" in h else np.nan,
                   "answer boxed in prefix": h.n_from_prefix.sum() / h.n.sum(),
                   "truncated share": h.n_trunc.sum() / h.n.sum()}
            if "committed" in h:
                row["cuts committed"] = h.committed.mean()
                row["cuts with answer written"] = h.written.mean()
            rows.append(row)
    return pd.DataFrame(rows)


def curves(roll, out, n_boot=1000):
    # mean r_hat and answer agreement per checkpoint and source outcome, CIs over problems.
    # t0 has no source trace: it is drawn at the start of both lines.
    order = ["t0"] + list(CKPTS)
    rows = []
    rng = np.random.default_rng(0)
    for src in ("correct", "wrong"):
        for c in order:
            g = roll[roll.checkpoint == c] if c == "t0" else roll[(roll.checkpoint == c) & (roll.y == (src == "wrong"))]
            for col in ("r", "agree"):
                if c == "t0" and col == "agree":
                    continue
                pm = g.groupby("problem_id")[col].mean().dropna().values
                if not len(pm):
                    continue
                b = rng.choice(pm, (n_boot, len(pm))).mean(1)
                rows.append({"source": src, "checkpoint": c, "measure": col, "mean": pm.mean(),
                             "ci_low": np.percentile(b, 2.5), "ci_high": np.percentile(b, 97.5), "problems": len(pm)})
    cv = pd.DataFrame(rows)
    cv.to_csv(os.path.join(out, "recovery_curves.csv"), index=False)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return cv
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=150, sharey=True)
    color = {"correct": "#0e6b66", "wrong": "#b4541f"}
    for ax, (col, title) in zip(axes, [("r", "recoverability: share of continuations correct (r_hat)"),
                                       ("agree", "answer agreement with the source trace")]):
        for src in ("correct", "wrong"):
            g = cv[(cv.source == src) & (cv.measure == col)].set_index("checkpoint").reindex(order)
            x = np.arange(len(order))
            ax.fill_between(x, g.ci_low, g.ci_high, color=color[src], alpha=0.15, lw=0)
            ax.plot(x, g["mean"], "-o", color=color[src], lw=2, ms=4, label=f"source trace ended {src}")
        ax.set_xticks(np.arange(len(order)), ["start"] + [CKPTS[c] for c in CKPTS])
        ax.set_xlabel("tokens of the source trace kept")
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "recovery_curves.png"), bbox_inches="tight")
    plt.close(fig)
    return cv


def run(roll, traces, preds_dir, out, drift=None, meta=None, n_boot=N_BOOT, log=print):
    os.makedirs(out, exist_ok=True)
    roll = roll.copy()
    has = roll.trace_id.notna()
    roll["y"] = np.where(has, traces.reindex(roll.trace_id.fillna("")).y.values, np.nan)
    # answer agreement: continuations whose final answer is the source trace's own final answer
    src_ans = traces.reindex(roll.trace_id.fillna("")).answer.values
    roll["agree"] = [np.mean([same(a, s) for a in ans]) if h else np.nan
                     for ans, s, h in zip(roll.answers, src_ans, has)]
    if meta is not None:
        pos = roll.checkpoint.map(lambda c: int(CKPTS[c]) if c in CKPTS else -1)
        keys = list(zip(roll.trace_id.fillna(""), pos))
        found = [k in meta.index for k in keys]
        miss = [k for k, f, h in zip(keys, found, has) if h and not f]
        if miss:
            raise SystemExit(f"{len(miss)} cuts have no meta row (e.g. {miss[:3]}); pass the meta the probes used")
        flags = [answer_flags(meta.at[k, "prefix"], s) if h else (np.nan, np.nan)
                 for k, s, h in zip(keys, src_ans, has)]
        roll["committed"] = [f[0] for f in flags]
        roll["written"] = [f[1] for f in flags]
        roll["restarts"] = [meta.at[k, "restarts"] if h else np.nan for k, h in zip(keys, has)]
    tr = roll[has].copy()

    # sanity: t0 restarts should track each problem's tag-B pass rate
    t0 = roll[roll.checkpoint == "t0"].set_index("problem_id")
    by_p = traces.groupby("problem_id").is_correct.mean()
    if len(t0):
        a, b = t0.r.values, by_p.loc[t0.index].values.astype(float)
        c = np.corrcoef(a, b)[0, 1] if a.std() > 0 and b.std() > 0 else float("nan")
        log(f"t0 sanity: r(t0) vs the problem's tag-B pass rate over {len(t0)} problems, r = {c:.2f} "
            f"(mean {a.mean():.3f} vs {b.mean():.3f})")
    desc = describe(roll, traces)
    desc.to_csv(os.path.join(out, "r_by_source.csv"), index=False)
    log("\nr_hat and answer agreement by checkpoint and source trace outcome\n"
        + desc.round(3).to_string(index=False))
    curves(roll, out)

    variants = [("all traces", tr)]
    if drift is not None:
        variants.append((f"without {len(drift & set(tr.trace_id))} retokenization-drift traces",
                         tr[~tr.trace_id.isin(drift)]))
    if meta is not None:
        # only cuts where the trace has not yet given its answer; the probe is scored on the same cuts
        variants.append(("before the answer is committed (no </think>, no \\boxed in the cut)",
                         tr[~tr.committed.astype(bool)]))
        variants.append(("before the answer value is written anywhere in the cut",
                         tr[~tr.written.astype(bool)]))
    results = []
    for vname, v in variants:
        for rc, pc in CKPTS.items():
            sub = v[v.checkpoint == rc]
            if not len(sub):
                continue
            path = os.path.join(preds_dir, f"{pc}.parquet")
            res, ties = compare(sub, load_preds(path), n_boot)
            res.insert(0, "checkpoint", pc)
            res.insert(0, "variant", vname)
            res["tied_pairs_under_r_hat"] = ties
            results.append(res)
    allr = pd.concat(results, ignore_index=True)
    allr.to_csv(os.path.join(out, "ceiling_vs_probe.csv"), index=False)

    f = lambda v, lo, hi, sgn=False: (f"{v:+.3f} [{lo:+.3f}, {hi:+.3f}]" if sgn else f"{v:.3f} [{lo:.3f}, {hi:.3f}]")
    lines = ["# Recoverability (r_hat) vs probes, within problem, identical traces\n",
             "- Orientation: P(wrong); AUROC over each problem's correct x wrong pairs, pair-weighted. "
             "r_hat score = 1 - r_hat. CIs: problem-clustered bootstrap, draws shared by every row of a "
             "checkpoint, so deltas are paired.",
             "- The true recovery probability r(t) is the theoretical ceiling. r_hat is its estimate from K "
             "sampled continuations; ties and binomial noise pull its AUROC toward 0.5, so it understates "
             "the ceiling. `r_hat finished` drops truncated continuations from the denominator.",
             "- `restart density (raw)` is the unfitted rate of 'wait / hold on / let me re- / actually' per "
             "1000 tokens of the cut (taxonomy.RESTART), a self-correction proxy; higher = P(wrong).",
             "- The 'before the answer' variants keep only cuts that do not yet hold the answer; each "
             "checkpoint's pairs are rebuilt from those cuts, and the probes are scored on the same cuts.",
             "- t0 is omitted from the table: every trace of a problem shares its t0, so its within-problem "
             "AUROC is 0.5 by construction.\n"]
    for vname, g in allr.groupby("variant", sort=False):
        lines.append(f"## {vname}\n")
        lines.append("| checkpoint | score | AUROC [95% CI] | r_hat − score [95% CI] | p | traces / problems / pairs |")
        lines.append("|---|---|---|---|---|---|")
        for r in g.itertuples():
            delta = "–" if r.score == "r_hat" else f(r.delta_r_hat_minus, r.delta_ci_low, r.delta_ci_high, True)
            p = "–" if np.isnan(r.delta_p) else f"{r.delta_p:.3f}"
            lines.append(f"| {r.checkpoint} | {r.score} | {f(r.auroc, r.ci_low, r.ci_high)} | {delta} | {p} "
                         f"| {r.n_traces} / {r.n_problems} / {r.n_pairs} |")
        tied = g.drop_duplicates("checkpoint")[["checkpoint", "tied_pairs_under_r_hat"]]
        lines.append("\nShare of pairs tied under r_hat: " + ", ".join(
            f"{r.checkpoint} {r.tied_pairs_under_r_hat:.2f}" for r in tied.itertuples()) + "\n")
    open(os.path.join(out, "ceiling_summary.md"), "w").write("\n".join(lines) + "\n")
    log("\n" + "\n".join(lines))
    return allr, desc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", required=True, help="rollouts.jsonl from the Modal ceiling run")
    ap.add_argument("--traces", default="data/traces_B.jsonl")
    ap.add_argument("--preds", required=True, help="preds/ of the <=4096 probe_cmp run")
    ap.add_argument("--out", required=True)
    ap.add_argument("--meta", default=None,
                    help="meta_B.parquet the probes used: prefix text and restarts at each cut")
    ap.add_argument("--no-drift-check", action="store_true", help="skip the tokenizer-based drift variant")
    a = ap.parse_args()
    roll, torn = load_rollouts(a.rollouts)
    print(f"{len(roll)} prefixes, {int(roll.n.sum())} continuations "
          f"(K per prefix: {sorted(roll.n.unique().tolist())}); torn lines skipped: {torn}")
    print("prefixes per checkpoint: " + ", ".join(f"{k} {v}" for k, v in roll.checkpoint.value_counts().items()))
    traces = load_traces(a.traces)
    ids = set(roll.trace_id.dropna())
    drift = None if a.no_drift_check else drift_ids(traces, ids)
    if drift is not None:
        print(f"retokenization drift: {len(drift)} of {len(ids)} traces")
    run(roll, traces, a.preds, a.out, drift, load_meta(a.meta) if a.meta else None)


if __name__ == "__main__":
    main()
