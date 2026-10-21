# oracle-r_hat intervention upper bound, from the existing ceiling rollouts. CPU only, no model.
# a monitor reads each trace at a cut and flags a share q of them; a flagged trace is abandoned and
# replaced. the question is how much accuracy that can buy, and at what token cost, when the monitor
# ranks by the rollout recoverability r_hat itself: the most any prefix-only monitor could do.
#   restart:  replace the trace with a fresh sample from the prompt; succeeds with r(t0) of its problem
#   resample: replace the remainder with a fresh continuation of the same prefix; succeeds with r(prefix)
# cross-fitted: each prefix's samples (t0 included) are split in two halves; one half ranks, the
# other scores the replacement, then the halves swap and the two values are averaged. so no
# replacement is scored on the samples that chose it.
# a prefix-only score cannot gain from resample in expectation: given the prefix, the source's own
# remainder and a resample are draws from the same distribution. resample is the control: whatever it
# gains under a prefix-only score is population selection (mixed problems, traces >= 4096 tokens,
# untruncated), not monitoring. restart under random flagging carries the same selection, so every
# gain is also reported over random flagging at the same q, which is the monitor's value.
# ties in a score are flagged fractionally (the expected value of a random tie-break).
# --length-matched: the control fails on the raw rollouts (2026-10-20: oracle r_hat resample -3.1 pp at
# 512, q=0.5), because the sources were selected on their own future (untruncated, >= 4096 tokens) and
# the rollouts were not. this keeps only continuations whose whole trajectory would have passed the
# same selection: finished, closes </think> (in the continuation or already in the prefix), at least
# MIN_LEN tokens in all and under the source budget MAX_NEW (the rollouts ran max_tokens 32768 past
# the cut). halves are re-dealt over the kept samples; a prefix with < 2 left, or a problem whose t0
# has < 2 left, is dropped and logged.
#   python src/probe_cmp/oracle.py --rollouts <recovery_ceiling> --traces data/traces_B.jsonl \
#       [--preds <probe_cmp out>/preds] [--length-matched] --out <dir>
import argparse, json, os
import numpy as np
import pandas as pd

import stats as S
from ceiling import CKPTS, load_preds, load_traces

RATES = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0)
SHOW = (0.1, 0.2, 0.3, 0.5)
PROBE_ARMS = [("lr/raw", 20), ("hgb/svd50", 20), ("len", -999), ("tfidf", -999)]
N_BOOT = 2000
MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"  # generate.MODEL, not imported: it pulls in torch
MIN_LEN = 4096  # ceiling population: retokenized source length >= the deepest cut
MAX_NEW = 32768  # tag-B budget for the whole trace
THINK_END = "</think>"


def load_samples(path):
    # one row per continuation; rounds pool and a rewritten sample keeps the last, as ceiling.py
    last, torn = {}, 0
    for line in open(path):
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            torn += 1
            continue
        for s in r["samples"]:
            last[(r["prefix_id"], r["round"], s["sample_idx"])] = {
                "prefix_id": r["prefix_id"], "round": r["round"], "sample_idx": s["sample_idx"],
                "problem_id": r["problem_id"], "trace_id": r["trace_id"], "checkpoint": r["checkpoint"],
                "correct": bool(s["correct"]), "n_gen": int(s["n_gen"]), "truncated": bool(s.get("truncated")),
                "think": THINK_END in (s.get("text") or "")}
    df = pd.DataFrame(list(last.values()))
    df["cut"] = df.checkpoint.map(lambda c: int(CKPTS.get(c, 0)))  # t0 = 0
    return deal(df), torn


def deal(df):
    # half 0 / half 1 alternate in sample order, so each half of a K=8 prefix holds 4
    df = df.sort_values(["prefix_id", "round", "sample_idx"]).reset_index(drop=True)
    df["half"] = df.groupby("prefix_id").cumcount() % 2
    return df


def think_tokens(traces, ids, tokenizer):
    # token index of the source's </think>, so a prefix holds it when that index < cut
    out = {}
    for tid in ids:
        text = traces.at[tid, "text"]
        i = text.find(THINK_END)
        out[tid] = len(tokenizer(text[:i], add_special_tokens=False)["input_ids"]) if i >= 0 else np.inf
    return out


def length_match(samples, think_tok, log=print):
    # keep continuations whose trajectory passes the population's own selection; see the header
    # n_gen counts the EOS on a stop, the retokenized source length does not, hence the -1
    total = samples.cut + samples.n_gen
    closed = samples.think | (samples.trace_id.map(think_tok).fillna(np.inf) < samples.cut)
    keep = ~samples.truncated & closed & (total - 1 >= MIN_LEN) & (total < MAX_NEW)
    for c, g in samples.assign(keep=keep).groupby("checkpoint"):
        log(f"  length match {c}: kept {g.keep.mean():.1%} of {len(g)} continuations "
            f"(truncated {g.truncated.mean():.1%}, short {((g.cut + g.n_gen - 1 < MIN_LEN) & ~g.truncated).mean():.1%}, "
            f"over budget {((g.cut + g.n_gen >= MAX_NEW) & ~g.truncated).mean():.1%}); "
            f"accuracy {g.correct.mean():.3f} -> {g[g.keep].correct.mean():.3f}")
    kept = samples[keep]
    n = kept.groupby("prefix_id").size()
    thin = set(samples.prefix_id.unique()) - set(n.index[n >= 2])
    t0 = samples[samples.checkpoint == "t0"].drop_duplicates("prefix_id")
    lost = set(t0.problem_id[t0.prefix_id.isin(thin)])
    if thin:
        log(f"  length match: {len(thin)} prefixes left with < 2 continuations, dropped; "
            f"{len(lost)} problems lose their t0 and are dropped whole: {sorted(lost)}")
    kept = kept[~kept.prefix_id.isin(thin) & ~kept.problem_id.isin(lost)]
    return deal(kept.copy())


def half_stats(samples):
    # per prefix and half: r_hat and mean continuation length
    g = samples.groupby(["prefix_id", "half"]).agg(r=("correct", "mean"), L=("n_gen", "mean"), n=("correct", "size"))
    g = g.unstack("half")
    if g["n"].isna().any().any():
        bad = g.index[g["n"].isna().any(axis=1)][:3].tolist()
        raise SystemExit(f"prefixes with fewer than 2 samples cannot be split in halves (e.g. {bad})")
    return g


def table(samples, traces, ckpt):
    # one row per source trace at this cut, with both halves of its prefix and of its problem's t0
    hs = half_stats(samples)
    meta = samples.drop_duplicates("prefix_id").set_index("prefix_id")
    t0 = {meta.at[p, "problem_id"]: p for p in meta.index[meta.checkpoint == "t0"]}
    rows = []
    for pid in meta.index[meta.checkpoint == ckpt]:
        tid, prob = meta.at[pid, "trace_id"], meta.at[pid, "problem_id"]
        if prob not in t0:
            raise SystemExit(f"problem {prob} has no t0 rollouts; restart cannot be scored")
        row = {"prefix_id": pid, "trace_id": tid, "problem_id": prob, "y": int(traces.at[tid, "y"]),
               "src_tokens": int(traces.at[tid, "n_gen"]), "cut": int(CKPTS[ckpt])}
        for h in (0, 1):
            row[f"rp{h}"], row[f"Lp{h}"] = hs.at[pid, ("r", h)], hs.at[pid, ("L", h)]
            row[f"rt{h}"], row[f"Lt{h}"] = hs.at[t0[prob], ("r", h)], hs.at[t0[prob], ("L", h)]
        rows.append(row)
    if not rows:
        return pd.DataFrame()  # no prefixes at this cut
    return pd.DataFrame(rows).sort_values(["problem_id", "trace_id"]).reset_index(drop=True)


def flag_weights(score, q):
    # share q of rows flagged, highest score first; a tie group straddling the budget is flagged
    # fractionally, which is the expected value of breaking the tie at random
    score = np.asarray(score, float)
    budget = q * len(score)
    w = np.zeros(len(score))
    done = 0.0
    for v in np.unique(score)[::-1]:
        idx = np.flatnonzero(score == v)
        take = min(len(idx), budget - done)
        if take <= 0:
            break
        w[idx] = take / len(idx)
        done += take
    return w


def gains(t, action, e):
    # per trace, replacing it scored on half e: accuracy change and token change
    r, L = (t[f"rt{e}"], t[f"Lt{e}"]) if action == "restart" else (t[f"rp{e}"], t[f"Lp{e}"])
    return (r - (1 - t.y)).values, (t.cut + L - t.src_tokens).values


def scores(t, preds):
    # score used to rank when half h decides; higher = flag first. None-half scores do not use rollouts
    out = {
        "oracle r_hat (low prefix r first)": lambda h: 1 - t[f"rp{h}"].values,
        "oracle advantage (r_t0 - r_prefix)": lambda h: (t[f"rt{h}"] - t[f"rp{h}"]).values,
        "label oracle (source wrong first)": lambda h: t.y.values.astype(float),
    }
    for name, p in (preds or {}).items():
        out[name] = (lambda v: lambda h: v)(p)
    return out


def policy(t, score_fn, action, q):
    # cross-fitted per-trace contributions: half h ranks, half 1-h scores the replacement
    acc, tok, flag = np.zeros(len(t)), np.zeros(len(t)), np.zeros(len(t))
    for h in (0, 1):
        w = flag_weights(score_fn(h), q) if score_fn is not None else np.full(len(t), q)  # None = random
        g, dt = gains(t, action, 1 - h)
        acc += w * g / 2
        tok += w * dt / 2
        flag += w / 2
    return acc, tok, flag


def summarize(t, acc, tok, base_acc, draws):
    # trace-weighted means with a problem-clustered bootstrap; draws are shared across all policies
    # at a checkpoint, so a difference of two policies is a paired bootstrap
    g = t.problem_id.values
    pids = np.unique(g)
    col = np.searchsorted(pids, g)
    n = np.bincount(col, minlength=len(pids)).astype(float)
    sa = np.bincount(col, acc, minlength=len(pids))
    sb = np.bincount(col, base_acc, minlength=len(pids))
    st = np.bincount(col, tok, minlength=len(pids))
    den = draws @ n
    return {"d_acc": acc.mean(), "boot": draws @ sa / den, "over_random": (acc - base_acc).mean(),
            "boot_over": draws @ (sa - sb) / den, "d_tokens": tok.mean(), "boot_tok": draws @ st / den}


def run(samples, traces, preds_dir=None, out=None, log=print, n_boot=N_BOOT, note=None):
    rows = []
    for ckpt in CKPTS:
        t = table(samples, traces, ckpt)
        if not len(t):
            continue
        preds = None
        if preds_dir:
            p = load_preds(os.path.join(preds_dir, f"{CKPTS[ckpt]}.parquet"))
            preds = {}
            for arm, layer in PROBE_ARMS:
                a = p[(p.arm == arm) & (p.layer == layer)].set_index("trace_id")
                if not len(a):
                    continue
                miss = [i for i in t.trace_id if i not in a.index]
                if miss:
                    raise SystemExit(f"{arm}: {len(miss)} rollout traces have no prediction (e.g. {miss[:3]})")
                preds[arm if layer == -999 else f"{arm} L{layer}"] = a.loc[t.trace_id, "p"].values
        draws = S.problem_draws(t.problem_id.nunique(), n_boot, seed=0)
        src_acc = 1 - t.y.mean()
        log(f"{CKPTS[ckpt]}: {len(t)} traces / {t.problem_id.nunique()} problems, source accuracy {src_acc:.3f}, "
            f"mean r_hat prefix {t[['rp0', 'rp1']].values.mean():.3f}, t0 {t[['rt0', 'rt1']].values.mean():.3f}")
        for action in ("restart", "resample"):
            for q in RATES:
                rand_acc, rand_tok, _ = policy(t, None, action, q)
                pols = {"random": None, **scores(t, preds)}
                for name, fn in pols.items():
                    acc, tok, flag = (rand_acc, rand_tok, np.full(len(t), q)) if fn is None else policy(t, fn, action, q)
                    s = summarize(t, acc, tok, rand_acc, draws)
                    lo, hi = S.ci(s["boot"])
                    olo, ohi = S.ci(s["boot_over"])
                    rows.append({"checkpoint": CKPTS[ckpt], "action": action, "score": name, "q": q,
                                 "flagged": flag.mean(), "source_acc": src_acc, "d_acc": s["d_acc"],
                                 "ci_low": lo, "ci_high": hi, "over_random": s["over_random"],
                                 "over_ci_low": olo, "over_ci_high": ohi,
                                 "over_p": S.boot_p(s["boot_over"]) if fn is not None else np.nan,
                                 "d_tokens_per_trace": s["d_tokens"],
                                 "src_tokens_per_trace": t.src_tokens.mean(),
                                 "traces": len(t), "problems": t.problem_id.nunique()})
    res = pd.DataFrame(rows)
    if out:
        os.makedirs(out, exist_ok=True)
        res.to_csv(os.path.join(out, "oracle_curve.csv"), index=False)
        md = report(res, note)
        open(os.path.join(out, "oracle_summary.md"), "w").write(md)
        log("\n" + md)
        plot(res, out)
    return res


def report(res, note=None):
    pp = lambda v: f"{100 * v:+.1f}"
    cell = lambda r: f"{pp(r.over_random)} [{pp(r.over_ci_low)}, {pp(r.over_ci_high)}]"
    lines = ["# Oracle-r_hat intervention upper bound\n"] + ([f"> {note}\n"] if note else []) + [
             "- A monitor flags a share q of traces at the cut; each flagged trace is replaced. **restart**: a fresh "
             "sample from the prompt, scored by the problem's t0 rollouts. **resample**: a fresh continuation of the same "
             "prefix, scored by that prefix's rollouts. Unflagged traces keep their own outcome.",
             "- Cross-fitted: one half of each prefix's samples ranks, the other half scores the replacement, then swap.",
             "- **Value over random** = accuracy change minus that of flagging a random q of traces with the same action. "
             "Random flagging already gains, because the population is selected (mixed problems, untruncated traces "
             ">= 4096 tokens) and its source accuracy sits below r_hat; the difference removes that.",
             "- *oracle r_hat* ranks by the prefix's own rollout success; *oracle advantage* by r(t0) - r(prefix), the "
             "expected gain of a restart; both are the most a prefix-only monitor could know. *label oracle* knows how the "
             "source trace actually ended, which no prefix-only monitor can.",
             "- Accuracy in percentage points of the whole population; tokens = extra generated tokens per trace in the "
             "population (cut + replacement length - the source's own length, over flagged traces). CIs: problem bootstrap.\n"]
    for action in ("restart", "resample"):
        g = res[res.action == action]
        lines.append(f"## {action}: value over random flagging, pp [95% CI]\n")
        lines.append("| checkpoint | score | " + " | ".join(f"q={q}" for q in SHOW) + " | best q: pp (tokens/trace) |")
        lines.append("|---|---|" + "---|" * len(SHOW) + "---|")
        for (c, name), h in g[g.score != "random"].groupby(["checkpoint", "score"], sort=False):
            h = h.set_index("q")
            best = h.over_random.idxmax()
            lines.append(f"| {c} | {name} | " + " | ".join(cell(h.loc[q]) for q in SHOW) +
                         f" | {pp(h.at[best, 'over_random'])} at q={best} ({h.at[best, 'd_tokens_per_trace']:+,.0f}) |")
        rnd = g[g.score == "random"].set_index(["checkpoint", "q"])
        lines.append("\nRandom flagging itself (accuracy change, pp; tokens/trace): " + "; ".join(
            f"{c} q={q}: {pp(rnd.at[(c, q), 'd_acc'])} ({rnd.at[(c, q), 'd_tokens_per_trace']:+,.0f})"
            for c in rnd.index.get_level_values(0).unique() for q in (0.2, 0.5)) + "\n")
    # plain reading of the restart ceiling
    r = res[(res.action == "restart") & (res.score == "oracle advantage (r_t0 - r_prefix)")]
    if len(r):
        lines.append("## In plain words\n")
        for c, h in r.groupby("checkpoint", sort=False):
            b = h.loc[h.over_random.idxmax()]
            lab = res[(res.action == "restart") & (res.score == "label oracle (source wrong first)") &
                      (res.checkpoint == c)].over_random.max()
            lines.append(f"- At {c} tokens, a perfect prefix-only monitor that restarts flagged traces gains at most "
                         f"{pp(b.over_random)} pp over random flagging (q={b.q}, {b.d_tokens_per_trace:+,.0f} tokens per "
                         f"trace; total accuracy change {pp(b.d_acc)} pp from {b.source_acc:.3f}). Knowing the source's "
                         f"actual outcome would give at most {pp(lab)} pp.")
        lines.append("- These are upper bounds: the q with the best value is picked on the same data, and K=4 per half "
                     "makes r_hat noisy, which pulls the oracle rows down; the label oracle row has no such noise.")
    return "\n".join(lines) + "\n"


def plot(res, out):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    g = res[(res.action == "restart") & (res.score != "random")]
    cks = list(dict.fromkeys(g.checkpoint))
    fig, axes = plt.subplots(1, len(cks), figsize=(4.2 * len(cks), 3.6), dpi=150, sharey=True, squeeze=False)
    for ax, c in zip(axes[0], cks):
        for name, h in g[g.checkpoint == c].groupby("score", sort=False):
            ax.fill_between(h.q, 100 * h.over_ci_low, 100 * h.over_ci_high, alpha=0.12, lw=0)
            ax.plot(h.q, 100 * h.over_random, "-o", ms=3, lw=1.6, label=name)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_title(f"{c} tokens", fontsize=10, loc="left")
        ax.set_xlabel("share of traces flagged and restarted (q)")
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("accuracy gain over random flagging (pp)")
    axes[0][-1].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "oracle_restart.png"), bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", required=True, help="rollouts.jsonl from the Modal ceiling run")
    ap.add_argument("--traces", default="data/traces_B.jsonl")
    ap.add_argument("--preds", default=None, help="optional preds/ of the <=4096 probe_cmp run: probe monitors")
    ap.add_argument("--length-matched", action="store_true",
                    help="keep only continuations that pass the population's selection (needs the tokenizer)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    samples, torn = load_samples(a.rollouts)
    print(f"{len(samples)} continuations over {samples.prefix_id.nunique()} prefixes; torn lines skipped: {torn}")
    traces, note = load_traces(a.traces), None
    if a.length_matched:
        from transformers import AutoTokenizer
        ids = samples.trace_id.dropna().unique()
        samples = length_match(samples, think_tokens(traces, ids, AutoTokenizer.from_pretrained(MODEL)))
        note = (f"**Length-matched.** Only continuations whose whole trajectory passes the population's selection "
                f"(finished, </think> closed, >= {MIN_LEN} and < {MAX_NEW} tokens in all); halves re-dealt over them. "
                f"The resample control should now sit at 0. Restart here is a like-for-like comparison, not a "
                f"deployable gain: a fresh sample counts only if it too runs long, which no monitor can choose; the "
                f"raw run is the deployable reading. Prefixes or problems with < 2 kept samples are dropped, so "
                f"check the trace counts per cut.")
        print(f"length matched: {len(samples)} continuations over {samples.prefix_id.nunique()} prefixes")
    run(samples, traces, a.preds, a.out, note=note)


if __name__ == "__main__":
    main()
