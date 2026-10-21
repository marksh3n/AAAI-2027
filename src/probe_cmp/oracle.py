# oracle-r_hat intervention upper bound, from the existing ceiling rollouts. CPU only, no model.
# a monitor reads each trace at a cut and flags a share q of them; a flagged trace is abandoned and
# replaced. the question is how much accuracy that can buy, and at what token cost, when the monitor
# ranks by the rollout recoverability r_hat itself: the most any prefix-only monitor could do.
#   restart:   a fresh sample from the prompt; succeeds with r(t0) of its problem
#   backtrack: a fresh continuation from the trace's previous checkpoint (2048 -> 512, 4096 -> 2048);
#              succeeds with r(previous prefix). at 512 the previous point is t0, i.e. restart
#   resample:  a fresh continuation of the same prefix; succeeds with r(prefix)
# two scorings of what keeping the trace is worth:
#   selection-free (primary): r(prefix), the held-out rollout estimate. given the prefix, the trace's
#     own remainder and a rollout are draws from one distribution, so this needs nothing from the source
#     outcome and is immune to how the population was selected. resample gains exactly 0 here.
#   outcome: the source's actual outcome. resample is then the control: a prefix-only score cannot gain
#     from it in expectation, so what it shows is a gap between source outcomes and rollouts. on this
#     cohort it does not vanish (2026-10-20: -2.3 pp at 512, q=0.5, even with --length-matched), and
#     length and mixed-problem selection do not explain it (mixed_sel.py: the -0.039 residual remains
#     after mixed selection, in both BOS blocks). it is reported as unexplained.
# cross-fitted: each prefix's samples are split in two halves; one half ranks, the other scores the
# replacement and the baseline, then the halves swap and the two values are averaged.
# every gain is also reported over random flagging at the same q, which is the monitor's value.
# compute: an arm spends cut (generated, then discarded) + its replacement's length; extra tokens are
# that minus the baseline (cut + the prefix's held-out continuation length, or the source's own length
# under outcome scoring). ties in a score are flagged fractionally (the expected value of a random
# tie-break).
# the source budget is enforced offline: the rollouts ran max_tokens 32768 past the cut, so any
# continuation with cut + n_gen >= MAX_NEW counts as truncated (wrong), as generate.py's trunc rule
# (n_gen >= max_new) would have it, and its length is capped at MAX_NEW - cut.
# cohort: the matched cohort, i.e. the ceiling population (traces reaching 4096 tokens and finishing,
# mixed problems only). --problems main keeps the 52 main-block problems, whose sources were sampled
# with two BOS like the rollouts; pilot keeps the 35 one-BOS problems.
# --length-matched: keeps only continuations whose whole trajectory would have passed the cohort's own
# selection: finished, closes </think> (in the continuation or already in the prefix), at least MIN_LEN
# tokens in all. halves are re-dealt over the kept samples; a prefix with < 2 left, or a problem whose
# t0 has < 2 left, is dropped and logged.
#   python src/probe_cmp/oracle.py --rollouts <recovery_ceiling> --traces data/traces_B.jsonl \
#       [--preds <probe_cmp out>/preds] [--length-matched] [--problems all main] --out <dir>
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
PILOT_LINES = 480  # traces_B lines sampled with one BOS
PREV = {"p2048": "p512", "p4096": "p2048"}  # backtrack target; at p512 it would be t0, i.e. restart
ARMS = {"restart": "t", "backtrack": "b", "resample": "p"}  # column suffix of each arm's replacement
SCORINGS = {"selection-free": ("restart", "backtrack"), "outcome": ("restart", "backtrack", "resample")}
COHORT = "matched cohort: traces reaching 4096 tokens and finishing, mixed problems only"


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
    # the source budget, offline: a continuation reaching MAX_NEW in all is truncated under it
    over = ~df.truncated & (df.cut + df.n_gen >= MAX_NEW)
    df.attrs["over_budget"] = (int(over.sum()), int((over & df.correct).sum()))
    df.loc[over, ["truncated", "correct"]] = [True, False]
    df["n_gen"] = np.minimum(df.n_gen, MAX_NEW - df.cut)
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
            f"(truncated {g.truncated.mean():.1%}, short {((g.cut + g.n_gen - 1 < MIN_LEN) & ~g.truncated).mean():.1%}); "
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


def cohort(samples, traces, which):
    # all problems, or one generation block: main (traces_B lines >= 480, two BOS) or pilot (one BOS)
    if which == "all":
        return samples
    line = pd.Series(np.arange(len(traces)), index=traces.problem_id.values)
    pilot = set(line[line < PILOT_LINES].index)
    keep = ~samples.problem_id.isin(pilot) if which == "main" else samples.problem_id.isin(pilot)
    return samples[keep]


def half_stats(samples):
    # per prefix and half: r_hat and mean continuation length
    g = samples.groupby(["prefix_id", "half"]).agg(r=("correct", "mean"), L=("n_gen", "mean"), n=("correct", "size"))
    g = g.unstack("half")
    if g["n"].isna().any().any():
        bad = g.index[g["n"].isna().any(axis=1)][:3].tolist()
        raise SystemExit(f"prefixes with fewer than 2 samples cannot be split in halves (e.g. {bad})")
    return g


def table(samples, traces, ckpt, log=print):
    # one row per source trace at this cut, with both halves of its prefix, of its problem's t0 and,
    # where backtrack is defined, of its own previous checkpoint
    hs = half_stats(samples)
    meta = samples.drop_duplicates("prefix_id").set_index("prefix_id")
    t0 = {meta.at[p, "problem_id"]: p for p in meta.index[meta.checkpoint == "t0"]}
    prev = {meta.at[p, "trace_id"]: p for p in meta.index[meta.checkpoint == PREV.get(ckpt)]}
    rows, no_prev = [], []
    for pid in meta.index[meta.checkpoint == ckpt]:
        tid, prob = meta.at[pid, "trace_id"], meta.at[pid, "problem_id"]
        if prob not in t0:
            raise SystemExit(f"problem {prob} has no t0 rollouts; restart cannot be scored")
        if ckpt in PREV and tid not in prev:
            no_prev.append(tid)  # kept out of every arm, so the arms stay paired
            continue
        row = {"prefix_id": pid, "trace_id": tid, "problem_id": prob, "y": int(traces.at[tid, "y"]),
               "src_tokens": int(traces.at[tid, "n_gen"]), "cut": int(CKPTS[ckpt])}
        for h in (0, 1):
            row[f"rp{h}"], row[f"Lp{h}"] = hs.at[pid, ("r", h)], hs.at[pid, ("L", h)]
            row[f"rt{h}"], row[f"Lt{h}"] = hs.at[t0[prob], ("r", h)], hs.at[t0[prob], ("L", h)]
            if ckpt in PREV:
                row[f"rb{h}"], row[f"Lb{h}"] = hs.at[prev[tid], ("r", h)], hs.at[prev[tid], ("L", h)]
        rows.append(row)
    if no_prev:
        log(f"  {CKPTS[ckpt]}: {len(no_prev)} trace(s) without a {PREV[ckpt]} prefix left out: {no_prev[:5]}")
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


def gains(t, action, e, scoring="outcome"):
    # per trace, replacing it scored on half e: accuracy change and extra tokens. an arm spends the cut
    # (discarded) plus its replacement; the baseline is the trace kept
    a = ARMS[action]
    r, L = t[f"r{a}{e}"], t[f"L{a}{e}"]
    if scoring == "outcome":
        base_r, base_tok = 1 - t.y, t.src_tokens
    else:
        base_r, base_tok = t[f"rp{e}"], t.cut + t[f"Lp{e}"]
    return (r - base_r).values, (t.cut + L - base_tok).values


def scores(t, preds):
    # score used to rank when half h decides; higher = flag first. None-half scores do not use rollouts
    out = {
        "oracle r_hat (low prefix r first)": lambda h: 1 - t[f"rp{h}"].values,
        "oracle advantage (r_t0 - r_prefix)": lambda h: (t[f"rt{h}"] - t[f"rp{h}"]).values,
    }
    if "rb0" in t:
        out["oracle advantage (r_prev - r_prefix)"] = lambda h: (t[f"rb{h}"] - t[f"rp{h}"]).values
    out["label oracle (source wrong first)"] = lambda h: t.y.values.astype(float)
    for name, p in (preds or {}).items():
        out[name] = (lambda v: lambda h: v)(p)
    return out


def policy(t, score_fn, action, q, scoring="outcome"):
    # cross-fitted per-trace contributions: half h ranks, half 1-h scores the replacement and baseline
    acc, tok, flag = np.zeros(len(t)), np.zeros(len(t)), np.zeros(len(t))
    for h in (0, 1):
        w = flag_weights(score_fn(h), q) if score_fn is not None else np.full(len(t), q)  # None = random
        g, dt = gains(t, action, 1 - h, scoring)
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


def run(samples, traces, preds_dir=None, out=None, log=print, n_boot=N_BOOT, note=None, label="all problems"):
    rows = []
    for ckpt in CKPTS:
        t = table(samples, traces, ckpt, log)
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
            f"mean r_hat prefix {t[['rp0', 'rp1']].values.mean():.3f}, t0 {t[['rt0', 'rt1']].values.mean():.3f}"
            + (f", previous checkpoint {t[['rb0', 'rb1']].values.mean():.3f}" if "rb0" in t else ""))
        pols = {"random": None, **scores(t, preds)}
        for scoring, actions in SCORINGS.items():
            for action in actions:
                if action == "backtrack" and ckpt not in PREV:
                    continue
                for q in RATES:
                    rand_acc, rand_tok, _ = policy(t, None, action, q, scoring)
                    for name, fn in pols.items():
                        acc, tok, flag = ((rand_acc, rand_tok, np.full(len(t), q)) if fn is None
                                          else policy(t, fn, action, q, scoring))
                        s = summarize(t, acc, tok, rand_acc, draws)
                        lo, hi = S.ci(s["boot"])
                        olo, ohi = S.ci(s["boot_over"])
                        rows.append({"cohort": label, "checkpoint": CKPTS[ckpt], "scoring": scoring,
                                     "action": action, "score": name, "q": q, "flagged": flag.mean(),
                                     "source_acc": src_acc, "r_prefix": t[["rp0", "rp1"]].values.mean(),
                                     "d_acc": s["d_acc"], "ci_low": lo, "ci_high": hi,
                                     "over_random": s["over_random"], "over_ci_low": olo, "over_ci_high": ohi,
                                     "over_p": S.boot_p(s["boot_over"]) if fn is not None else np.nan,
                                     "d_tokens_per_trace": s["d_tokens"],
                                     "src_tokens_per_trace": t.src_tokens.mean(),
                                     "traces": len(t), "problems": t.problem_id.nunique()})
    res = pd.DataFrame(rows)
    if out:
        os.makedirs(out, exist_ok=True)
        res.to_csv(os.path.join(out, "oracle_curve.csv"), index=False)
        md = report(res, note, label)
        open(os.path.join(out, "oracle_summary.md"), "w").write(md)
        log("\n" + md)
        plot(res, out)
    return res


def report(res, note=None, label="all problems"):
    pp = lambda v: f"{100 * v:+.1f}"
    cell = lambda r: f"{pp(r.over_random)} [{pp(r.over_ci_low)}, {pp(r.over_ci_high)}]"
    sizes = res.drop_duplicates("checkpoint")
    tag = f"{COHORT}; {label}"
    lines = [f"# Oracle-r_hat intervention upper bound ({label})\n",
             f"> **{tag[0].upper() + tag[1:]}.** "
             + ", ".join(f"{r.checkpoint}: {r.traces} traces / {r.problems} problems" for r in sizes.itertuples())
             + ". Nothing here generalizes past this cohort.\n"]
    if note:
        lines.append(f"> {note}\n")
    lines += ["- A monitor flags a share q of traces at the cut; each flagged trace is replaced. **restart**: a fresh "
              "sample from the prompt (the problem's t0 rollouts). **backtrack**: a fresh continuation from the trace's "
              "previous checkpoint (2048 -> 512, 4096 -> 2048). **resample**: a fresh continuation of the same prefix.",
              "- **Selection-free** scoring (primary) values a kept trace at its prefix's held-out r_hat, so nothing "
              "depends on the source outcome. **Outcome** scoring uses the source's actual outcome; its resample row "
              "is the control and it does not sit at 0 on this cohort (a source-vs-rollout gap that length and "
              "mixed-problem selection do not explain).",
              "- Cross-fitted: one half of each prefix's samples ranks, the other half scores replacement and baseline, "
              "then swap. **Value over random** = accuracy change minus that of flagging a random q with the same arm.",
              "- *oracle r_hat* ranks by the prefix's own rollout success; *oracle advantage* by r(replacement) - "
              "r(prefix); both are the most a prefix-only monitor could know. *label oracle* ranks by how the source "
              "actually ended.",
              "- Accuracy in pp of the cohort. Tokens = extra tokens per trace: an arm spends the cut (discarded) plus "
              "its replacement, minus the baseline (cut + held-out continuation length; the source's own length under "
              "outcome scoring). Continuations reaching 32768 tokens in all count as truncated. CIs: problem bootstrap.\n"]
    for scoring, actions in SCORINGS.items():
        for action in actions:
            g = res[(res.scoring == scoring) & (res.action == action)]
            if not len(g):
                continue
            lines.append(f"## {action}, {scoring} scoring: value over random flagging, pp [95% CI] ({tag})\n")
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
    lines.append(f"## In plain words ({tag})\n")
    sf = res[res.scoring == "selection-free"]
    for action, adv in (("restart", "oracle advantage (r_t0 - r_prefix)"),
                        ("backtrack", "oracle advantage (r_prev - r_prefix)")):
        for c, h in sf[(sf.action == action) & (sf.score == adv)].groupby("checkpoint", sort=False):
            b = h.loc[h.over_random.idxmax()]
            lines.append(f"- {action} at {c}: a perfect prefix-only monitor gains at most {pp(b.over_random)} pp over "
                         f"random flagging [{pp(b.over_ci_low)}, {pp(b.over_ci_high)}] (q={b.q}, "
                         f"{b.d_tokens_per_trace:+,.0f} tokens per trace; {pp(b.d_acc)} pp over keeping every trace).")
    ctl = res[(res.scoring == "outcome") & (res.action == "resample") & (res.score == "oracle r_hat (low prefix r first)")
              & (res.q == 0.5)]
    if len(ctl):
        lines.append("- Outcome-scoring control (resample, oracle r_hat, q=0.5; 0 if source outcomes and rollouts "
                     "agree): " + "; ".join(f"{r.checkpoint}: {pp(r.over_random)} [{pp(r.over_ci_low)}, "
                                            f"{pp(r.over_ci_high)}]" for r in ctl.itertuples())
                     + ". Source accuracy vs mean prefix r_hat: " + "; ".join(
                         f"{r.checkpoint}: {r.source_acc:.3f} vs {r.r_prefix:.3f}" for r in ctl.itertuples()) + ".")
    lines.append("- These are upper bounds: the q with the best value is picked on the same data. K=4 per half makes "
                 "r_hat noisy, which pulls the oracle rows down.")
    return "\n".join(lines) + "\n"


def plot(res, out):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    g = res[(res.scoring == "selection-free") & (res.action == "restart") & (res.score != "random")]
    cks = list(dict.fromkeys(g.checkpoint))
    if not cks:
        return
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
    fig.suptitle(f"restart, selection-free scoring; {res.cohort.iloc[0]}", fontsize=9)
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
    ap.add_argument("--problems", nargs="+", default=["all"], choices=["all", "main", "pilot"],
                    help="cohorts to run, each into <out>/<name>: all 87, main block 52 (two BOS), pilot 35 (one BOS)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    samples, torn = load_samples(a.rollouts)
    n_over, n_over_ok = samples.attrs["over_budget"]
    print(f"{len(samples)} continuations over {samples.prefix_id.nunique()} prefixes; torn lines skipped: {torn}; "
          f"{n_over} finished past the 32768 source budget now count as truncated ({n_over_ok} were correct)")
    traces, note = load_traces(a.traces), None
    if a.length_matched:
        from transformers import AutoTokenizer
        ids = samples.trace_id.dropna().unique()
        samples = length_match(samples, think_tokens(traces, ids, AutoTokenizer.from_pretrained(MODEL)))
        note = (f"**Length-matched.** Only continuations whose whole trajectory passes the cohort's selection "
                f"(finished, </think> closed, >= {MIN_LEN} and < {MAX_NEW} tokens in all); halves re-dealt over them. "
                f"Restart here is a like-for-like comparison, not a deployable gain: a fresh sample counts only if it "
                f"too runs long, which no monitor can choose. Prefixes or problems with < 2 kept samples are dropped.")
        print(f"length matched: {len(samples)} continuations over {samples.prefix_id.nunique()} prefixes")
    names = {"all": "all problems", "main": "main block only (two-BOS sources, like the rollouts)",
             "pilot": "pilot block only (one-BOS sources)"}
    for which in a.problems:
        print(f"\n===== {names[which]} =====")
        run(cohort(samples, traces, which), traces, a.preds, os.path.join(a.out, which), note=note,
            label=names[which])


if __name__ == "__main__":
    main()
