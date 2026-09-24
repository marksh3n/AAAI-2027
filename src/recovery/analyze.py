# recovery v2 analysis, CPU only. per-prefix and per-checkpoint summaries, variance
# decomposition, commitment, t0 sanity check, top-up list, two plots.
import argparse, json, os
from collections import defaultdict
from functools import lru_cache
import numpy as np
import pandas as pd

from common import OUT_DIR, answers_match, read_jsonl, wilson

LABELS = ("correct", "wrong")


@lru_cache(maxsize=None)
def same(a, b):
    # math_verify equivalence, symmetric and cached; None never matches
    if a is None or b is None:
        return False
    return a == b or answers_match(a, b)


def clusters(answers):
    # greedy equivalence classes over the non-null answers, largest first
    reps, sizes = [], []
    for a in answers:
        if a is None:
            continue
        for i, r in enumerate(reps):
            if same(r, a):
                sizes[i] += 1
                break
        else:
            reps.append(a)
            sizes.append(1)
    order = sorted(range(len(reps)), key=lambda i: -sizes[i])
    return [reps[i] for i in order], [sizes[i] for i in order]


def stability(answers, target):
    return float(np.mean([same(a, target) for a in answers])) if answers else float("nan")


def prefix_stats(conts, gold):
    n = len(conts)
    k = sum(c["correct"] for c in conts)
    n_trunc = sum(c["truncated"] for c in conts)
    lo, hi = wilson(k, n)
    answers = [c["answer"] for c in conts]
    reps, sizes = clusters(answers)
    lens = [c["n_gen"] for c in conts]
    return {
        "n": n, "k": k, "r": k / n if n else float("nan"), "r_lo": lo, "r_hi": hi,
        "trunc_rate": n_trunc / n if n else float("nan"),
        # truncated samples are never correct, so this is k over the finished ones
        "r_finished": k / (n - n_trunc) if n > n_trunc else float("nan"),
        "modal_answer": reps[0] if reps else None,
        "modal_agree": sizes[0] / n if reps else 0.0,
        "majority_correct": bool(reps) and same(reps[0], gold),
        "pass_at_n": k > 0,
        "answer_from_prefix": sum(c.get("answer_from_prefix", False) for c in conts),
        "mean_len": float(np.mean(lens)) if lens else float("nan"),
        "median_len": float(np.median(lens)) if lens else float("nan"),
    }


def pool_rounds(rows):
    # one row per continuation; every round's rows for a prefix, a re-written
    # (prefix, round, sample) keeps the last
    last = {(r["prefix_id"], r["round"], r["sample_idx"]): r for r in rows}
    out = defaultdict(list)
    for key in sorted(last, key=lambda k: (k[1], k[2])):
        out[key[0]].append(last[key])
    return out


def trace_table(prefixes, conts):
    # one row per (trace, checkpoint); the problem's t0 is repeated onto each of its traces
    t0 = {p["problem_id"]: p for p in prefixes if p["checkpoint"] == "t0"}
    rows = []
    by_trace = defaultdict(list)
    for p in prefixes:
        if p["trace_id"] is not None:
            by_trace[p["trace_id"]].append(p)
    for tid, ps in by_trace.items():
        base = ps[0]
        for p in ps + ([t0[base["problem_id"]]] if base["problem_id"] in t0 else []):
            if p["prefix_id"] not in conts:
                continue
            c = conts[p["prefix_id"]]
            s = prefix_stats(c, p["gold_answer"])
            rows.append({
                "trace_id": tid, "problem_id": base["problem_id"], "level": base["level"],
                "trace_label": base["trace_label"], "trace_gen_len": base["trace_gen_len"],
                "prefix_id": p["prefix_id"], "checkpoint": p["checkpoint"],
                "cut_token_idx": p["cut_token_idx"], "requested_token_idx": p["requested_token_idx"],
                "cut_frac": p["cut_token_idx"] / base["trace_gen_len"],
                "stability": stability([x["answer"] for x in c], base["trace_final_answer"]),
                **{k: s[k] for k in ("n", "k", "r", "r_lo", "r_hi", "trunc_rate", "r_finished")},
            })
    df = pd.DataFrame(rows)
    # checkpoint order is where the cut falls in this trace, never the name
    return df.sort_values(["trace_id", "cut_token_idx", "requested_token_idx"]).reset_index(drop=True)


def commitment(points, tau):
    # points: [(cut, requested, name, stability)]. earliest point from which stability
    # stays >= tau at every later cut; None if the last one is below tau.
    pts = sorted(points, key=lambda p: (p[0], p[1]))
    hit = None
    for cut, req, name, s in reversed(pts):
        if not (s >= tau):
            break
        hit = (cut, req, name, s)
    return hit


def variance(df, label=None):
    # method of moments: sampling variance of r_hat is r(1-r)/n, estimated unbiasedly
    # by r(1-r)/(n-1); what the spread of r_hat has beyond that is between-prefix
    rows = []
    for ckpt, g in df.groupby("checkpoint"):
        g = g[g.n > 1]
        if len(g) < 2:
            continue
        within = float((g.r * (1 - g.r) / (g.n - 1)).mean())
        total = float(g.r.var(ddof=1))
        between = max(total - within, 0.0)
        icc = between / (between + within) if between + within > 0 else float("nan")
        rows.append({"label": label or "all", "checkpoint": ckpt, "prefixes": len(g),
                     "within": within, "between": between, "icc": icc})
    return rows


def topup(df, prefix_r, lo, hi, drop, n):
    ids = {pid for pid, r in prefix_r.items() if lo <= r <= hi}
    for _, g in df.groupby("trace_id"):
        r, pid = g.r.values, g.prefix_id.values
        for i in range(len(g) - 1):
            if r[i] - r[i + 1] >= drop:
                ids |= {pid[i], pid[i + 1]}
    return [{"prefix_id": pid, "n": n} for pid in sorted(ids)]


def ckpt_order(df):
    # x axis: checkpoint names by median cut fraction, t0 first
    return list(df.groupby("checkpoint").cut_frac.median().sort_values().index)


def plot(df, col, ylabel, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    order = ckpt_order(df)
    med = df.groupby("checkpoint").cut_frac.median()
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, label in enumerate(LABELS):
        g = df[df.trace_label == label].groupby("checkpoint")[col]
        m = g.mean().reindex(order)
        se = (g.std(ddof=1) / np.sqrt(g.count())).reindex(order).fillna(0)
        x = np.arange(len(order)) + (i - 0.5) * 0.08
        ax.errorbar(x, m, yerr=1.96 * se, marker="o", capsize=3, label=f"{label} source trace")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([f"{c}\n({med[c]:.0%})" for c in order])
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("checkpoint (median cut as a fraction of the trace)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix-file", default=f"{OUT_DIR}/prefixes.jsonl")
    parser.add_argument("--rollouts", default=f"{OUT_DIR}/rollouts.jsonl")
    parser.add_argument("--out-dir", default=OUT_DIR)
    parser.add_argument("--tau", type=float, default=0.75, help="fix before looking at results")
    parser.add_argument("--topup-n", type=int, default=24, help="extra samples per flagged prefix")
    parser.add_argument("--topup-band", type=float, nargs=2, default=[0.2, 0.8])
    parser.add_argument("--topup-drop", type=float, default=0.25)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    prefixes = read_jsonl(args.prefix_file)
    conts = pool_rounds(read_jsonl(args.rollouts))
    by_id = {p["prefix_id"]: p for p in prefixes}
    missing = [p for p in by_id if p not in conts]
    print(f"{len(conts)}/{len(prefixes)} prefixes have rollouts"
          + (f"; {len(missing)} without" if missing else ""))

    # per prefix
    rows = []
    for pid, c in conts.items():
        p = by_id[pid]
        s = prefix_stats(c, p["gold_answer"])
        rows.append({
            "prefix_id": pid, "problem_id": p["problem_id"], "level": p["level"],
            "trace_id": p["trace_id"], "trace_label": p["trace_label"],
            "checkpoint": p["checkpoint"], "cut_token_idx": p["cut_token_idx"],
            "requested_token_idx": p["requested_token_idx"],
            "stability": (stability([x["answer"] for x in c], p["trace_final_answer"])
                          if p["trace_id"] else float("nan")),
            **s,
        })
    per_prefix = pd.DataFrame(rows).sort_values(["problem_id", "trace_id", "cut_token_idx"],
                                                na_position="first")
    per_prefix.to_csv(f"{args.out_dir}/summary_prefix.csv", index=False)

    df = trace_table(prefixes, conts)
    order = ckpt_order(df)

    # per problem x checkpoint x source label
    agg = (df.groupby(["problem_id", "checkpoint", "trace_label"])
           .agg(n=("n", "sum"), k=("k", "sum"), stability=("stability", "mean"),
                trunc_rate=("trunc_rate", "mean"), cut_frac=("cut_frac", "mean")).reset_index())
    agg["r"] = agg.k / agg.n
    ci = [wilson(k, n) for k, n in zip(agg.k, agg.n)]
    agg["r_lo"], agg["r_hi"] = [c[0] for c in ci], [c[1] for c in ci]
    agg.to_csv(f"{args.out_dir}/summary_checkpoint.csv", index=False)

    pd.set_option("display.width", 160)
    print("\nby checkpoint x source label (checkpoints ordered by median cut fraction)")
    tbl = (df.groupby(["checkpoint", "trace_label"])
           .agg(traces=("r", "size"), r=("r", "mean"), r_finished=("r_finished", "mean"),
                trunc=("trunc_rate", "mean"), stability=("stability", "mean"),
                cut_frac=("cut_frac", "median")))
    print(tbl.reindex(order, level=0).round(3).to_string())

    # variance decomposition; t0 is per problem, so it is taken from the prefix table
    print("\nvariance of r_hat: within-prefix (binomial) vs between-prefix")
    var = variance(per_prefix)
    for label in LABELS:
        var += variance(per_prefix[per_prefix.trace_label == label], label)
    var = pd.DataFrame(var)
    print(var.round(4).to_string(index=False))
    pooled = var[var.label == "all"]
    w, b = pooled.within.mean(), pooled.between.mean()
    print(f"-> mean within {w:.4f} vs between {b:.4f}: "
          + ("add samples per prefix" if w > b else "add problems"))

    # commitment
    print(f"\ncommitment (stability >= {args.tau} from that cut onward)")
    com = []
    for tid, g in df.groupby("trace_id"):
        hit = commitment(list(zip(g.cut_token_idx, g.requested_token_idx, g.checkpoint,
                                  g.stability)), args.tau)
        row = g.iloc[0]
        at = g[g.checkpoint == hit[2]].iloc[0] if hit else None
        com.append({"trace_id": tid, "trace_label": row.trace_label,
                    "commit_checkpoint": hit[2] if hit else "never",
                    "commit_frac": at.cut_frac if hit else float("nan"),
                    "r_at_commit": at.r if hit else float("nan")})
    com = pd.DataFrame(com)
    for label in LABELS:
        g = com[com.trace_label == label]
        counts = g.commit_checkpoint.value_counts().reindex(order + ["never"]).dropna().astype(int)
        print(f"  {label:>7}: {dict(counts)} | median commit frac {g.commit_frac.median():.2f} "
              f"| mean r at commit {g.r_at_commit.mean():.3f}")
    com.to_csv(f"{args.out_dir}/commitment.csv", index=False)

    # sanity: t0 against the problem's tag-B pass rate
    from scipy.stats import fisher_exact
    print("\nt0 r_hat vs tag-B pass rate (flag: Fisher p < 0.05)")
    flags = 0
    for _, r in per_prefix[per_prefix.checkpoint == "t0"].iterrows():
        p = by_id[r.prefix_id]
        pk, pn = p["problem_pass_k"], p["problem_pass_n"]
        _, pv = fisher_exact([[r.k, r.n - r.k], [pk, pn - pk]])
        flag = pv < 0.05
        flags += flag
        print(f"  problem {r.problem_id:>4}: t0 {r.k}/{r.n} = {r.r:.2f} vs tag-B {pk}/{pn} = "
              f"{pk / pn:.2f}  p={pv:.3f}{'  <-- FLAG' if flag else ''}")
    print(f"{flags} flagged")

    # top-up list
    prefix_r = dict(zip(per_prefix.prefix_id, per_prefix.r))
    up = topup(df, prefix_r, *args.topup_band, args.topup_drop, args.topup_n)
    with open(f"{args.out_dir}/topup.jsonl", "w") as f:
        for r in up:
            f.write(json.dumps(r) + "\n")
    print(f"\n{len(up)} prefixes flagged for a top-up of {args.topup_n} -> {args.out_dir}/topup.jsonl")

    plot(df, "r", "recovery r_hat", f"{args.out_dir}/r_by_checkpoint.png")
    plot(df, "stability", "stability (agrees with source trace's answer)",
         f"{args.out_dir}/stability_by_checkpoint.png")
    print(f"plots -> {args.out_dir}/r_by_checkpoint.png, stability_by_checkpoint.png")


if __name__ == "__main__":
    main()
