# r(t) ceiling next to the probes, on identical traces. r_hat = share of the K continuations from a
# prefix that end correct. its within-problem AUROC against the trace's own outcome (score
# 1 - r_hat = P(wrong)) is the best a prefix-only predictor can do under the model's own sampling.
# read it as a lower bound: with K=8, ties and binomial noise pull it toward 0.5.
# every AUROC here uses probe_cmp's own stats (pair-weighted, problem-clustered bootstrap, draws
# shared by all arms at a checkpoint, so every ceiling - probe delta is paired).
#   python src/probe_cmp/ceiling.py --rollouts <rollouts.jsonl> --preds <probe_cmp out>/preds --out <dir>
import argparse, json, os
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
                                  "checkpoint": r["checkpoint"], "n": 0, "k": 0, "n_trunc": 0})
        d["n"] += 1
        d["k"] += bool(s["correct"])
        d["n_trunc"] += bool(s["truncated"])
    df = pd.DataFrame(list(rows.values()))
    df["r"] = df.k / df.n
    # truncated continuations are never correct; this drops them from the denominator instead
    df["r_finished"] = (df.k / (df.n - df.n_trunc).replace(0, np.nan)).fillna(df.r)
    return df, torn


def load_traces(path):
    t = pd.DataFrame(json.loads(line) for line in open(path)).set_index("trace_id", drop=False)
    t["y"] = (~t.is_correct.astype(bool)).astype(int)  # 1 = wrong, as in probe_cmp
    return t


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
    scores = {"ceiling r_hat": 1 - sub.r.values, "ceiling r_finished": 1 - sub.r_finished.values}
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
    ref = boots["ceiling r_hat"]
    d = [(r.score, r.auroc) for r in out.itertuples()]
    base = out.set_index("score").auroc["ceiling r_hat"]
    out["delta_ceiling_minus"] = [base - v for _, v in d]
    out["delta_ci_low"] = [S.ci(ref - boots[n])[0] for n, _ in d]
    out["delta_ci_high"] = [S.ci(ref - boots[n])[1] for n, _ in d]
    out["delta_p"] = [S.boot_p(ref - boots[n]) if n != "ceiling r_hat" else np.nan for n, _ in d]
    # tied pairs under r_hat: how much K=8 discreteness caps the ceiling
    dd = (1 - sub.r.values)[pairs.wrong] - (1 - sub.r.values)[pairs.correct]
    return out, float((dd == 0).mean()) if len(dd) else np.nan


def describe(r, traces):
    # mean r by the source trace's outcome, and how often a prefix is already decided
    rows = []
    for c, g in r.groupby("checkpoint"):
        lab = np.where(g.trace_id.isna(), "t0", np.where(g.y == 1, "wrong", "correct"))
        for name, h in g.groupby(lab):
            rows.append({"checkpoint": c, "source trace": name, "prefixes": len(h), "mean r": h.r.mean(),
                         "r = 0": (h.r == 0).mean(), "r = 1": (h.r == 1).mean(),
                         "truncated share": h.n_trunc.sum() / h.n.sum()})
    return pd.DataFrame(rows)


def run(roll, traces, preds_dir, out, drift=None, n_boot=N_BOOT, log=print):
    os.makedirs(out, exist_ok=True)
    tr = roll[roll.trace_id.notna()].copy()
    tr["y"] = traces.loc[tr.trace_id, "y"].values
    roll = roll.copy()
    roll["y"] = np.where(roll.trace_id.notna(), traces.reindex(roll.trace_id.fillna("")).y.values, np.nan)

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
    log("\nr_hat by checkpoint and source trace outcome\n" + desc.round(3).to_string(index=False))

    variants = [("all traces", tr)]
    if drift is not None:
        variants.append((f"without {len(drift & set(tr.trace_id))} retokenization-drift traces",
                         tr[~tr.trace_id.isin(drift)]))
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
    lines = ["# r(t) ceiling vs probes, within problem, identical traces\n",
             "- Orientation: P(wrong); AUROC over each problem's correct x wrong pairs, pair-weighted. "
             "Ceiling score = 1 - r_hat. CIs: problem-clustered bootstrap, draws shared by every row of a "
             "checkpoint, so deltas are paired.",
             "- r_hat uses K continuations per prefix; ties and binomial noise make the ceiling a lower bound. "
             "`r_finished` drops truncated continuations from the denominator.",
             "- t0 is omitted from the table: every trace of a problem shares its t0, so its within-problem "
             "AUROC is 0.5 by construction.\n"]
    for vname, g in allr.groupby("variant", sort=False):
        lines.append(f"## {vname}\n")
        lines.append("| checkpoint | score | AUROC [95% CI] | ceiling − score [95% CI] | p | traces / problems / pairs |")
        lines.append("|---|---|---|---|---|---|")
        for r in g.itertuples():
            delta = "–" if r.score == "ceiling r_hat" else f(r.delta_ceiling_minus, r.delta_ci_low, r.delta_ci_high, True)
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
    run(roll, traces, a.preds, a.out, drift)


if __name__ == "__main__":
    main()
