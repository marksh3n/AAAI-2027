# oracle.py on synthetic rollouts: halves, fractional ties, cross-fitting, the random baseline, the
# resample control and the restart ceiling. plain script, CPU only, exits non-zero on failure.
import json, os, sys, tempfile
import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src", "probe_cmp"))
import oracle as O  # noqa: E402
from ceiling import load_traces  # noqa: E402

fails = []


def check(name, cond, extra=None):
    print(f"{'ok  ' if cond else 'FAIL'} {name}{(' -- ' + str(extra)) if extra is not None and str(extra) else ''}")
    if not cond:
        fails.append(name)


# 1. fractional ties: a group straddling the budget shares what is left
w = O.flag_weights([3, 2, 2, 1], 0.5)
check("tie group split evenly", np.allclose(w, [1, 0.5, 0.5, 0]), w)
w = O.flag_weights(np.arange(10.0), 0.3)
check("top 3 of 10 flagged", np.allclose(w, [0] * 7 + [1] * 3), w)
check("q=1 flags all", np.allclose(O.flag_weights([1, 1, 0], 1.0), 1))
check("weights sum to q * n", np.isclose(O.flag_weights(np.random.default_rng(0).integers(0, 4, 50), 0.37).sum(), 18.5))

# 2. cross-fitting: half 0 calls trace A hopeless and B fine, half 1 the reverse. with q=0.5 half 0
# flags A and is scored on half 1 (A's resample succeeds), half 1 flags B, scored on half 0
t = pd.DataFrame({"trace_id": ["A", "B"], "problem_id": [0, 0], "y": [1, 1], "src_tokens": [5000, 5000],
                  "cut": [512, 512], "rp0": [0.0, 1.0], "rp1": [1.0, 0.0], "Lp0": [100.0, 300.0],
                  "Lp1": [200.0, 400.0], "rt0": [0.5, 0.5], "rt1": [0.5, 0.5], "Lt0": [3000.0, 3000.0],
                  "Lt1": [3000.0, 3000.0]})
fn = O.scores(t, None)["oracle r_hat (low prefix r first)"]
acc, tok, flag = O.policy(t, fn, "resample", 0.5)
# A: flagged by half 0 only, scored on half 1: gain (1 - 0) / 2; B: flagged by half 1, scored on half 0: (1 - 0) / 2
check("cross-fit scores the replacement on the other half", np.allclose(acc, [0.5, 0.5]), acc)
check("cross-fit tokens from the other half", np.allclose(tok, [(512 + 200 - 5000) / 2, (512 + 300 - 5000) / 2]), tok)
check("each trace flagged in one half of two", np.allclose(flag, [0.5, 0.5]), flag)
racc, _, _ = O.policy(t, None, "restart", 0.4)
check("random flagging = q x mean gain over halves", np.allclose(racc, 0.4 * 0.5), racc)

# 3. synthetic population. each trace has a latent r; its own outcome and all its rollouts are draws
# from it, so the source and a resample are exchangeable given the prefix. every problem's t0 is 0.5
rng = np.random.default_rng(1)
d = tempfile.mkdtemp()
traces, rows = [], []
for pid in range(80):
    for j in range(6):
        tid = f"{pid}_{j}"
        r = rng.uniform(0, 1)
        ok = bool(rng.random() < r)
        traces.append({"trace_id": tid, "problem_id": pid, "is_correct": ok, "n_gen": 6000, "text": "x",
                       "trunc": False, "answer": "1"})
        for c in O.CKPTS:
            rows.append({"prefix_id": f"ceil/{tid}/{c}", "round": 0, "problem_id": pid, "trace_id": tid,
                         "checkpoint": c, "samples": [{"sample_idx": i, "correct": bool(rng.random() < r),
                                                       "n_gen": 4000, "truncated": False} for i in range(8)]})
    rows.append({"prefix_id": f"ceil/{pid}/t0", "round": 0, "problem_id": pid, "trace_id": None, "checkpoint": "t0",
                 "samples": [{"sample_idx": i, "correct": i % 2 == 0, "n_gen": 5000, "truncated": False}
                             for i in range(8)]})
# a rewritten sample keeps the last copy; a torn last line is skipped
rows.append({"prefix_id": "ceil/0_0/p512", "round": 0, "problem_id": 0, "trace_id": "0_0", "checkpoint": "p512",
             "samples": [{"sample_idx": 0, "correct": True, "n_gen": 1234, "truncated": False}]})
rp, tp = os.path.join(d, "r.jsonl"), os.path.join(d, "t.jsonl")
with open(rp, "w") as f:
    f.write("\n".join(json.dumps(r) for r in rows) + "\n" + '{"prefix_id": "torn')
with open(tp, "w") as f:
    f.write("\n".join(json.dumps(t) for t in traces) + "\n")
samples, torn = O.load_samples(rp)
check("torn line skipped", torn == 1)
check("rewritten sample kept once, last copy",
      ((samples.prefix_id == "ceil/0_0/p512") & (samples.sample_idx == 0)).sum() == 1 and
      samples[(samples.prefix_id == "ceil/0_0/p512") & (samples.sample_idx == 0)].n_gen.iloc[0] == 1234)
check("halves alternate, 4 + 4 per prefix", (samples.groupby(["prefix_id", "half"]).size() == 4).all())

out = os.path.join(d, "out")
res = O.run(samples, load_traces(tp), out=out, log=lambda *_: None, n_boot=500)
get = lambda c, a, s, q, sc="outcome": res[(res.checkpoint == c) & (res.action == a) & (res.score == s) & (res.q == q)
                                           & (res.scoring == sc)].iloc[0]
ctl = get("2048", "resample", "oracle r_hat (low prefix r first)", 0.3)
check("resample control: prefix-only oracle has no value over random", ctl.over_ci_low < 0 < ctl.over_ci_high,
      f"{ctl.over_random:+.3f} [{ctl.over_ci_low:+.3f}, {ctl.over_ci_high:+.3f}]")
lab = get("2048", "resample", "label oracle (source wrong first)", 0.3)
check("resample with the source's outcome known does gain", lab.over_ci_low > 0, f"{lab.over_random:+.3f}")
adv = get("2048", "restart", "oracle advantage (r_t0 - r_prefix)", 0.2)
labr = get("2048", "restart", "label oracle (source wrong first)", 0.2)
check("restart: oracle advantage beats random", adv.over_ci_low > 0, f"{adv.over_random:+.3f}")
check("restart: label oracle at least as good as the prefix-only oracle", labr.over_random >= adv.over_random,
      f"{labr.over_random:+.3f} vs {adv.over_random:+.3f}")
rnd = get("2048", "restart", "random", 0.5)
src = 1 - np.mean([not t["is_correct"] for t in traces])
check("random restart at q=0.5 moves accuracy halfway to t0", np.isclose(rnd.d_acc, 0.5 * (0.5 - src)),
      f"{rnd.d_acc:.4f} vs {0.5 * (0.5 - src):.4f}")
check("random restart tokens: q x (cut + t0 length - source length)",
      np.isclose(rnd.d_tokens_per_trace, 0.5 * (2048 + 5000 - 6000)), rnd.d_tokens_per_trace)
check("q=1 flags everything", np.allclose(res[res.q == 1.0].flagged, 1))
md = open(os.path.join(out, "oracle_summary.md")).read()
check("summary written", "In plain words" in md)
heads = [l for l in md.splitlines() if l.startswith("## ")]
check("every table labelled as the matched cohort", all(O.COHORT in l for l in heads), heads[:2])

# selection-free scoring: the kept trace is worth its prefix's held-out r_hat, so resample is not run,
# random restart moves accuracy by q x mean(r_t0 - r_prefix), and the source outcome cannot matter
sf = res[res.scoring == "selection-free"]
check("selection-free: arms are restart and backtrack", set(sf.action) == {"restart", "backtrack"})
t2048 = O.table(samples, load_traces(tp), "p2048", log=lambda *_: None)
want = 0.5 * np.mean([(t2048[f"rt{e}"] - t2048[f"rp{e}"]).mean() for e in (0, 1)])
rnd = sf[(sf.checkpoint == "2048") & (sf.action == "restart") & (sf.score == "random") & (sf.q == 0.5)].iloc[0]
check("selection-free random restart = q x mean(r_t0 - r_prefix)", np.isclose(rnd.d_acc, want), (rnd.d_acc, want))
want_tok = 0.5 * np.mean([(t2048[f"Lt{e}"] - t2048[f"Lp{e}"]).mean() for e in (0, 1)])
check("selection-free tokens: cut (discarded) + replacement - (cut + held-out continuation)",
      np.isclose(rnd.d_tokens_per_trace, want_tok), (rnd.d_tokens_per_trace, want_tok))
flipped = load_traces(tp)
flipped["y"] = 1 - flipped["y"]
res_f = O.run(samples, flipped, log=lambda *_: None, n_boot=500)
key = ["checkpoint", "action", "score", "q"]
a = sf[~sf.score.str.startswith("label")].set_index(key).over_random
b = res_f[(res_f.scoring == "selection-free") & ~res_f.score.str.startswith("label")].set_index(key).over_random
check("selection-free results do not depend on the source outcomes", np.allclose(a, b.loc[a.index]))
check("backtrack only where a previous checkpoint exists", set(res[res.action == "backtrack"].checkpoint) == {"2048", "4096"})
check("backtrack table reads the previous prefix", "rb0" in t2048 and
      np.isclose(t2048.rb0.iloc[0], samples[(samples.prefix_id == f"ceil/{t2048.trace_id.iloc[0]}/p512")
                                            & (samples.half == 0)].correct.mean()))
logs = []
gone = samples[samples.prefix_id != "ceil/5_0/p512"]
t_gone = O.table(gone, load_traces(tp), "p2048", log=logs.append)
check("a trace without its previous prefix leaves every arm at that cut, logged",
      "5_0" not in set(t_gone.trace_id) and len(t_gone) == len(t2048) - 1 and any("5_0" in l for l in logs))
check("curve csv written", len(pd.read_csv(os.path.join(out, "oracle_curve.csv"))) == len(res))

# 3b. with probe predictions: pre-fixed q, Holm over probes x cuts x q per arm, AUROC table, headline figure.
# len is perfect (p = wrong), the other arms noise
pdir = os.path.join(d, "preds")
os.makedirs(pdir, exist_ok=True)
rngp = np.random.default_rng(5)
for c in ("512", "2048", "4096"):
    pr = []
    for t in traces:
        for model, feat, layer in (("lr", "raw", 20), ("lr", "raw", 16), ("lr", "len", -999), ("lr", "tfidf", -999)):
            y = int(not t["is_correct"])
            pr.append({"ckpt": c, "layer": layer, "model": model, "feature": feat, "seed": 0, "trace_id": t["trace_id"],
                       "problem_id": t["problem_id"], "y": y, "p": float(y) if feat == "len" else rngp.random()})
    pd.DataFrame(pr).to_parquet(os.path.join(pdir, f"{c}.parquet"))
out2 = os.path.join(d, "out2")
res2 = O.run(samples, load_traces(tp), preds_dir=pdir, out=out2, log=lambda *_: None, n_boot=300)
fam = res2[res2.holm_p.notna()]
check("Holm only on selection-free probe rows at the pre-fixed q",
      set(fam.scoring) == {"selection-free"} and set(fam.kind) == {"probe"} and set(fam.q) == set(O.PROBE_Q))
check("Holm family per arm = probes x cuts x q", fam.groupby("action").size().to_dict() == {"restart": 4 * 3 * 3,
                                                                                         "backtrack": 4 * 2 * 3},
      fam.groupby("action").size().to_dict())
check("Holm p never below the raw Wald p", (fam.holm_p >= fam.over_p_wald - 1e-12).all())
floor = 2 / (300 + 1) * 36  # the smallest Holm p a percentile-bootstrap p could give in a 36-cell family
check("Holm can pass below the bootstrap-p floor (Wald p from the bootstrap SE)", fam.holm_p.min() < floor,
      f"min Holm p {fam.holm_p.min():.2g} vs floor {floor:.2g}")
check("Wald p tracks the bootstrap p where both are informative",
      np.corrcoef(fam.over_p_wald, fam.over_p)[0, 1] > 0.8, np.corrcoef(fam.over_p_wald, fam.over_p)[0, 1])
auc = pd.read_csv(os.path.join(out2, "auroc.csv"))
check("AUROC table: r_hat and every probe at each cut",
      set(auc.checkpoint.astype(str)) == {"512", "2048", "4096"} and
      {"r_hat", "lr/raw L20", "lr/raw L16", "len", "tfidf"} <= set(auc.score), sorted(auc.score.unique()))
check("AUROC table: a perfect score reads 1", np.allclose(auc[auc.score == "len"].auroc, 1.0))
md2 = open(os.path.join(out2, "oracle_summary.md")).read()
check("summary shows p_Holm cells and labels best q as an upper bound", "p_Holm" in md2 and "upper bound" in md2)
check("headline figure written", os.path.getsize(os.path.join(out2, "headline.png")) > 10_000)
hl, band, others = O.headline_lines(res2)
check("headline band: every probe but the primary", others == ["len", "lr/raw L16", "tfidf"], others)
check("headline lines at the pre-fixed q only", all(set(g.q) <= set(O.PROBE_Q) for g in hl.values()))
check("headline backtrack only at 2048 and 4096", set(hl["oracle backtrack (r_prev - r_prefix)"].checkpoint) == {"2048", "4096"})

# 4. a problem without t0 cannot be scored for restart
bad = samples[~((samples.checkpoint == "t0") & (samples.problem_id == 3))]
try:
    O.table(bad, load_traces(tp), "p512")
    check("missing t0 refused", False)
except SystemExit:
    check("missing t0 refused", True)

# 4b. source budget, offline: cut + n_gen >= 32768 is truncated and wrong, its length capped
bp = os.path.join(d, "budget.jsonl")
open(bp, "w").write(json.dumps({"prefix_id": "ceil/0_0/p4096", "round": 0, "problem_id": 0, "trace_id": "0_0",
                                "checkpoint": "p4096", "samples": [
                                    {"sample_idx": 0, "correct": True, "n_gen": 28671, "truncated": False},
                                    {"sample_idx": 1, "correct": True, "n_gen": 28672, "truncated": False},
                                    {"sample_idx": 2, "correct": False, "n_gen": 32768, "truncated": True}]}) + "\n")
bs, _ = O.load_samples(bp)
check("budget: 4096 + 28671 finished stays correct", bool(bs.correct[0]) and not bs.truncated[0])
check("budget: 4096 + 28672 = 32768 is truncated and wrong", bool(bs.truncated[1]) and not bs.correct[1])
check("budget: lengths capped at 32768 - cut", (bs.n_gen <= 32768 - 4096).all() and bs.n_gen[2] == 28672)
check("budget: relabelled count recorded", bs.attrs["over_budget"] == (1, 1))

# 4c. cohorts: main = problems whose traces sit at traces_B lines >= 480, pilot = the rest
tr = pd.DataFrame({"problem_id": [7] * 480 + [8] * 6, "trace_id": [f"x{i}" for i in range(486)]})
sm = pd.DataFrame({"problem_id": [7, 8, 8]})
check("cohort main drops pilot problems", list(O.cohort(sm, tr, "main").problem_id) == [8, 8])
check("cohort pilot keeps them", list(O.cohort(sm, tr, "pilot").problem_id) == [7])

# 5. length match: the keep rule on hand-built continuations
S = pd.DataFrame([
    # prefix, problem, trace, ckpt, cut, n_gen, truncated, think, expected keep
    ("a", 0, "A", "p512", 512, 5000, False, True, True),
    ("a", 0, "A", "p512", 512, 3584, False, True, False),   # 512 + 3584 - 1 = 4095 tokens: short
    ("a", 0, "A", "p512", 512, 3585, False, True, True),    # exactly 4096
    ("a", 0, "A", "p512", 512, 32256, False, True, False),  # 512 + 32256 = 32768: over the source budget
    ("a", 0, "A", "p512", 512, 32256, True, True, False),   # truncated
    ("a", 0, "A", "p512", 512, 5000, False, False, False),  # never closes </think>
    ("b", 0, "B", "p4096", 4096, 500, False, False, True),  # </think> already in B's prefix
    ("b", 0, "B", "p4096", 4096, 600, False, False, True),
    ("t", 0, None, "t0", 0, 5000, False, True, True),
    ("t", 0, None, "t0", 0, 4000, False, True, False),
    ("t", 0, None, "t0", 0, 6000, False, True, True),
], columns=["prefix_id", "problem_id", "trace_id", "checkpoint", "cut", "n_gen", "truncated", "think", "want"])
S["round"], S["sample_idx"], S["correct"] = 0, range(len(S)), True
kept = O.length_match(S.drop(columns="want"), {"A": 9000, "B": 3000}, log=lambda *_: None)
want = S[S.want]
check("length match: keep rule (length, budget, truncation, </think> in prefix or continuation)",
      sorted(kept.sample_idx) == sorted(want.sample_idx), sorted(kept.sample_idx))
check("length match: halves re-dealt over kept samples", (kept.groupby("prefix_id").half.apply(list) ==
      pd.Series({"a": [0, 1], "b": [0, 1], "t": [0, 1]})).all())
thin = O.length_match(S[S.sample_idx != 10].drop(columns="want"), {"A": 9000, "B": 3000}, log=lambda *_: None)
check("length match: a t0 left with < 2 drops its problem", thin.empty)

# 6. sources selected on their own future: a trace is kept only if it runs >= 4096 tokens, and wrong
# continuations run long more often. the raw resample control then goes negative (flagging low-r
# prefixes skips the high-r ones, where selection pushed the source furthest below r); matching the
# continuations to the same selection brings it back to 0
def selected(seed, K=16, long_wrong=0.9, long_right=0.2):
    rng = np.random.default_rng(seed)
    traces, rows = [], []
    for pid in range(80):
        for j in range(6):
            tid, r = f"{pid}_{j}", rng.uniform(0, 1)
            ok = bool(rng.random() < r * long_right / (r * long_right + (1 - r) * long_wrong))  # given long
            traces.append({"trace_id": tid, "problem_id": pid, "is_correct": ok, "n_gen": 6000, "text": "x</think>",
                           "trunc": False, "answer": "1"})
            sm = []
            for i in range(K):
                c = bool(rng.random() < r)
                long = rng.random() < (long_right if c else long_wrong)
                sm.append({"sample_idx": i, "correct": c, "n_gen": (6000 if long else 3000) - 512,
                           "truncated": False, "text": "</think>"})
            rows.append({"prefix_id": f"ceil/{tid}/p512", "round": 0, "problem_id": pid, "trace_id": tid,
                         "checkpoint": "p512", "samples": sm})
        rows.append({"prefix_id": f"ceil/{pid}/t0", "round": 0, "problem_id": pid, "trace_id": None,
                     "checkpoint": "t0", "samples": [{"sample_idx": i, "correct": i % 2 == 0, "n_gen": 5000,
                                                      "truncated": False, "text": "</think>"} for i in range(K)]})
    dd = tempfile.mkdtemp()
    rp, tp = os.path.join(dd, "r.jsonl"), os.path.join(dd, "t.jsonl")
    open(rp, "w").write("\n".join(json.dumps(r) for r in rows) + "\n")
    open(tp, "w").write("\n".join(json.dumps(t) for t in traces) + "\n")
    return O.load_samples(rp)[0], load_traces(tp)


smp, trc = selected(2)
ctl = lambda df: (lambda r: r[(r.action == "resample") & (r.score == "oracle r_hat (low prefix r first)") & (r.q == 0.5)
                             & (r.scoring == "outcome")]
                  .iloc[0])(O.run(df, trc, log=lambda *_: None, n_boot=500))
raw, matched = ctl(smp), ctl(O.length_match(smp, {}, log=lambda *_: None))
check("selected sources: raw resample control is negative", raw.over_ci_high < 0,
      f"{raw.over_random:+.3f} [{raw.over_ci_low:+.3f}, {raw.over_ci_high:+.3f}]")
check("selected sources: length-matched control contains 0", matched.over_ci_low < 0 < matched.over_ci_high,
      f"{matched.over_random:+.3f} [{matched.over_ci_low:+.3f}, {matched.over_ci_high:+.3f}]")

print(f"\n{len(fails)} failure(s)" + (f": {fails}" if fails else ""))
sys.exit(1 if fails else 0)
