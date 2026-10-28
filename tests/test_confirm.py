# src/confirm on synthetic data: monitors equal probe_cmp's fits, the vote, arm costs, token matching,
# thresholds, and analyze.py end to end on a bundle with a planted monitor and a null one.
# plain script, CPU only, exits non-zero on failure.
import copy, json, os, sys, tempfile
import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src", "confirm"))
sys.path.insert(0, os.path.join(ROOT, "src", "probe_cmp"))
import spec  # noqa: E402
import monitors as MON  # noqa: E402
import policy as P  # noqa: E402
import analyze as A  # noqa: E402
import bundle_lib as BL  # noqa: E402
import models as M  # noqa: E402
import run as R  # noqa: E402
import data as D  # noqa: E402

fails = []


def check(name, cond, extra=None):
    print(f"{'ok  ' if cond else 'FAIL'} {name}{(' -- ' + str(extra)) if extra is not None else ''}")
    if not cond:
        fails.append(name)


rng = np.random.default_rng(0)

# 1. monitors reproduce probe_cmp's out-of-fold predictions exactly
n_prob, per = 24, 6
groups = np.repeat(np.arange(n_prob), per)
y = np.concatenate([rng.permutation([0, 0, 0, 1, 1, rng.integers(0, 2)]) for _ in range(n_prob)])
X = rng.normal(size=(len(y), 120)).astype(np.float32)
X[:, 3] += 1.5 * y
words = [f"w{i}" for i in range(40)]
texts = [" ".join(rng.choice(words, rng.integers(20, 60))) + (" oops" * int(yy)) + "\n\nnext" for yy in y]
tokens = np.full(len(y), 512.0)
pop = pd.DataFrame({"trace_id": [f"t{i}" for i in range(len(y))], "problem_id": groups, "y": y})
folds = R.fold_indices(pop, R.make_folds(pop, [0])["0"])
E = MON.len_features(tokens, texts)
builders = {"raw": M.Hidden(X, groups, extra=E, seed=0), "topk50": M.Hidden(X, groups, extra=E, seed=0),
            "len": M.Plain(E), "tfidf": M.Tfidf(texts)}
for model, feat in [("lr", "raw"), ("hgb", "topk50"), ("lr", "len"), ("lr", "tfidf")]:
    ref, ref_C = M.oof(builders[feat], feat, model, y, groups, folds, seed=0)
    got, got_C = MON.oof_scores(model, feat, y, groups, folds, 0, X, texts, tokens)
    check(f"{model}/{feat}: out-of-fold P(wrong) equals probe_cmp's", np.allclose(ref, got, atol=1e-10),
          float(np.max(np.abs(ref - got))))
    check(f"{model}/{feat}: same C per fold", ref_C == got_C, (ref_C, got_C))
m = MON.Monitor("lr", "raw", seed=0).fit(y, groups, X)
check("a fitted monitor scores new rows", m.score(X[:5]).shape == (5,))
tau = MON.threshold(np.arange(100.0), 0.25)
check("threshold flags a share q strictly above it", np.mean(np.arange(100.0) > tau) == 0.25, tau)

# 2. the vote: plurality, ties split, no votes is wrong
check("plurality: correct majority", P.plurality_correct(["C", "C", "W0"]) == 1.0)
check("plurality: wrong majority", P.plurality_correct(["W0", "W0", "C"]) == 0.0)
check("plurality: three-way tie is 1/3", np.isclose(P.plurality_correct(["C", "W0", "W1"]), 1 / 3))
check("plurality: None does not vote", P.plurality_correct([None, None, "C"]) == 1.0)
check("plurality: nobody answered", P.plurality_correct([None, None]) == 0.0)
# source wrong W0, bank [C, C, W0, W1]: pairs CC -> 1, C W0 -> 0 (W0 x2), C W1 -> 1/3, W0 W1 -> 0
acc = P.aggregate_acc("W0", ["C", "C", "W0", "W1"], 2)
check("maj@3 exact over the 6 pairs", np.isclose(acc, (1 + 0 + 1 / 3 + 0 + 1 / 3 + 0) / 6), acc)

# 3. arm costs and token matching, by hand
t = pd.DataFrame({"problem_id": [0, 0, 1, 1], "correct": [0, 1, 0, 1], "n": [9000.0, 3000.0, 8000.0, 2000.0],
                  "live": [True, False, True, True], "r0": [0.5, 0.5, 0.25, 0.25], "L0": [4000.0, 4000.0, 6000.0, 6000.0],
                  "r_prev": [0.75, np.nan, 0.5, 0.0], "L_prev": [3000.0, np.nan, 5000.0, 9000.0]})
gain, extra = P.arm_effect(t, "restart", 4096)
check("restart gain = r0 - outcome, 0 if not live", np.allclose(gain, [0.5, 0, 0.25, -0.75]), gain)
check("restart extra = c + L0 - n, 0 if not live", np.allclose(extra, [-904, 0, 2096, 8096]), extra)
gain_b, extra_b = P.arm_effect(t, "backtrack", 4096)
check("backtrack uses the earlier cut's bank", np.allclose(gain_b, [0.75, 0, 0.5, -1]) and np.allclose(extra_b, [-1904, 0, 1096, 11096]))
sums = P.Sums(t.problem_id.values, np.ones((3, 2)))
w = np.array([0.0, 0, 1, 0])  # policy flags attempt 2
pa, pt = t.correct + w * gain, t.n + w * extra
acc_r, pi, st = P.matched(sums, t.correct.values, t.n.values, pa, pt, gain, extra)
pi_hand = 2096 / (-904 + 2096 + 8096)
check("random rate funded to the policy's tokens", np.isclose(pi[0], pi_hand) and st[0] == 0, (pi[0], pi_hand))
check("random accuracy at that rate", np.isclose(acc_r[0], 0.5 + pi_hand * (0.5 + 0.25 - 0.75) / 4), acc_r[0])
tok_r = t.n.mean() + pi[0] * extra.sum() / 4
check("matched comparator spends exactly the policy's tokens", np.isclose(tok_r, pt.mean()), (tok_r, pt.mean()))
w2 = np.array([1.0, 0, 0, 0])  # this policy saves tokens (-904); aggregation can only add
ag_extra = 2 * t.L0.values
_, pi2, st2 = P.matched(sums, t.correct.values, t.n.values, t.correct + w2 * gain, t.n + w2 * extra, np.zeros(4), ag_extra)
check("a comparator that cannot spend less falls back to none", st2[0] == -1 and pi2[0] == 0, (st2[0], pi2[0]))
try:
    P.arm_effect(t.assign(r_prev=np.nan), "backtrack", 4096)
    check("a live attempt without a bank is an error", False)
except ValueError:
    check("a live attempt without a bank is an error", True)
w3 = P.top_share([3, 2, 2, 1], 2.0)
check("top share splits a straddling tie", np.allclose(w3, [1, 0.5, 0.5, 0]), w3)


# 4. analyze.py end to end on a synthetic bundle: a planted monitor that knows which prefixes are
# doomed must beat token-matched random restart; a pure-noise monitor must not; resample-free
def synth_bundle(d, n_problems=120, s=6, seed=1):
    r = np.random.default_rng(seed)
    src, cuts, bank, votes, scores = [], [], [], [], []
    for p in range(n_problems):
        base = r.uniform(0.2, 0.8)
        L0 = r.uniform(3000, 9000)
        bank.append({"problem_id": p, "trace_id": "", "checkpoint": 0, "k": 8, "r": base, "L": L0,
                     "r_h0": base, "r_h1": base, "L_h0": L0, "L_h1": L0})
        keys = ["C" if r.random() < base else f"W{r.integers(0, 3)}" for _ in range(8)]
        votes += [{"problem_id": p, "sample_idx": i, "answer_key": k, "correct": int(k == "C"), "n_gen": int(L0)}
                  for i, k in enumerate(keys)]
        for k in range(s):
            tid = f"{p}_{k}"
            doomed = r.random() < 0.4
            n = int(r.uniform(5000, 20000))
            rc = {c: (0.05 if doomed else min(0.95, base + 0.2)) for c in spec.CHECKPOINTS}
            correct = int(r.random() < rc[spec.DECISION]) if n > 4096 else int(r.random() < base)
            src.append({"trace_id": tid, "problem_id": p, "n": n, "correct": correct,
                        "answer_key": "C" if correct else "W0"})
            for c in spec.CHECKPOINTS:
                live = n > c
                cuts.append({"trace_id": tid, "checkpoint": c, "live": live, "pre_answer": live})
                if live:
                    bank.append({"problem_id": p, "trace_id": tid, "checkpoint": c, "k": 8, "r": rc[c], "L": 3000.0,
                                 "r_h0": rc[c], "r_h1": rc[c], "L_h0": 3000.0, "L_h1": 3000.0})
                for mon, sc in (("lr_raw_L20", float(doomed) + r.normal(0, 0.1)), ("len", r.normal()),
                                ("hgb_topk50_L20", float(doomed) + r.normal(0, 0.1)), ("tfidf", r.normal())):
                    if mon == "hgb_topk50_L20" and c != 6144:
                        continue
                    scores.append({"trace_id": tid, "checkpoint": c, "monitor": mon, "score": sc,
                                   "threshold": 0.5 if mon in ("lr_raw_L20", "hgb_topk50_L20") else 0.67})
    BL.write(d, "development", {"synthetic": True}, pd.DataFrame(src), pd.DataFrame(cuts), pd.DataFrame(bank),
             pd.DataFrame(votes), pd.DataFrame(scores))


tmp = tempfile.mkdtemp()
synth_bundle(os.path.join(tmp, "b"))
pol, auc, res = A.run(os.path.join(tmp, "b"), os.path.join(tmp, "out"), n_boot=300, log=lambda *a: None)
key = lambda mon, comp: pol[(pol.checkpoint == 4096) & (pol.monitor == mon) & (pol.action == "restart")
                            & (pol.comparator.str.startswith(comp))].iloc[0]
planted, noise = key("lr_raw_L20", "random"), key("len", "random")
check("planted monitor beats token-matched random restart", planted.delta[1] > 0, planted.delta)
# one draw of the noise flags can land outside its CI 1 time in 20; the bias check is the mean over seeds
deltas = []
for sd in range(6):
    synth_bundle(os.path.join(tmp, f"n{sd}"), seed=100 + sd)
    pn, _, _ = A.run(os.path.join(tmp, f"n{sd}"), os.path.join(tmp, f"no{sd}"), n_boot=100, log=lambda *a: None)
    deltas.append(pn[(pn.checkpoint == 4096) & (pn.monitor == "len") & (pn.action == "restart")
                     & pn.comparator.str.startswith("random")].iloc[0].delta[0])
check("noise monitor vs token-matched random is unbiased (mean of 6 seeds within 1 pp)", abs(np.mean(deltas)) < 0.01,
      np.round(100 * np.array(deltas), 2))
check("primary tests are filled", res["primary vs random"] is not None and res["primary vs aggregate2"] is not None)
check("no verdict while team slots are pending", res["primary verdict"].startswith("not evaluated"), res["primary verdict"])
check("S7 pre-answer AUROC test computed", any(s["id"] == "S7" and s.get("delta") is not None for s in res["secondary"]))
check("secondary family Holm-adjusted", all("p_holm" in s for s in res["secondary"]
                                            if s.get("delta") is not None and not s.get("estimate_only")))
s7 = next(s for s in res["secondary"] if s["id"] == "S7")
check("S7 below S7_MIN_MIXED is estimate-only, no p, not in Holm",
      s7.get("estimate_only") and "p" not in s7 and "p_holm" not in s7 and s7["mixed"] < spec.S7_MIN_MIXED, s7)
report_md = open(os.path.join(tmp, "out", "report.md")).read()
check("report says S7 is estimate only", "estimate only" in report_md and "not in the Holm family" in report_md)
# the same S7 bootstrap with enough mixed problems enters Holm; the other six keep their p values
k7 = (6144, "pre_answer")
fake_boot = {k7: {"hgb_topk50_L20": np.linspace(.55, .65, 50), "len": np.full(50, .5)}}
fake_est = {k7: {"hgb_topk50_L20": .6, "len": .5}}
big = A.tests({}, fake_boot, fake_est, None, s7_mixed=spec.S7_MIN_MIXED)
small = A.tests({}, fake_boot, fake_est, None, s7_mixed=spec.S7_MIN_MIXED - 1)
b7, s7s = (next(s for s in r["secondary"] if s["id"] == "S7") for r in (big, small))
check("S7 at S7_MIN_MIXED is tested and Holm-adjusted", "p_holm" in b7 and not b7.get("estimate_only"), b7)
check("S7 one below S7_MIN_MIXED is estimate-only", s7s.get("estimate_only") and "p_holm" not in s7s, s7s)
check("S7 with no count is estimate-only", next(s for s in A.tests({}, fake_boot, fake_est, None)["secondary"]
                                                if s["id"] == "S7").get("estimate_only"))
check("AUROC of the planted monitor is high", auc[(auc.score == "lr_raw_L20") & (auc.population == "live")].auroc.min() > 0.8)
fixed = pol[(pol.checkpoint == 4096) & pol.arm.str.startswith("fixed restart")].iloc[0]
check("fixed arm flags every live trace", np.isclose(fixed.pi, 1.0))
orc = pol[pol.arm.str.startswith("rollout-ranked")].iloc[0]
none = pol[(pol.checkpoint == 4096) & (pol.arm == "none")].iloc[0]
check("rollout-ranked diagnostic is charged its selection rollouts", orc.tokens[0] > none.tokens[0] + 4 * 3000 * 0.5)
val = pol[(pol.checkpoint == 4096) & pol.arm.str.startswith("resample validity check (every")].iloc[0]
check("resample validity check sits near 0 when sources and bank share a sampler", val.delta[1] <= 0 <= val.delta[2], val.delta)
pol_k, _, _ = A.run(os.path.join(tmp, "b"), os.path.join(tmp, "out_k"), n_boot=100, log=lambda *a: None, keep="rollout")
vk = pol_k[(pol_k.checkpoint == 4096) & pol_k.arm.str.startswith("resample validity check (every")].iloc[0]
check("keep=rollout values kept traces at their bank mean: the resample check is exactly 0", abs(vk.delta[0]) < 1e-12, vk.delta)
md = open(os.path.join(tmp, "out", "report.md")).read()
check("report avoids the retired wording", not any(w in md.lower() for w in ("perfect", "no extra token", "audit but",
                                                                            "cannot correct", "upper bound")))
spec_saved = spec.SLOTS["margin_pp"]
spec.SLOTS["margin_pp"] = 1.0
try:
    check("a margin gives a verdict", A.tests({}, {}, {}, 1.0)["primary verdict"] == "not evaluable on this bundle")
finally:
    spec.SLOTS["margin_pp"] = spec_saved

# a confirmation bundle with a hole is refused
synth = os.path.join(tmp, "b2")
synth_bundle(synth, n_problems=10, seed=2)
meta = json.load(open(os.path.join(synth, "bundle.json")))
meta["kind"] = "confirmation"
json.dump(meta, open(os.path.join(synth, "bundle.json"), "w"))
b = pd.read_parquet(os.path.join(synth, "bank.parquet"))
b.drop(index=b.index[b.checkpoint == 4096][:1]).to_parquet(os.path.join(synth, "bank.parquet"))
try:
    A.run(synth, os.path.join(tmp, "out2"), n_boot=50, log=lambda *a: None)
    check("confirmation bundle with a live cut missing its bank is refused", False)
except SystemExit as e:
    check("confirmation bundle with a live cut missing its bank is refused", "no rollout bank" in str(e))

# 5. bundle helpers
check("answer clusters: correct share C, equal wrongs share a key",
      BL.cluster_keys(["5", "7", "7", None, "5"], [True, False, False, False, True]) == ["C", "W0", "W0", None, "C"])
r = BL.cut_rules("so x = 12 and then", "12", True)
check("a coincidental mention: pre-answer under the primary rule, not under the strict one",
      r["live"] and r["pre_answer"] and not r["pre_answer_strict"] and r["pre_answer_think_boxed"], r)
r = BL.cut_rules("so the answer is 12.", "12", True)
check("a stated answer is not pre-answer (primary), but is under </think>/boxed only",
      not r["pre_answer"] and not r["pre_answer_strict"] and r["pre_answer_think_boxed"], r)
check("cue then another number is still pre-answer", BL.cut_rules("the answer is 112", "12", True)["pre_answer"])
check("cue on an earlier line does not count", BL.cut_rules("the answer is\n 12", "12", True)["pre_answer"])
check("cue is case-insensitive, \\dfrac normalized",
      not BL.cut_rules("Final Answer: \\dfrac{3}{4}", "\\frac{3}{4}", True)["pre_answer"])
check("a \\boxed answer is not pre-answer under any rule",
      not any(v for k, v in BL.cut_rules("so \\boxed{7}", "7", True).items() if k != "live"))
check("a cut past </think> is not live", BL.cut_flags("... </think> x", "3", True) == (False, False))
samples = pd.DataFrame({"problem_id": [0] * 4, "trace_id": [""] * 4, "checkpoint": [0] * 4, "sample_idx": range(4),
                        "correct": [1, 0, 1, 1], "n_gen": [10, 20, 30, 40]})
bt = BL.bank_table(samples)
check("bank halves alternate", bt.r_h0[0] == 1.0 and bt.r_h1[0] == 0.5 and bt.L_h0[0] == 20 and bt.r[0] == 0.75)

# 6. fit_one on a synthetic probe_cmp store (positions 8 and 16 stand in for the real checkpoints)
import fit_monitors as F  # noqa: E402
dd = tempfile.mkdtemp()
DIM, LAYERS = 64, [-1, 0, 4, 8, 12, 16, 20, 24, 27]
trs, rows = [], []
for p in range(30):
    ys = [0, 0, 0, 1, 1, int(rng.integers(0, 2))]
    for k in range(6):
        tid = f"{p}_{k}"
        txt = " ".join(f"w{rng.integers(0, 20)}" for _ in range(30)) + "\n\nmore"
        trs.append({"trace_id": tid, "problem_id": p, "problem": "q", "text": txt + " </think> done", "n_gen": 40,
                    "trunc": False, "is_correct": ys[k] == 0})
        for pos in (8, 16):
            rows.append((tid, p, pos, ys[k] == 0, 40, txt[: pos * 3]))
with open(os.path.join(dd, "traces_B.jsonl"), "w") as f:
    for t_ in trs:
        f.write(json.dumps(t_) + "\n")
mb = pd.DataFrame(rows, columns=["trace_id", "problem_id", "pos", "is_correct", "n_gen", "prefix"])
mb["row"] = np.arange(len(mb))
mb.to_parquet(os.path.join(dd, "meta_B.parquet"))
acts = rng.normal(size=(len(mb), len(LAYERS), DIM)).astype(np.float16)
acts[:, LAYERS.index(20), 0] += 2.0 * (~mb.is_correct.values).astype(float)
np.save(os.path.join(dd, "acts_B.npy"), acts)
reg = copy.deepcopy(D.load_registry())
reg["hidden_dim"] = DIM
store = D.Store(dd, reg=reg, names=["B_fixed"], log=lambda *a: None)
final, info, oof = F.fit_one(store, "lr_raw_L20", "16", log=lambda *a: None)
check("fit_one: trains on the mixed population at c", info["train_traces"] == 180 and info["train_problems"] == 30)
check("fit_one: planted layer-20 signal found out of fold", info["dev_oof_auroc"] > 0.75, info["dev_oof_auroc"])
check("fit_one: threshold flags about a quarter of dev live traces", abs(info["dev_live_flag_share"] - 0.25) < 0.02)
final_t, info_t, _ = F.fit_one(store, "hgb_topk50_L20", "16", log=lambda *a: None)
check("fit_one: the tree fits with C None", info_t["C"] is None and info_t["dev_oof_auroc"] > 0.6, info_t["dev_oof_auroc"])

# 7. readers of the GPU outputs: run_rollouts rows -> sources and bank, the same-engine check, cuts
import gpu_prep as G  # noqa: E402
import bundle as BU  # noqa: E402
import types  # noqa: E402
sys.path.insert(0, os.path.join(ROOT, "src", "recovery"))
for name, attrs in (("torch", {}), ("datasets", {"load_dataset": None}),
                    ("math_verify", {"parse": lambda x: x, "verify": lambda a, b: a == b}),
                    ("transformers", {"AutoTokenizer": object, "AutoModelForCausalLM": object})):
    if name not in sys.modules:
        try:
            __import__(name)
        except ImportError:  # generate.py imports these at module level; the runner's helpers do not need them
            sys.modules[name] = types.SimpleNamespace(__name__=name, **attrs)
import run_rollouts as RR  # noqa: E402
check("token_record: full ids, sha, or nothing",
      RR.token_record({0: [5, 6]}, 0, "full") == {"token_ids": [5, 6]} and "token_ids_sha256" in RR.token_record({0: [5]}, 0, "sha")
      and RR.token_record(None, 0, "none") == {})
gd = tempfile.mkdtemp()
cfg = {"model": spec.MODEL, "temp": spec.TEMPERATURE, "top_p": spec.TOP_P, "max_tokens_cap": 0, "row_format": 3,
       "revision": spec.MODEL_REVISION, "vllm": spec.VLLM_VERSION}
srcp, bankp = os.path.join(gd, "sources.jsonl"), os.path.join(gd, "bank.jsonl")
json.dump({"config_hash": "a", "config": {**cfg, "save_token_ids": "full"}}, open(srcp + ".config.json", "w"))
json.dump({"config_hash": "b", "config": {**cfg, "save_token_ids": "sha"}}, open(bankp + ".config.json", "w"))
smp = lambda i, ids, ok: {"sample_idx": i, "seed": i, "text": "x", "n_gen": len(ids), "finish_reason": "stop",
                          "answer": "4" if ok else "5", "answer_from_prefix": False, "correct": ok, "truncated": False,
                          "token_ids": ids}
row = {"prefix_id": "t0/7", "round": 0, "n": 2, "seed": 1, "max_tokens": 32768, "prefix_sha": "p", "config_hash": "a",
       "source": "confirmation", "problem_id": 7, "trace_id": None, "checkpoint": "t0", "cut_token_idx": 0,
       "samples": [smp(0, [1, 2, 3, 4, 5], True), smp(1, [1, 9, 9], False)]}
open(srcp, "w").write(json.dumps(row) + "\n" + '{"torn": ')
st = G.sources_table(srcp)
check("sources: one row per sample, torn tail skipped, ids kept", list(st.trace_id) == ["7_0", "7_1"] and st.n.tolist() == [5, 3])
BU.same_engine(srcp, bankp)
check("same engine: sources and bank differ only in what they save", True)
json.dump({"config_hash": "b", "config": {**cfg, "temp": 0.6, "save_token_ids": "sha"}}, open(bankp + ".config.json", "w"))
try:
    BU.same_engine(srcp, bankp)
    check("same engine: a bank at another temperature is refused", False)
except SystemExit:
    check("same engine: a bank at another temperature is refused", True)
brow = {**row, "round": 1, "samples": [{**smp(0, [1], True), "n_gen": 100}]}
crow = {**row, "prefix_id": "cut/7_0/p2048", "round": 0, "trace_id": "7_0", "checkpoint": "p2048", "cut_token_idx": 2048,
        "max_tokens": 32768 - 2048, "samples": [{**smp(0, [1], False), "n_gen": 50}]}
bs = BU.bank_samples([row, brow, crow])
check("bank: t0 round 0 (the sources) excluded, t0 round 1 and cuts kept",
      sorted(zip(bs.trace_id, bs.checkpoint)) == [("", 0), ("7_0", 2048)])
try:
    BU.bank_samples([{**crow, "max_tokens": 32768}])
    check("bank: a cut not given 32768 - c is refused", False)
except SystemExit:
    check("bank: a cut not given 32768 - c is refused", True)


class Tok:
    def decode(self, ids, **kw):
        return "".join({1: "a ", 2: "</think>", 3: "b ", 9: "c "}.get(i, "z ") for i in ids)


ct = G.cut_table(st, Tok(), [2, 4])
got = {(r.trace_id, r.checkpoint): (r.live, r.pre_answer) for r in ct.itertuples()}
check("cuts: only checkpoints a trace runs past", set(got) == {("7_0", 2), ("7_0", 4), ("7_1", 2)}, sorted(got))
check("cuts: </think> inside the cut makes it not live", got[("7_0", 4)] == (False, False) and got[("7_0", 2)] == (False, False)
      and got[("7_1", 2)] == (True, True), got)
check("cuts: all three pre-answer columns written", {"pre_answer_strict", "pre_answer_think_boxed"} <= set(ct.columns))

import power as W  # noqa: E402
import discordance as DC  # noqa: E402
check("power reproduces planning McNemar (N 1000, q .25, d .05)", abs(W.mcnemar_power(1000, .25, .05) - .8763091) < 1e-6)
check("power reproduces planning Tango rule-out (N 1000, q .25, d 0)", abs(W.ruleout_power(1000, .25, 0.0) - .9346756) < 1e-6)
check("power reproduces planning N 900 rows", abs(W.mcnemar_power(900, .25, .05) - .8382798) < 1e-6
      and abs(W.ruleout_power(900, .25, 0.0) - .9110449) < 1e-6)
dc = DC.run(os.path.join(tmp, "b"), n=200, log=lambda *a: None)
row = pol[(pol.checkpoint == 4096) & (pol.monitor == "lr_raw_L20") & (pol.action == "restart")
          & pol.comparator.str.startswith("aggregate m=2")].iloc[0]
check("discordance: d vs aggregation equals the dry run's primary delta", np.isclose(dc["comparators"]["aggregate2"]["d"], row.delta[0]),
      (dc["comparators"]["aggregate2"]["d"], row.delta[0]))
check("discordance: shared-source q <= independent q", all(v["q_shared"] <= v["q_indep"] + 1e-9 for v in dc["comparators"].values()))

print(f"\n{'all passed' if not fails else f'{len(fails)} FAILED: {fails}'}")
sys.exit(1 if fails else 0)
