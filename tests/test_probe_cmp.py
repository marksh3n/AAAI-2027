# Plain script, exits non-zero on failure. CPU only, synthetic data, well under a minute.
import copy, json, os, subprocess, sys, tempfile
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src", "probe_cmp"))
import data as D  # noqa: E402
import models as M  # noqa: E402
import run as R  # noqa: E402
import stats as S  # noqa: E402

fails = []


def check(name, cond, extra=""):
    print(f"{'ok  ' if cond else 'FAIL'} {name}{(' -- ' + extra) if extra else ''}")
    if not cond:
        fails.append(name)


# ---------- synthetic E0-shaped data: 20 problems x 6 traces, positions 8 and 16, an end,
# and a "Mark-style" intermediate source at 24 added through the registry alone
DIM, LAYERS, L20 = 80, [-1, 0, 4, 8, 12, 16, 20, 24, 27], 6


def make_data(d, rng):
    traces, tokens = [], {}
    for p in range(20):
        ys = np.array([0, 0, 0, 1, 1, rng.integers(0, 2)])
        for k in range(6):
            n = int(rng.integers(18, 40))
            toks = [f"w{rng.integers(0, 30)}" + (".\n\n" if rng.random() < 0.2 else " ") for _ in range(n - 1)]
            toks.append("</think> done")
            tid = f"{p}_{k}"
            tokens[tid] = toks
            trunc = p == 0 and k == 5  # one truncated trace, always excluded
            traces.append({"trace_id": tid, "problem_id": p, "problem": "q", "text": "".join(toks),
                           "n_gen": n, "trunc": trunc, "is_correct": bool(ys[k] == 0) and not trunc})
    with open(os.path.join(d, "traces_B.jsonl"), "w") as f:
        for t in traces:
            f.write(json.dumps(t) + "\n")
    wrong = {t["trace_id"]: int(not t["is_correct"]) for t in traces}

    def acts_for(tids):
        a = rng.normal(size=(len(tids), len(LAYERS), DIM)).astype(np.float16)
        a[:, L20, 0] += 3.0 * np.array([wrong[t] for t in tids])  # planted in layer 20 only
        return a

    rows = [(t["trace_id"], t["problem_id"], pos, t["is_correct"], t["n_gen"])
            for t in traces for pos in (8, 16) if pos <= t["n_gen"]]
    mb = pd.DataFrame(rows, columns=["trace_id", "problem_id", "pos", "is_correct", "n_gen"])
    mb["prefix"] = ["".join(tokens[t][:p]) for t, p in zip(mb.trace_id, mb.pos)]
    mb["row"] = np.arange(len(mb))
    mb.to_parquet(os.path.join(d, "meta_B.parquet"))
    np.save(os.path.join(d, "acts_B.npy"), acts_for(mb.trace_id))
    me = pd.DataFrame({"trace_id": [t["trace_id"] for t in traces], "problem_id": [t["problem_id"] for t in traces],
                       "is_correct": [t["is_correct"] for t in traces], "row": np.arange(len(traces))})
    me.to_parquet(os.path.join(d, "meta_BS_last.parquet"))
    np.save(os.path.join(d, "acts_BS_last.npy"), acts_for(me.trace_id))
    rows = [(t["trace_id"], t["problem_id"], 24, t["is_correct"]) for t in traces if t["n_gen"] >= 24]
    mm = pd.DataFrame(rows, columns=["trace_id", "problem_id", "pos", "is_correct"])
    mm["prefix"] = ["".join(tokens[t][:24]) for t in mm.trace_id]
    mm.to_parquet(os.path.join(d, "meta_M.parquet"))
    np.save(os.path.join(d, "acts_M.npy"), acts_for(mm.trace_id))
    return traces


def registry(extra=False):
    reg = copy.deepcopy(D.load_registry())
    reg["hidden_dim"] = DIM
    if extra:
        reg["sources"].insert(1, {"name": "M_mid", "acts": "acts_M.npy", "meta": "meta_M.parquet",
                                  "kind": "position", "pos_col": "pos", "layers": LAYERS,
                                  "layer_convention": "hidden_states[l+1]", "token_index": "pos-1",
                                  "prefix": "meta:prefix"})
    return reg


tmp = tempfile.mkdtemp()
rng = np.random.default_rng(0)
make_data(tmp, rng)

# ---------- within-problem AUROC, hand computed
# p1: wrong [.9, .5] vs correct [.5, .2] -> 1 + 1 + 1/2 (tie) + 1 = 3.5 of 4
# p2: wrong [.1] vs correct [.3] -> 0 of 1.  p3: all correct -> no pairs
y = np.array([1, 1, 0, 0, 1, 0, 0, 0])
g = np.array([1, 1, 1, 1, 2, 2, 3, 3])
s = np.array([.9, .5, .5, .2, .1, .3, .7, .1])
a = S.Pairs(y, g).auroc(s)
check("within AUROC weights problems by pairs", abs(a["weighted"] - 3.5 / 5) < 1e-12, f"{a['weighted']}")
check("unweighted mean of per-problem AUROC", abs(a["unweighted"] - (0.875 + 0) / 2) < 1e-12, f"{a['unweighted']}")
check("pair and problem counts", a["n_pairs"] == 5 and a["n_problems"] == 2)
pairs = S.Pairs(y, g)
bw, bu = S.boot_within(pairs, s, np.ones((3, 2)))
check("bootstrap with unit draws returns the point estimate", np.allclose(bw, 0.7) and np.allclose(bu, 0.4375))
bw, _ = S.boot_within(pairs, s, np.array([[2.0, 0.0]]))
check("a draw taking problem 1 twice scores it alone", np.isclose(bw[0], 0.875))

yy, ss = rng.integers(0, 2, 60), np.round(rng.normal(size=60), 1)
check("pooled AUROC matches sklearn with ties", np.isclose(S.pooled_auroc(yy, ss), roc_auc_score(yy, ss)))
w = rng.integers(0, 3, 60)
rep = np.repeat(np.arange(60), w)
check("weighted pooled AUROC = AUROC on duplicated rows",
      np.isclose(S.weighted_pooled(yy, ss, w[None])[0], roc_auc_score(yy[rep], ss[rep])))

# ---------- permutation keeps each problem's counts
gg = np.repeat(np.arange(30), 6)
y0 = rng.integers(0, 2, len(gg))
yp = S.permute_within(y0, gg, np.random.default_rng(1))
same = all(y0[gg == k].sum() == yp[gg == k].sum() for k in range(30))
check("within-problem permutation preserves per-problem label counts", same)
check("and actually moves labels", (yp != y0).any())

# ---------- DeLong, hand computed (structural components)
# A: pos [.9,.6] neg [.7,.2] -> V10 [1, .5], V01 [.5, 1], AUC .75
# B: pos [.8,.3] neg [.4,.5] -> V10 [1, 0],  V01 [.5,.5], AUC .5
# var = (.125 + .5 - 2*.25)/2 + (.125 + 0 - 0)/2 = .125
yd = np.array([1, 1, 0, 0])
dl = S.delong(yd, np.array([.9, .6, .7, .2]), np.array([.8, .3, .4, .5]))
check("DeLong AUCs", np.isclose(dl["auc1"], .75) and np.isclose(dl["auc2"], .5))
check("DeLong variance 0.125 (hand)", np.isclose(dl["se"], np.sqrt(.125)), f"se {dl['se']}")
check("DeLong z and p", np.isclose(dl["z"], .25 / np.sqrt(.125)) and np.isclose(dl["p"], 0.47950, atol=1e-5),
      f"z {dl['z']:.4f} p {dl['p']:.5f}")


def naive_delong_var(y, a, b):
    pos, neg = y == 1, y == 0
    comps = []
    for s_ in (a, b):
        psi = (s_[pos][:, None] > s_[neg][None]) + 0.5 * (s_[pos][:, None] == s_[neg][None])
        comps.append((psi.mean(1), psi.mean(0)))
    s10 = np.cov([comps[0][0], comps[1][0]])
    s01 = np.cov([comps[0][1], comps[1][1]])
    v = s10 / pos.sum() + s01 / neg.sum()
    return v[0, 0] + v[1, 1] - 2 * v[0, 1]


ya = rng.integers(0, 2, 80)
sa, sb = np.round(rng.normal(size=80) + ya, 1), np.round(rng.normal(size=80), 1)
dl = S.delong(ya, sa, sb)
check("fast DeLong = naive O(mn) DeLong with ties", np.isclose(dl["se"] ** 2, naive_delong_var(ya, sa, sb)))
check("DeLong AUC = sklearn", np.isclose(dl["auc1"], roc_auc_score(ya, sa)))

# ---------- registry and alignment asserts
reg = registry()
store = D.Store(tmp, reg=reg, log=lambda *_: None)
check("checkpoints come from the registered files", store.ckpts == ["8", "16", "end"], f"{store.ckpts}")
store_m = D.Store(tmp, reg=registry(extra=True), log=lambda *_: None)
check("a new source is one registry entry, no code change", store_m.ckpts == ["8", "16", "24", "end"],
      f"{store_m.ckpts}")
p3, p4 = D.population(store, store.ckpts), D.population(store_m, store_m.ckpts)
check("fixed population = traces reaching every checkpoint",
      set(p4.trace_id) <= set(p3.trace_id) and (store.traces.loc[p4.trace_id, "n_gen"] >= 24).all())
check("truncated trace excluded", "0_5" not in set(p3.trace_id))
check("only mixed problems", all(0 < v.sum() < len(v) for _, v in p3.groupby("problem_id").y))

meta = pd.read_parquet(os.path.join(tmp, "meta_B.parquet"))
acts = np.load(os.path.join(tmp, "acts_B.npy"), mmap_mode="r")
src = reg["sources"][0]
ok = lambda m, a_: all(r[1] for r in D.check_pair(src, m, a_, store.traces, DIM))
check("clean pair passes", ok(meta, acts))
bad = meta.copy()
bad.loc[3, "is_correct"] = not bad.loc[3, "is_correct"]
check("a flipped label is caught", not ok(bad, acts))
check("a row-count mismatch is caught", not ok(meta.iloc[:-1], acts))
gap = meta[~((meta.pos == 8) & (meta.trace_id == meta.trace_id.iat[0]))].assign(row=lambda d: np.arange(len(d)))
check("a missing checkpoint row (gap) is caught", not ok(gap, np.asarray(acts)[gap.index.values]))
shifted = meta.copy()
shifted["prefix"] = shifted.prefix.shift(1).fillna("")
check("a prefix from another row is caught", not ok(shifted, acts))

full_prefix = meta.assign(prefix=[store.traces.at[t, "text"] for t in meta.trace_id])
check("a prefix holding the whole trace (future text) is caught", not ok(full_prefix, acts))

# a relative-checkpoint source (labels are fractions, pos differs per trace) is still one entry
rel = pd.DataFrame([(t, store.traces.at[t, "problem_id"], f, max(1, int(f * store.traces.at[t, "n_gen"])),
                     store.traces.at[t, "is_correct"]) for t in store.traces.index for f in (0.5, 0.75)],
                   columns=["trace_id", "problem_id", "frac", "pos", "is_correct"])
rel["prefix"] = [store.traces.at[t, "text"][: 3 * q] for t, q in zip(rel.trace_id, rel.pos)]
rel.to_parquet(os.path.join(tmp, "meta_R.parquet"))
np.save(os.path.join(tmp, "acts_R.npy"), rng.normal(size=(len(rel), len(LAYERS), DIM)).astype(np.float16))
reg_r = registry()
reg_r["sources"].insert(1, {"name": "R_frac", "acts": "acts_R.npy", "meta": "meta_R.parquet", "kind": "position",
                            "pos_col": "pos", "name_col": "frac", "layers": LAYERS,
                            "layer_convention": "hidden_states[l+1]", "token_index": "pos-1", "prefix": "meta:prefix"})
store_r = D.Store(tmp, reg=reg_r, log=lambda *_: None)
check("fraction-labelled source loads from a registry entry alone", {"0.5", "0.75"} <= set(store_r.ckpts),
      f"{store_r.ckpts}")
check("relative checkpoints sit on the token axis", store_r.ckpts.index("16") < store_r.ckpts.index("0.75"),
      f"{store_r.ckpts}")

check("bootstrap p is never 0", S.boot_p(np.ones(2000)) == 2 / 2001)

ids = p3.trace_id.values
E = store.len_features("8", ids)
check("len uses tokens observed, not the final length", np.allclose(np.expm1(E[:, 0]), 8))
full = [store.traces.at[t, "text"] for t in ids]
check("paragraphs are counted on the prefix only",
      (np.expm1(E[:, 1]) <= np.array([D.paragraphs(x) for x in full])).all())

# ---------- folds
pop = p3
fmap = R.make_folds(pop, [0, 1, 2])
for sd in (0, 1, 2):
    fi = R.fold_indices(pop, fmap[str(sd)])
    split = all(not set(pop.problem_id.values[tr]) & set(pop.problem_id.values[te]) for tr, te in fi)
    cover = sorted(np.concatenate([te for _, te in fi])) == list(range(len(pop)))
    check(f"seed {sd}: no problem in both train and test; every trace tested once", split and cover)
inner = M.inner_splits(fi[0][0], pop.y.values, pop.problem_id.values, 0)
check("inner CV is grouped and stays inside the training fold",
      all(not set(pop.problem_id.values[a]) & set(pop.problem_id.values[b]) and set(a) | set(b) <= set(fi[0][0])
          for a, b in inner))

# ---------- fold-local fitting: a spy records what each transform is fitted on
X = store.hidden("8", ids, 20)
yv, gv = pop.y.values, pop.problem_id.values
tr, te = fi[0]
seen = []


class SpyScaler(StandardScaler):
    def fit(self, X_, y_=None, sample_weight=None):
        seen.append(("scaler", X_.shape[0]))
        return super().fit(X_, y_, sample_weight)


orig_scaler, M.StandardScaler = M.StandardScaler, SpyScaler
build = M.Hidden(X, gv, extra=E, seed=0)
outs = {k: build(k, tr, te, yv) for k in ("raw", "svd50", "topk50", "svd50+len")}
M.StandardScaler = orig_scaler
check("every scaler is fitted on the training rows only", seen and all(n == len(tr) for _, n in seen), f"{seen}")
X2, y2 = X.copy(), yv.copy()
X2[te] = 1e6 * rng.normal(size=X2[te].shape)
y2[te] = 1 - y2[te]
build2 = M.Hidden(X2, gv, extra=E, seed=0)
for k in outs:
    same = np.allclose(build2(k, tr, te, y2)[0], outs[k][0])
    check(f"{k}: training features unchanged when test rows and labels change", same)
sub = M.Hidden(X[tr], gv[tr], extra=E[tr], seed=0)
a1 = sub("svd50", np.arange(len(tr)), np.arange(len(tr)), yv[tr])[0]
check("SVD features equal an SVD fitted on the training subset alone", np.allclose(np.abs(a1), np.abs(outs["svd50"][0])))

docs = [store.traces.at[t, "text"] for t in ids]
tf = M.Tfidf(docs)
Ftr, Fte = tf("tfidf", tr, te, yv)
ref = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit([docs[i] for i in tr])
check("fold tf-idf == TfidfVectorizer(min_df=2) fitted on training docs",
      np.allclose(Ftr.toarray(), ref.transform([docs[i] for i in tr]).toarray())
      and np.allclose(Fte.toarray(), ref.transform([docs[i] for i in te]).toarray()))

# ---------- end to end: run.py --quick on the synthetic data, planted signal recovered
reg_path = os.path.join(tmp, "registry.json")
json.dump(registry(extra=True), open(reg_path, "w"))
out = os.path.join(tmp, "out")
cmd = [sys.executable, os.path.join(ROOT, "src", "probe_cmp", "run.py"), "--data", tmp, "--out", out,
       "--registry", reg_path, "--quick", "--perms", "4", "--jobs", "1", "--min-problems", "5"]
r = subprocess.run(cmd, capture_output=True, text=True)
check("run.py --quick exits cleanly", r.returncode == 0, r.stderr[-800:])
q = os.path.join(out, "quick")
if r.returncode == 0:
    for f in ("population.json", "results_long.csv", "paired_deltas.csv", "perm_null.csv", "summary.md",
              "within_auroc_by_checkpoint.png"):
        check(f"wrote {f}", os.path.exists(os.path.join(q, f)))
    check("folds.json shared at the output root", os.path.exists(os.path.join(out, "folds.json")))
    long = pd.read_csv(os.path.join(q, "results_long.csv"))
    ww = long[long.metric == "within_weighted"]
    # one planted column among 80 with ~75 training rows: top-K pipelines find it, raw LR is diluted
    v = ww[ww.arm == "hgb/topk50"].value
    check("planted layer-20 signal recovered (hgb/topk50)", (v > 0.85).all(), f"{v.round(3).tolist()}")
    v = ww[ww.arm == "lr/raw"].value
    check("lr/raw clearly above chance on it", (v > 0.6).all(), f"{v.round(3).tolist()}")
    lens = long[(long.arm == "len") & (long.metric == "within_weighted")].value
    check("len is near chance on unrelated lengths", (lens.between(0.2, 0.8)).all(), f"{lens.round(3).tolist()}")
    d = pd.read_csv(os.path.join(q, "paired_deltas.csv"))
    check("primary rows are layer-20 lr/raw vs len only",
          set(map(tuple, d[d.role == "primary"][["arm_a", "arm_b"]].values)) == {("lr/raw", "len")}
          and (d[d.role == "primary"].layer == 20).all())
    summ = open(os.path.join(q, "summary.md")).read()
    check("summary states orientation and layer convention", "P(wrong)" in summ and "hidden_states[l+1]" in summ)
    r2 = subprocess.run(cmd, capture_output=True, text=True)
    check("a rerun resumes from saved predictions", "already saved, skipping" in r2.stdout, r2.stdout[-300:])
    check("default run is not flagged exploratory", "Exploratory run" not in summ)

# ---------- plain summary reads the sign: a significant negative delta is not "beats length"
import report  # noqa: E402
res = {(c, 20, "lr/raw"): {"within": v, "ci": (v - .1, v + .1)} for c, v in (("a", .42), ("b", .55), ("end", .78))}
res[("end", report.NO_LAYER, "len")] = {"within": .77, "ci": (.7, .84)}
dl = pd.DataFrame([{"layer": 20, "checkpoint": c, "arm_a": "lr/raw", "arm_b": "len", "delta": d, "ci_low": d - .1,
                    "ci_high": d + .1, "p_boot": p, "role": "primary", "p_holm_primary": h}
                   for c, d, p, h in (("a", -.13, .004, .03), ("b", .02, .5, 1.0), ("end", .01, .6, 1.0))])
early = report.plain_lines(["a", "b", "end"], res, dl, pd.DataFrame(), [20])[1]
check("significant negative delta is not reported as beating length",
      "beat length" not in early and "below" in early and "no early signal beyond length" in early, early)
dl.loc[dl.checkpoint == "b", ["delta", "p_holm_primary"]] = [.12, .01]
early = report.plain_lines(["a", "b", "end"], res, dl, pd.DataFrame(), [20])[1]
check("significant positive delta is reported as beating length", "beat length early, at b" in early, early)

# ---------- extra layers + --exploratory: own output dir, flagged summary, layer 20 kept
cmd3 = [sys.executable, os.path.join(ROOT, "src", "probe_cmp", "run.py"), "--data", tmp, "--out", out,
        "--registry", reg_path, "--checkpoints", "8", "--layers", "12", "--exploratory", "--perms", "0",
        "--jobs", "1", "--min-problems", "5"]
r3 = subprocess.run(cmd3, capture_output=True, text=True)
check("run.py --layers 12 --exploratory exits cleanly", r3.returncode == 0, r3.stderr[-800:])
e = os.path.join(out, "L20-12", "subset_8")
if r3.returncode == 0:
    check("extra layers write to their own dir, not over the default run", os.path.exists(os.path.join(e, "summary.md"))
          and not os.path.exists(os.path.join(out, "subset_8")))
    check("summary flagged exploratory", "Exploratory run" in open(os.path.join(e, "summary.md")).read())
    check("population.json records the flag", json.load(open(os.path.join(e, "population.json")))["exploratory"])
    ls = set(pd.read_csv(os.path.join(e, "results_long.csv")).layer.astype(str))
    check("layer 20 kept alongside 12", {"20", "12"} <= ls, str(sorted(ls)))

print()
if fails:
    print(f"{len(fails)} FAILED: {fails}")
    sys.exit(1)
print("all passed")
