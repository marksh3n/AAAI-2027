# Phase 4 audit of X1's headline (D is sizeable, readouts capture little), on all 87 cohort problems, exploratory:
#   halves   D with alternating halves (0,2,4,6 / 1,3,5,7, as the confirmation bank) vs blocks (0-3 / 4-7)
#   level    D by MATH level
#   pre      D on traces with no </think> or \boxed in the prefix
#   readouts stronger than X1's: tf-idf (word 1-2 grams) and a 4-layer probe (L12, 16, 20, 24), both logistic on the
#            source outcome over all dev traces at the cut (grouped out of fold); a nonlinear probe (boosted trees
#            on 50 principal components of L20) and a 4-layer ridge, both fit to within-problem-centered r_hat of
#            other cohort problems
#   python src/explore/robustness.py --rollouts data/recovery_ceiling --traces <traces_B.jsonl> --meta <meta_B.parquet> \
#       --acts <acts_B.npy> --out <dir>
import argparse, json, os, sys
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "probe_cmp"))
import stats as S  # noqa: E402
from cheap_rollouts import load  # noqa: E402
from determination import ALPHAS, inner_pick, probe_r, probe_y, sums, total_sums  # noqa: E402

SLOTS4 = [4, 5, 6, 7]  # layers 12, 16, 20, 24 in acts_B


def d_share(t, A, B, n_boot=2000):
    g = t.problem_id.values
    fix, tot = sums(g, A, B), total_sums(t)
    pids = fix.index.values
    f, o = fix.loc[pids].values, tot.loc[pids].values
    db = (S.problem_draws(len(pids), n_boot, 0) @ f) / (S.problem_draws(len(pids), n_boot, 0) @ o)
    return f.sum() / o.sum(), S.ci(db), len(pids)


def cap(g, x, A, B, n_boot=2000):
    sa, sb, sf, fx = (sums(g, u, v) for u, v in ((x, A), (x, B), (x, x), (A, B)))
    pids = fx.index.values
    sa, sb, sf, fx = (z.loc[pids].values for z in (sa, sb, sf, fx))
    dr = S.problem_draws(len(pids), n_boot, 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        b = (dr @ sa) * (dr @ sb) / ((dr @ sf) * (dr @ fx))
    return sa.sum() * sb.sum() / (sf.sum() * fx.sum()), S.ci(b)


def tfidf_y(texts, y, g):
    out = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(texts, y, g):
        v = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        Xtr = v.fit_transform([texts[i] for i in tr])
        m = LogisticRegression(C=1.0, max_iter=2000).fit(Xtr, y[tr])
        out[te] = m.decision_function(v.transform([texts[i] for i in te]))
    return out


def hgb_r(X, r, g):
    Xc = X - pd.DataFrame(X).groupby(g).transform("mean").values
    rc = r - pd.Series(r).groupby(g).transform("mean").values
    out = np.zeros(len(r))
    for tr, te in GroupKFold(5).split(Xc, rc, g):
        sc = StandardScaler().fit(Xc[tr])
        pca = PCA(50, random_state=0).fit(sc.transform(Xc[tr]))
        m = HistGradientBoostingRegressor(max_depth=3, max_iter=150, learning_rate=0.05, min_samples_leaf=20,
                                          random_state=0).fit(pca.transform(sc.transform(Xc[tr])), rc[tr])
        out[te] = m.predict(pca.transform(sc.transform(Xc[te])))
    return out


def main():
    ap = argparse.ArgumentParser()
    for k in ("rollouts", "traces", "meta", "acts", "out"):
        ap.add_argument(f"--{k}", required=True)
    a = ap.parse_args()
    t = load(a.rollouts)
    tr = pd.DataFrame(json.loads(line) for line in open(a.traces)).set_index("trace_id")
    meta = pd.read_parquet(a.meta, columns=["trace_id", "problem_id", "pos", "prefix", "row"])
    acts = np.load(a.acts, mmap_mode="r")
    rows = []
    for cut in (2048, 4096):
        tc = t[t.cut == cut].reset_index(drop=True)
        tc = tc[tc.groupby("problem_id").trace_id.transform("size") >= 2].reset_index(drop=True)
        ok = np.stack(tc.ok.values).astype(float)
        Ab, Bb = ok[:, :4].mean(1), ok[:, 4:].mean(1)
        Aa, Ba = ok[:, 0::2].mean(1), ok[:, 1::2].mean(1)
        for name, A, B in (("blocks", Ab, Bb), ("alternating", Aa, Ba)):
            D, ci, P = d_share(tc, A, B)
            rows.append({"cut": cut, "what": f"D, {name} halves", "est": D, "lo": ci[0], "hi": ci[1], "problems": P})
        lv = tr.loc[tc.trace_id, "level"].values
        for L in (3, 4, 5):
            m = lv == L
            sub = tc[m].reset_index(drop=True)
            sub = sub[sub.groupby("problem_id").trace_id.transform("size") >= 2].reset_index(drop=True)
            o = np.stack(sub.ok.values).astype(float)
            D, ci, P = d_share(sub, o[:, :4].mean(1), o[:, 4:].mean(1))
            rows.append({"cut": cut, "what": f"D, level {L}", "est": D, "lo": ci[0], "hi": ci[1], "problems": P})
        mt = meta.set_index(["trace_id", "pos"])
        pre = np.array([("</think>" not in mt.at[(i, cut), "prefix"]) and ("\\boxed" not in mt.at[(i, cut), "prefix"])
                        for i in tc.trace_id])
        sub = tc[pre].reset_index(drop=True)
        sub = sub[sub.groupby("problem_id").trace_id.transform("size") >= 2].reset_index(drop=True)
        o = np.stack(sub.ok.values).astype(float)
        D, ci, P = d_share(sub, o[:, :4].mean(1), o[:, 4:].mean(1))
        rows.append({"cut": cut, "what": "D, pre-answer traces only", "est": D, "lo": ci[0], "hi": ci[1], "problems": P})
        # stronger readouts, captures scored on the block halves
        g = tc.problem_id.values
        allr = meta[(meta.pos == cut) & meta.trace_id.isin(set(tr.index))]
        y = (~tr.loc[allr.trace_id, "is_correct"].astype(bool)).astype(int).values
        f = {}
        f["tfidf_y"] = pd.Series(tfidf_y(list(allr.prefix), y, allr.problem_id.values), index=allr.trace_id.values)
        X4 = acts[allr.row.values][:, SLOTS4].astype(np.float32).reshape(len(allr), -1)
        f["probe4_y"] = pd.Series(probe_y(X4, y, allr.problem_id.values), index=allr.trace_id.values)
        rows_c = mt.loc[[(i, cut) for i in tc.trace_id], "row"].values
        Xc4 = acts[rows_c][:, SLOTS4].astype(np.float32).reshape(len(tc), -1)
        r = ok.mean(1)
        f["probe4_r"] = pd.Series(probe_r(Xc4, r, g), index=tc.trace_id.values)
        f["hgb_r"] = pd.Series(hgb_r(acts[rows_c, 6].astype(np.float32), r, g), index=tc.trace_id.values)
        for name, s in f.items():
            c, ci = cap(g, s.loc[tc.trace_id].values, Ab, Bb)
            rows.append({"cut": cut, "what": f"capture {name}", "est": c, "lo": ci[0], "hi": ci[1], "problems": len(set(g))})
            print(cut, name, round(c, 3), np.round(ci, 3))
    res = pd.DataFrame(rows)
    os.makedirs(a.out, exist_ok=True)
    res.to_csv(os.path.join(a.out, "robustness.csv"), index=False)
    print(res.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
