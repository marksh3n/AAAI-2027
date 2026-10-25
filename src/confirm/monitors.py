# the frozen monitors as fittable, saveable objects. the math is probe_cmp's (models.py) step for
# step; probe_cmp's builders return transformed arrays and drop the fitted scaler / top-K columns /
# tf-idf vocabulary, which a monitor applied to new traces needs, so the fitted pieces are kept here.
# tests/test_confirm.py checks that out-of-fold predictions from this file equal probe_cmp's.
import os, sys, warnings
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "probe_cmp"))
import models as M  # noqa: E402
from data import paragraphs  # noqa: E402
from stats import Pairs  # noqa: E402


def len_features(tokens, texts):
    # probe_cmp's observable length: log1p of tokens seen and \n\n paragraphs inside <think>
    para = np.array([paragraphs(x) for x in texts], float)
    return np.log1p(np.column_stack([np.asarray(tokens, float), para]))


class Monitor:
    """One (model, feature) monitor at one checkpoint. fit() on training rows, score() on any rows.
    inputs: X (rows, hidden_dim) for hidden features, texts for tfidf, (tokens, texts) for len."""

    def __init__(self, model, feature, C=None, seed=0):
        self.model, self.feature, self.C, self.seed = model, feature, C, seed

    # ---- features: fit on training rows, apply to any rows ----
    def _fit_features(self, X, texts, tokens, y, groups):
        f = self.feature
        if f == "raw":
            self.scaler = StandardScaler().fit(X)
        elif f == "topk50":
            auc = Pairs(y, groups).column_auroc(X)
            self.keep = np.sort(np.argsort(np.abs(auc - 0.5), kind="stable")[-M.TOP_K:])
            self.scaler = StandardScaler().fit(X[:, self.keep])
        elif f == "len":
            self.scaler = StandardScaler().fit(len_features(tokens, texts))
        elif f == "tfidf":
            cv = CountVectorizer(ngram_range=(1, 2)).fit(texts)
            counts = cv.transform(texts)
            keep = np.flatnonzero(np.asarray((counts > 0).sum(0)).ravel() >= 2)
            names = cv.get_feature_names_out()[keep]
            # a vocabulary of the kept n-grams only, in the column order probe_cmp used
            self.vec = CountVectorizer(ngram_range=(1, 2), vocabulary={n: i for i, n in enumerate(names)})
            self.tf = TfidfTransformer(sublinear_tf=True).fit(counts[:, keep])
        else:
            raise ValueError(f)

    def features(self, X=None, texts=None, tokens=None):
        f = self.feature
        if f == "raw":
            return self.scaler.transform(X)
        if f == "topk50":
            return self.scaler.transform(X[:, self.keep])
        if f == "len":
            return self.scaler.transform(len_features(tokens, texts))
        if f == "tfidf":
            return self.tf.transform(self.vec.transform(texts))
        raise ValueError(f)

    def fit(self, y, groups, X=None, texts=None, tokens=None):
        y, groups = np.asarray(y), np.asarray(groups)
        if self.model == "lr" and self.C is None:
            self.C = pick_C(self.feature, y, groups, X, texts, tokens, self.seed)
        self._fit_features(X, texts, tokens, y, groups)
        self.clf = M.make_model(self.model, self.C, self.seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            self.clf.fit(self.features(X, texts, tokens), y)
        return self

    def score(self, X=None, texts=None, tokens=None):
        # P(wrong); higher = flag first
        return self.clf.predict_proba(self.features(X, texts, tokens))[:, list(self.clf.classes_).index(1)]


def _take(v, idx):
    if v is None:
        return None
    return v[idx] if isinstance(v, np.ndarray) else [v[i] for i in idx]


def pick_C(feature, y, groups, X, texts, tokens, seed):
    # models.pick_C with this file's features: grouped inner CV on the rows given, scored by
    # within-problem AUROC, warm-started over CS, ties to the smaller C
    idx = np.arange(len(y))
    oof = {C: np.full(len(y), np.nan) for C in M.CS}
    for itr, ite in M.inner_splits(idx, y, groups, seed):
        m = Monitor("lr", feature, C=1.0, seed=seed)
        m._fit_features(_take(X, itr), _take(texts, itr), _take(tokens, itr), y[itr], groups[itr])
        Ftr = m.features(_take(X, itr), _take(texts, itr), _take(tokens, itr))
        Fte = m.features(_take(X, ite), _take(texts, ite), _take(tokens, ite))
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression(max_iter=5000, warm_start=True)
        for C in M.CS:
            oof[C][ite] = M.p_wrong(lr.set_params(C=C), Ftr, y[itr], Fte)
    pairs = Pairs(y, groups)
    score = {C: pairs.auroc(oof[C])["weighted"] for C in M.CS}
    best = max(score.values())
    return min(C for C in M.CS if score[C] >= best - 1e-12)


def oof_scores(model, feature, y, groups, folds, seed, X=None, texts=None, tokens=None):
    # out-of-fold P(wrong) over (train, test) folds, as models.oof: C picked inside each fold
    p = np.full(len(y), np.nan)
    Cs = []
    for tr, te in folds:
        m = Monitor(model, feature, seed=seed).fit(y[tr], groups[tr], _take(X, tr), _take(texts, tr), _take(tokens, tr))
        p[te] = m.score(_take(X, te), _take(texts, te), _take(tokens, te))
        Cs.append(m.C)
    return p, Cs


def threshold(scores, q):
    # flag a score strictly above this: the (1 - q) quantile of the dev scores, so a share q of the
    # dev live traces sits above it (ties at the threshold are not flagged)
    return float(np.quantile(np.asarray(scores, float), 1 - q, method="lower"))
