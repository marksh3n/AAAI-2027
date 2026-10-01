# fold-local features and models. every transform (scaler, SVD, top-K, tf-idf vocabulary and
# idf) is fitted on the training rows it is handed and only applied to the test rows.
import warnings
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from stats import Pairs

CS = [1e-3, 1e-2, 1e-1, 1.0, 10.0]
N_INNER = 3
SVD_K = 50
TOP_K = 50
PERM_C = 0.1


def standardize(tr, te):
    sc = StandardScaler().fit(tr)
    return sc.transform(tr), sc.transform(te)


class Hidden:
    """raw / svd50 / topk50 views of one hidden block. label-free fits are cached per
    training set, so a permutation null reuses the SVD instead of refitting it."""

    def __init__(self, X, groups, extra=None, seed=0):
        self.X, self.groups, self.extra, self.seed = X, np.asarray(groups), extra, seed
        self._svd = {}

    def svd(self, tr, te):
        key = (tr.tobytes(), te.tobytes())
        if key not in self._svd:
            ztr, zte = standardize(self.X[tr], self.X[te])
            red = TruncatedSVD(SVD_K, random_state=self.seed).fit(ztr)
            self._svd[key] = standardize(red.transform(ztr), red.transform(zte))
        return self._svd[key]

    def __call__(self, kind, tr, te, y):
        if kind == "raw":
            return standardize(self.X[tr], self.X[te])
        if kind == "svd50":
            return self.svd(tr, te)
        if kind == "topk50":
            # ranked by within-problem |AUROC - 0.5| on training rows only
            auc = Pairs(y[tr], self.groups[tr]).column_auroc(self.X[tr])
            keep = np.sort(np.argsort(np.abs(auc - 0.5), kind="stable")[-TOP_K:])
            return standardize(self.X[tr][:, keep], self.X[te][:, keep])
        if kind == "svd50+len":
            a, b = self.svd(tr, te)
            e, f = standardize(self.extra[tr], self.extra[te])
            return np.hstack([a, e]), np.hstack([b, f])
        raise ValueError(kind)


class Plain:
    """a small dense block (the len features), standardized on the training rows."""

    def __init__(self, E):
        self.E = E

    def __call__(self, kind, tr, te, y):
        return standardize(self.E[tr], self.E[te])


class Tfidf:
    """TfidfVectorizer(word 1-2 grams, min_df=2, sublinear_tf) fitted on the training rows.
    documents are tokenized once; each fold keeps the n-grams in >= 2 of its training docs and
    learns idf from them, which is exactly what fitting the vectorizer on those docs does."""

    def __init__(self, texts):
        self.counts = CountVectorizer(ngram_range=(1, 2)).fit_transform(texts).tocsr()

    def __call__(self, kind, tr, te, y):
        ctr, cte = self.counts[tr], self.counts[te]
        keep = np.flatnonzero(np.asarray((ctr > 0).sum(0)).ravel() >= 2)
        tf = TfidfTransformer(sublinear_tf=True).fit(ctr[:, keep])
        return tf.transform(ctr[:, keep]), tf.transform(cte[:, keep])


def make_model(name, C=None, seed=0):
    if name == "lr":
        return LogisticRegression(C=C, max_iter=5000)
    if name == "hgb":
        # early stopping would carve a random, ungrouped validation split: problems leak
        return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=150,
                                              min_samples_leaf=20, l2_regularization=1.0,
                                              early_stopping=False, random_state=seed)
    if name == "mlp":
        return MLPClassifier(hidden_layer_sizes=(64,), alpha=1e-2, max_iter=500,
                             early_stopping=False, random_state=seed)
    raise ValueError(name)


def p_wrong(model, Ftr, ytr, Fte):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(Ftr, ytr)
    return model.predict_proba(Fte)[:, list(model.classes_).index(1)]


def inner_splits(tr, y, groups, seed):
    cv = StratifiedGroupKFold(N_INNER, shuffle=True, random_state=seed)
    return [(tr[a], tr[b]) for a, b in cv.split(tr, y[tr], groups[tr])]


def pick_C(build, kind, tr, y, groups, seed):
    # grouped inner CV inside the training fold, scored by within-problem AUROC. the C path
    # is warm-started low to high. ties go to the smaller C.
    oof = {C: np.full(len(y), np.nan) for C in CS}
    for itr, ite in inner_splits(tr, y, groups, seed):
        Ftr, Fte = build(kind, itr, ite, y)
        lr = LogisticRegression(max_iter=5000, warm_start=True)
        for C in CS:
            oof[C][ite] = p_wrong(lr.set_params(C=C), Ftr, y[itr], Fte)
    pairs = Pairs(y[tr], groups[tr])
    score = {C: pairs.auroc(oof[C][tr])["weighted"] for C in CS}
    best = max(score.values())
    return min(C for C in CS if score[C] >= best - 1e-12)


def oof(build, kind, model, y, groups, folds, seed, C=None):
    # out-of-fold P(wrong); folds is a list of (train idx, test idx) over the population
    p = np.full(len(y), np.nan)
    chosen = []
    for tr, te in folds:
        c = C if model != "lr" or C is not None else pick_C(build, kind, tr, y, groups, seed)
        Ftr, Fte = build(kind, tr, te, y)
        p[te] = p_wrong(make_model(model, c, seed), Ftr, y[tr], Fte)
        chosen.append(c)
    return p, chosen


