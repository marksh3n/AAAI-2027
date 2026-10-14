# metrics and resampling. y = 1 means the trace ended WRONG and every score is P(wrong), so
# within-problem AUROC = P(score_wrong > score_correct) over a problem's correct x wrong pairs.
import numpy as np
from scipy.stats import norm


class Pairs:
    """Every (wrong, correct) pair inside one problem, indexed once and reused for all scores."""

    def __init__(self, y, groups):
        y, groups = np.asarray(y), np.asarray(groups)
        self.pids, gi = np.unique(groups, return_inverse=True)
        w, c, p = [], [], []
        for k in range(len(self.pids)):
            rows = np.flatnonzero(gi == k)
            a, b = rows[y[rows] == 1], rows[y[rows] == 0]
            if len(a) and len(b):
                aa, bb = np.meshgrid(a, b, indexing="ij")
                w.append(aa.ravel()), c.append(bb.ravel()), p.append(np.full(aa.size, k))
        cat = lambda v: np.concatenate(v) if v else np.array([], int)
        self.wrong, self.correct, pk = cat(w), cat(c), cat(p)
        # keep only problems that have pairs, renumbered 0..P-1
        self.pids = self.pids[np.unique(pk)] if len(pk) else self.pids[:0]
        self.pair_problem = np.unique(pk, return_inverse=True)[1] if len(pk) else pk
        self.pairs = np.bincount(self.pair_problem, minlength=len(self.pids)).astype(float)

    def wins(self, s):
        # per-problem count of pairs the score orders correctly, ties worth 1/2
        s = np.asarray(s, float)
        d = s[self.wrong] - s[self.correct]
        return np.bincount(self.pair_problem, (d > 0) + 0.5 * (d == 0), minlength=len(self.pids))

    def auroc(self, s):
        w = self.wins(s)
        return {"weighted": w.sum() / self.pairs.sum(), "unweighted": float(np.mean(w / self.pairs)),
                "n_pairs": int(self.pairs.sum()), "n_problems": len(self.pids)}

    def column_auroc(self, X):
        # within-problem AUROC of every column at once (top-K ranking)
        d = X[self.wrong] - X[self.correct]
        return ((d > 0).sum(0) + 0.5 * (d == 0).sum(0)) / len(self.wrong)


def pooled_auroc(y, s):
    return float(weighted_pooled(y, s, np.ones((1, len(y))))[0])


def weighted_pooled(y, s, w):
    # pooled AUROC under row weights w (B, N): one sort, every bootstrap draw at once
    y, s = np.asarray(y), np.asarray(s, float)
    order = np.argsort(s, kind="mergesort")
    _, g = np.unique(s[order], return_inverse=True)
    w = np.asarray(w, float)[:, order]
    onehot = np.zeros((len(y), g.max() + 1))
    onehot[np.arange(len(y)), g] = 1.0
    pos = w * (y[order] == 1)
    neg_g = (w * (y[order] == 0)) @ onehot
    below = np.cumsum(neg_g, axis=1) - neg_g
    num = ((pos @ onehot) * (below + 0.5 * neg_g)).sum(1)
    return num / (pos.sum(1) * neg_g.sum(1))


def problem_draws(n_problems, n_boot=2000, seed=0):
    # (B, P) multiplicities: how often each problem is drawn. shared by every arm, so any
    # difference of two arms under the same draws is a paired bootstrap
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_problems, (n_boot, n_problems))
    return np.stack([np.bincount(r, minlength=n_problems) for r in idx]).astype(float)


def boot_within(pairs, s, draws):
    # (weighted, unweighted) within-problem AUROC under each draw
    w = pairs.wins(s)
    return draws @ w / (draws @ pairs.pairs), draws @ (w / pairs.pairs) / draws.sum(1)


def boot_pooled(y, s, groups, pids, draws):
    # rows inherit their problem's multiplicity; problems outside pids get weight 0
    col = {p: i for i, p in enumerate(pids)}
    idx = np.array([col.get(g, -1) for g in groups])
    w = np.where(idx >= 0, draws[:, np.maximum(idx, 0)], 0.0)
    return weighted_pooled(y, s, w)


def ci(vals):
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def boot_p(deltas):
    # two-sided bootstrap p for delta = 0, never below 2/(B+1); a guide next to the CI
    d = np.asarray(deltas)
    tail = min((d <= 0).sum(), (d >= 0).sum())
    return float(min(1.0, 2 * (1 + tail) / (len(d) + 1)))


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order])
    out = np.empty_like(p)
    out[order] = np.minimum(adj, 1.0)
    return out


def permute_within(y, groups, rng):
    # shuffle labels among traces of the same problem: each problem keeps its counts, so
    # difficulty structure stays in the null
    y, groups = np.asarray(y), np.asarray(groups)
    out = y.copy()
    for g in np.unique(groups):
        rows = np.flatnonzero(groups == g)
        out[rows] = rng.permutation(y[rows])
    return out


def _midrank(x):
    order = np.argsort(x, kind="mergesort")
    z = x[order]
    n = len(x)
    t = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j < n and z[j] == z[i]:
            j += 1
        t[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n)
    out[order] = t
    return out


def delong(y, s1, s2):
    # fast DeLong (Sun & Xu 2014) for two correlated AUROCs on the same rows; positives = y==1
    y = np.asarray(y).astype(bool)
    m, n = int(y.sum()), int((~y).sum())
    v10, v01, aucs = [], [], []
    for s in (np.asarray(s1, float), np.asarray(s2, float)):
        x, z = s[y], s[~y]
        tx, tz, tall = _midrank(x), _midrank(z), _midrank(np.concatenate([x, z]))
        aucs.append((tall[:m].sum() - m * (m + 1) / 2) / (m * n))
        v10.append((tall[:m] - tx) / n)  # per positive: share of negatives below
        v01.append(1 - (tall[m:] - tz) / m)  # per negative: share of positives above
    s = np.cov(np.array(v10)) / m + np.cov(np.array(v01)) / n
    var = s[0, 0] + s[1, 1] - 2 * s[0, 1]
    d = aucs[0] - aucs[1]
    se = float(np.sqrt(max(var, 0.0)))
    z = d / se if se > 0 else np.nan
    return {"auc1": float(aucs[0]), "auc2": float(aucs[1]), "delta": float(d), "se": se,
            "z": float(z), "p": float(2 * norm.sf(abs(z))) if se > 0 else np.nan,
            "ci": [float(d - 1.959964 * se), float(d + 1.959964 * se)], "n_pos": m, "n_neg": n}
