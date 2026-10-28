# power of the planning design's two paired tests (one binary outcome per arm per problem), written independently
# of his package and checked against his reported numbers (tests/test_confirm.py):
#   detect   exact two-sided McNemar at alpha: reject when 2 * min binomial tail of b | b + c <= alpha
#   rule out Tango's paired score test of H0: delta >= margin, one-sided alpha, asymptotic normal
#            reference (restricted MLE of the discordance at delta0, Tango 1998 eq. 24-26)
# both enumerate M ~ Bin(n, q) discordant pairs, B | M ~ Bin(M, (q + d) / 2q): exact rejection
# probabilities of the stated rules under (q, d), not exact-level tests.
#   python src/confirm/power.py --n 1000 --q 0.25 0.07 --d 0.05 0.016
import argparse
import numpy as np
from scipy.stats import binom, norm


def mcnemar_power(n, q, d, alpha=0.05):
    if q <= 0:
        return 0.0
    theta = (q + d) / (2 * q)
    m = np.arange(n + 1)
    wm = binom.pmf(m, n, q)
    power = 0.0
    for mi, w in zip(m, wm):
        if w < 1e-15 or mi == 0:
            continue
        b = np.arange(mi + 1)
        tail = np.minimum(binom.sf(b - 1, mi, 0.5), binom.cdf(b, mi, 0.5))  # P(B >= b), P(B <= b) under H0
        rej = np.minimum(1.0, 2 * tail) <= alpha
        power += w * binom.pmf(b, mi, theta)[rej].sum()
    return float(power)


def tango_z(n, b, c, delta0):
    s, dd = b + c, b - c
    a = s + dd * delta0
    disc = a * a - 4 * n * (dd * delta0 - (n - s) * delta0 ** 2)
    qr = np.clip((a + np.sqrt(np.maximum(0.0, disc))) / (2 * n), abs(delta0), 1.0)
    var = n * np.maximum(0.0, qr - delta0 ** 2)
    num = dd - n * delta0
    with np.errstate(divide="ignore", invalid="ignore"):
        z = num / np.sqrt(var)
    return np.where(var > 0, z, np.where(num == 0, 0.0, np.copysign(np.inf, num)))


def ruleout_power(n, q, d, margin=0.05, alpha=0.05):
    # P(reject H0: delta >= margin) when the true difference is d
    if q <= 0:
        return float(tango_z(n, np.array([0]), np.array([0]), margin)[0] < norm.ppf(alpha))
    theta = (q + d) / (2 * q)
    crit = norm.ppf(alpha)
    power = 0.0
    for mi, w in zip(range(n + 1), binom.pmf(np.arange(n + 1), n, q)):
        if w < 1e-15:
            continue
        b = np.arange(mi + 1)
        rej = tango_z(n, b, mi - b, margin) < crit
        power += w * binom.pmf(b, mi, theta)[rej].sum()
    return float(power)


def table(n, qs, ds, margin=0.05, alpha=0.05):
    rows = []
    for q in qs:
        for d in ds:
            if abs(d) > q:
                continue
            rows.append({"n": n, "q": q, "d": d, "mcnemar_detect": mcnemar_power(n, q, d, alpha),
                         "tango_rule_out": ruleout_power(n, q, d, margin, alpha)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, nargs="+", default=[1000])
    ap.add_argument("--q", type=float, nargs="+", required=True)
    ap.add_argument("--d", type=float, nargs="+", required=True)
    ap.add_argument("--margin", type=float, default=0.05)
    a = ap.parse_args()
    print("n      q      d      McNemar detect (2-sided .05)   Tango rule out d >= margin (1-sided .05)")
    for n in a.n:
        for r in table(n, a.q, a.d, a.margin):
            print(f"{r['n']:<6} {r['q']:<6.3f} {r['d']:<+6.3f} {r['mcnemar_detect']:>10.3f}   {r['tango_rule_out']:>30.3f}")


if __name__ == "__main__":
    main()
