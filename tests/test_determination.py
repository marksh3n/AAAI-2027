# determination.py estimators on simulated rollouts with a known r per prefix. plain script, CPU only.
import os, sys
import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src", "analysis"))
sys.path.insert(0, os.path.join(ROOT, "src", "probe_cmp"))
import determination as X  # noqa: E402

fails = []


def check(name, cond, extra=None):
    print(f"{'ok  ' if cond else 'FAIL'} {name}{(' -- ' + str(extra)) if extra is not None else ''}")
    if not cond:
        fails.append(name)


rng = np.random.default_rng(0)
P, n = 400, 6
g = np.repeat(np.arange(P), n)
base = rng.uniform(0.2, 0.8, P)[g]
signal = rng.normal(0, 1, len(g))
r = np.clip(base + 0.15 * signal, 0.01, 0.99)
ok = rng.random((len(g), 8)) < r[:, None]
t = pd.DataFrame({"problem_id": g, "trace_id": [f"t{i}" for i in range(len(g))]})
t["ok"] = list(ok)
A, B, _ = X.halves(t)
fix, tot = X.sums(g, A, B), X.total_sums(t)
# truth: within-problem var of r over within-problem var of one outcome
rc = r - pd.Series(r).groupby(g).transform("mean").values
true_fix = (rc ** 2).sum() / (len(g) - P)
q = pd.Series(r).groupby(g).transform("mean").values
true_tot = (true_fix * (len(g) - P) + (r * (1 - r)).sum() * (n - 1) / n) / (len(g) - P)
D = fix.sum() / tot.sum()
check("determined share recovers the simulated one", abs(D - true_fix / true_tot) < 0.03, (round(D, 3), round(true_fix / true_tot, 3)))
cap = lambda x: X.sums(g, x, A).sum() * X.sums(g, x, B).sum() / (X.sums(g, x, x).sum() * fix.sum())
check("capture of the true r is about 1", abs(cap(r) - 1) < 0.1, round(cap(r), 3))
check("capture of noise is about 0", abs(cap(rng.normal(size=len(g)))) < 0.03, round(cap(rng.normal(size=len(g))), 3))
half = signal + rng.normal(0, 1, len(g))
check("capture of signal + equal noise is about 1/2", abs(cap(half) - 0.5) < 0.1, round(cap(half), 3))
check("capture ignores a predictor's per-problem offset", np.isclose(cap(r + 5 * base), cap(r)))

print(f"\n{'all passed' if not fails else f'{len(fails)} FAILED: {fails}'}")
sys.exit(1 if fails else 0)
