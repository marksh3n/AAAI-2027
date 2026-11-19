# cheap_rollouts.py scoring on hand-made rows. plain script, CPU only, exits non-zero on failure.
import os, sys
import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src", "analysis"))
sys.path.insert(0, os.path.join(ROOT, "src", "probe_cmp"))
import cheap_rollouts as X  # noqa: E402

fails = []


def check(name, cond, extra=None):
    print(f"{'ok  ' if cond else 'FAIL'} {name}{(' -- ' + str(extra)) if extra is not None else ''}")
    if not cond:
        fails.append(name)


row = lambda n, ok, fin, cls: {"n": np.array(n), "ok": np.array(ok), "fin": np.array(fin), "cls": np.array(cls)}
t = pd.DataFrame([row([100, 3000, 100, 100, 100, 100, 100, 100], [1, 1, 0, 0, 0, 0, 0, 0], [1] * 8, [0, 0, 1, 1, 1, 2, 2, -1]),
                  row([5000] * 8, [1] * 8, [0] * 8, [-1] * 8)]).astype(object)
for c in ("ok", "fin"):
    t[c] = t[c].map(lambda a: a.astype(bool))
all8 = np.tile(np.arange(8), (2, 1))
check("correct: share of all 8 correct", np.allclose(X.score(t, all8, 32768, "correct"), [1 - 2 / 8, 0]))
check("correct: a cap drops continuations longer than L", np.allclose(X.score(t, all8, 1000, "correct"), [1 - 1 / 8, 1]))
check("finished: unfinished never count", np.allclose(X.score(t, all8, 32768, "finished"), [0, 1]))
check("agree: modal class among finished within L", np.allclose(X.score(t, all8, 32768, "agree"), [1 - 3 / 8, 1]))
check("agree: the cap drops a long non-modal continuation, modal count unchanged", np.allclose(X.score(t, all8, 1000, "agree"), [1 - 3 / 8, 1]))
first2 = np.tile([0, 1], (2, 1))
check("subset of two", np.allclose(X.score(t, first2, 32768, "correct"), [0, 0]))
sub = X.subsets(5, 2, np.random.default_rng(0))
check("subsets: N_SUB draws of k distinct indices", sub.shape == (X.N_SUB, 5, 2) and (sub[..., 0] != sub[..., 1]).all())
check("subsets: K 8 is one draw", X.subsets(5, 8, np.random.default_rng(0)).shape == (1, 5, 8))

print(f"\n{'all passed' if not fails else f'{len(fails)} FAILED: {fails}'}")
sys.exit(1 if fails else 0)
