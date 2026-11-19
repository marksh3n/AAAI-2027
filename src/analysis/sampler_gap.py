# sampler gap under the cohort rule, three models. null: each source outcome is one Bernoulli(r_i) draw; a problem is
# kept iff its cohort sources are mixed, so expected source accuracy of a kept problem is
# (sum r - n prod r) / (n (1 - prod r - prod (1-r))). also a null simulation of D under the same rule.
#   python src/analysis/sampler_gap.py <deepseek rollouts> <traces_B.jsonl> <qwen outputs dir> <ministral outputs dir>
import json, os, sys
import numpy as np, pandas as pd
DSB, DST, QO, MO = sys.argv[1:5]
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(os.path.dirname(HERE), "probe_cmp")]
import cheap_rollouts as CR  # noqa: E402
rng = np.random.default_rng(0)


def pred(r):
    n, pc, pw = len(r), np.prod(r), np.prod(1 - r)
    return (r.sum() - n * pc) / (n * (1 - pc - pw))


def gap(name, t, y):
    g = t.problem_id.values
    ok = np.stack(t.ok.values).astype(float)
    k, w = ok.sum(1), ok.shape[1]
    for sm in (0.5, 1.0):
        r = (k + sm) / (w + 2 * sm)
        a = np.array([(len(i), pred(r[i]), y[i].mean(), (k[i] / w).mean())
                      for i in (np.where(g == p)[0] for p in np.unique(g))])

        def stat(j):
            n = a[j, 0]
            m = lambda c: (n * a[j, c]).sum() / n.sum()
            return m(3) - m(2), m(2) - m(1)
        est = stat(np.arange(len(a)))
        B = np.array([stat(rng.integers(0, len(a), len(a))) for _ in range(2000)])
        ci = lambda c: (100 * np.percentile(B[:, c], 2.5), 100 * np.percentile(B[:, c], 97.5))
        print(f"{name} smooth {sm}: {len(a)} problems, {len(t)} cuts | raw r_hat - source {100*est[0]:+.1f} "
              f"[{ci(0)[0]:+.1f}, {ci(0)[1]:+.1f}] | source - null prediction {100*est[1]:+.1f} [{ci(1)[0]:+.1f}, {ci(1)[1]:+.1f}]")


def null_D(name, t, reps=300):
    # no prefix effect within a problem: every cut of a problem shares r = its mean r_hat. draw sources and 8 re-runs,
    # apply the cohort rule (mixed sources), compute the split-half D. should centre on 0.
    g = t.problem_id.values
    ok = np.stack(t.ok.values).astype(float)
    rp = pd.Series(ok.mean(1)).groupby(g).transform("mean").values
    ds = []
    for _ in range(reps):
        src = rng.random(len(g)) < rp
        nun = pd.Series(src).groupby(g).transform("nunique").values
        keep = nun == 2
        sim = rng.random(ok.shape) < rp[:, None]
        A, B = sim[:, 0::2].mean(1), sim[:, 1::2].mean(1)
        num = den = 0.0
        for p in np.unique(g[keep]):
            i = np.where((g == p) & keep)[0]
            if len(i) < 2:
                continue
            num += ((A[i] - A[i].mean()) * (B[i] - B[i].mean())).sum() / (len(i) - 1) * len(i)
            m = sim[i].mean()
            den += m * (1 - m) * len(i)
        ds.append(num / den)
    ds = np.array(ds)
    print(f"{name} null D under the cohort rule: mean {ds.mean():+.3f}, 95% range [{np.percentile(ds,2.5):+.3f}, {np.percentile(ds,97.5):+.3f}]")


t = CR.load(DSB)
tr = pd.DataFrame(json.loads(l) for l in open(DST)).set_index("trace_id")
for cut in (512, 2048, 4096):
    tc = t[t.cut == cut].reset_index(drop=True)
    gap(f"deepseek {cut}", tc, tr.loc[tc.trace_id, "is_correct"].astype(float).values)
null_D("deepseek 4096", t[t.cut == 4096].reset_index(drop=True))
for name, O in (("qwen3 4096", QO), ("ministral 4096 (limit hits wrong)", MO)):
    rows = [{"trace_id": f"{r['problem_id']}_{s['sample_idx']}", "ok": bool(s["correct"])}
            for r in map(json.loads, open(f"{O}/sources.jsonl")) for s in r["samples"]]
    src = pd.DataFrame(rows).set_index("trace_id").ok
    tc = CR.load(f"{O}/bank.jsonl")
    tc["problem_id"] = tc.problem_id.astype(str)
    y = src.loc[tc.trace_id].astype(float).values
    both = pd.Series(y).groupby(tc.problem_id.values).transform("nunique").values == 2
    tc, y = tc[both].reset_index(drop=True), y[both]
    gap(name, tc, y)
    null_D(name, tc)
