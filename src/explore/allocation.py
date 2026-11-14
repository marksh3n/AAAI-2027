# X4: compute allocation at equal generated tokens, from source traces only (no rollouts, so no sampler gap).
# each problem's 6 traces are its sample pool; a policy draws them in a random order (200 orders per problem)
# and pays every token it generates, including aborted ones. policies:
#   maj@k          k samples, plurality vote over answer keys, ties split
#   short-m@k      k in parallel, stop when m have finished, plurality among those m (m = 1, 3)
#   restart(T, R)  abort a sample still running at T tokens and draw the next, at most R restarts; the last runs out
#   probe(q)       restart once at 4096 if the L20 probe (grouped out-of-fold) puts a live trace in the top q
#   random(q)      restart once at 4096 a random share q of live traces (exact mixture of base and restart(4096, 1))
# every policy is compared with the maj@k frontier at its own token cost: maj@k and maj@(k+1) mixed at the rate
# that matches tokens (linear interpolation), inside each problem-bootstrap draw.
#   python src/explore/allocation.py --traces <traces_B.jsonl> --meta <meta_B.parquet> --acts <acts_B.npy> \
#       --split results/split_all.json --half explore --out <dir>
import argparse, json, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "probe_cmp"))
import stats as S  # noqa: E402
from ceiling import norm  # noqa: E402
from determination import LAYER_SLOT, probe_y  # noqa: E402

CAP = 32768
N_PERM = 200
TS = (2048, 4096, 6144, 8192, 12288, 16384)
RS = (1, 2, 5)
QS = (0.25, 0.5)


def pools(tr):
    # per problem: lengths, correctness, answer keys ("C" for every correct answer; wrong answers by value;
    # no answer = None, never votes)
    out = {}
    for p, d in tr.groupby("problem_id"):
        ok = (d.is_correct.astype(bool) & ~d.trunc.astype(bool)).values
        keys = ["C" if o else ("W" + norm(a) if a else None) for o, a in zip(ok, d.answer)]
        out[p] = {"n": np.minimum(d.n_gen.values, CAP).astype(float), "ok": ok.astype(float), "key": keys,
                  "fin": (~d.trunc.astype(bool)).values, "trace_id": d.index.values}
    return out


def vote(keys):
    k = [x for x in keys if x is not None]
    if not k:
        return 0.0
    v, c = np.unique(k, return_counts=True)
    top = v[c == c.max()]
    return float(np.mean([t == "C" for t in top]))


def short(P, order, k, m):
    idx = order[:k]
    n, fin = P["n"][idx], P["fin"][idx]
    done = np.argsort(np.where(fin, n, np.inf), kind="stable")[: m]
    done = done[fin[done]]
    if not len(done):
        return 0.0, n.sum()
    stop = n[done].max()
    return vote([P["key"][idx[j]] for j in done]), np.minimum(n, stop).sum()


def restart(P, order, T, R, flag=None):
    tok, used = 0.0, 0
    for pos, j in enumerate(order):
        last = used == R or pos == len(order) - 1
        abort = P["n"][j] > T and not last and (flag is None or (pos == 0 and flag[j]))
        if not abort:
            return P["ok"][j], tok + P["n"][j]
        tok += T
        used += 1
    return P["ok"][order[-1]], tok


def evaluate(pool, flags, seed=0):
    # per problem, expected accuracy and tokens of every policy over N_PERM random orders
    rng = np.random.default_rng(seed)
    rows = []
    for p, P in pool.items():
        n = len(P["n"])
        orders = [rng.permutation(n) for _ in range(N_PERM)]
        acc = {}
        for k in range(1, min(n, 5) + 1):
            acc[f"maj@{k}"] = [(vote([P["key"][j] for j in o[:k]]), P["n"][o[:k]].sum()) for o in orders]
        for m in (1, 3):
            for k in range(max(2, m), n + 1):
                acc[f"short-{m}@{k}"] = [short(P, o, k, m) for o in orders]
        for T in TS:
            for R in RS:
                acc[f"restart(T{T},R{R})"] = [restart(P, o, T, R) for o in orders]
        for q, fl in flags.items():
            f = np.array([fl.get(t, False) for t in P["trace_id"]])
            acc[f"probe(q{q})"] = [restart(P, o, 4096, 1, f) for o in orders]
        for name, v in acc.items():
            a = np.array(v)
            rows.append({"problem_id": p, "policy": name, "acc": a[:, 0].mean(), "tokens": a[:, 1].mean()})
        base, rs = acc["maj@1"], acc["restart(T4096,R1)"]
        for q in QS:  # random restart of a share q of live traces: exact mixture
            a0, a1 = np.array(base).mean(0), np.array(rs).mean(0)
            rows.append({"problem_id": p, "policy": f"random(q{q})", "acc": (1 - q) * a0[0] + q * a1[0],
                         "tokens": (1 - q) * a0[1] + q * a1[1]})
    return pd.DataFrame(rows)


def frontier(acc_k, tok_k, t):
    # maj@k frontier accuracy at token cost t (linear between neighbouring k); nan outside its range
    return np.interp(t, tok_k, acc_k, left=np.nan, right=np.nan)


def compare(per, n_boot=2000, seed=0):
    pids = np.sort(per.problem_id.unique())
    A = per.pivot(index="problem_id", columns="policy", values="acc").loc[pids]
    T = per.pivot(index="problem_id", columns="policy", values="tokens").loc[pids]
    draws = S.problem_draws(len(pids), n_boot, seed)
    w = np.vstack([np.ones(len(pids)), draws])  # row 0 = the estimate
    Am, Tm = (w @ A.values) / w.sum(1, keepdims=True), (w @ T.values) / w.sum(1, keepdims=True)
    col = {c: i for i, c in enumerate(A.columns)}
    majs = [f"maj@{k}" for k in range(1, 6) if f"maj@{k}" in col]
    rows = []
    for c in A.columns:
        d = np.array([Am[b, col[c]] - frontier(Am[b, [col[m] for m in majs]], Tm[b, [col[m] for m in majs]], Tm[b, col[c]])
                      for b in range(len(w))])
        lo, hi = S.ci(d[1:]) if np.isfinite(d[1:]).mean() > 0.9 else (np.nan, np.nan)
        rows.append({"policy": c, "acc": Am[0, col[c]], "tokens": Tm[0, col[c]], "vs_maj_frontier": d[0],
                     "lo": lo, "hi": hi, "problems": len(pids)})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    for k in ("traces", "meta", "acts", "split", "out"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--half", choices=["explore", "holdout"], required=True)
    a = ap.parse_args()
    tr = pd.DataFrame(json.loads(line) for line in open(a.traces)).set_index("trace_id", drop=False)
    keep = set(json.load(open(a.split))[a.half])
    # probe scores: grouped out-of-fold over ALL dev traces live at 4096 (the probe never sees the half's labels
    # for its own fold); thresholds = quantiles of this half's live scores
    meta = pd.read_parquet(a.meta, columns=["trace_id", "problem_id", "pos", "row"])
    rows = meta[(meta.pos == 4096)]
    acts = np.load(a.acts, mmap_mode="r")
    X = acts[rows.row.values, LAYER_SLOT].astype(np.float32)
    y = (~tr.loc[rows.trace_id, "is_correct"].astype(bool)).astype(int).values
    sc = pd.Series(probe_y(X, y, rows.problem_id.values), index=rows.trace_id.values)
    live = sc[[t in keep for t in rows.problem_id.values]]
    flags = {q: (live > live.quantile(1 - q)).to_dict() for q in QS}
    pool = pools(tr[tr.problem_id.isin(keep)])
    per = evaluate(pool, flags)
    res = compare(per)
    res["half"] = a.half
    os.makedirs(a.out, exist_ok=True)
    per.to_csv(os.path.join(a.out, f"x4_per_problem_{a.half}.csv"), index=False)
    res.to_csv(os.path.join(a.out, f"x4_{a.half}.csv"), index=False)
    print(res.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
