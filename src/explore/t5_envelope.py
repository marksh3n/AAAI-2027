# T5 "vs best no-monitor baseline": upper envelope of the maj@k and short-1@k frontiers at equal tokens, per bootstrap
# draw; paired probe-minus-random interval; hindsight-oracle row (restart exactly the live traces that end wrong).
#   python src/explore/t5_envelope.py <dir with x4_per_problem_*.csv> <traces_B.jsonl> <results/split_all.json> <out dir>
import json, os, sys
import numpy as np, pandas as pd
X4, TRACES, SPLIT, OUT = sys.argv[1:5]
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(os.path.dirname(HERE), "probe_cmp")]
import allocation as AL, stats as S  # noqa: E402

tr = pd.DataFrame(json.loads(l) for l in open(TRACES)).set_index("trace_id", drop=False)
split = json.load(open(SPLIT))
os.makedirs(OUT, exist_ok=True)
SHOW = ["probe(q0.25)", "random(q0.25)", "probe(q0.5)", "random(q0.5)", "restart(T4096,R1)", "restart(T4096,R2)",
        "restart(T8192,R1)", "restart(T8192,R2)", "restart(T8192,R5)", "short-1@2", "oracle", "random(q_oracle)"]
for half in ("holdout", "explore"):
    per = pd.read_csv(f"{X4}/x4_per_problem_{half}.csv")
    sub = tr[tr.problem_id.isin(set(split[half]))]
    wrong = ~(sub.is_correct.astype(bool) & ~sub.trunc.astype(bool))
    live = sub.n_gen > 4096
    q = (wrong & live).sum() / live.sum()
    new = AL.evaluate(AL.pools(sub), {"oracle": (wrong & live).to_dict()})
    chk = new[new.policy == "restart(T4096,R1)"].merge(per[per.policy == "restart(T4096,R1)"], on="problem_id")
    assert np.allclose(chk.acc_x, chk.acc_y) and np.allclose(chk.tokens_x, chk.tokens_y), "orders differ from x4"
    o = new[new.policy == "probe(qoracle)"].assign(policy="oracle")
    b, r = (per[per.policy == p].set_index("problem_id") for p in ("maj@1", "restart(T4096,R1)"))
    rq = ((1 - q) * b[["acc", "tokens"]] + q * r[["acc", "tokens"]]).reset_index().assign(policy="random(q_oracle)")
    per = pd.concat([per, o, rq], ignore_index=True)
    pids = np.sort(per.problem_id.unique())
    A = per.pivot(index="problem_id", columns="policy", values="acc").loc[pids]
    T = per.pivot(index="problem_id", columns="policy", values="tokens").loc[pids]
    w = np.vstack([np.ones(len(pids)), S.problem_draws(len(pids), 2000, 0)])
    Am, Tm = (w @ A.values) / w.sum(1, keepdims=True), (w @ T.values) / w.sum(1, keepdims=True)
    col = {c: i for i, c in enumerate(A.columns)}
    maj = [col[f"maj@{k}"] for k in range(1, 6)]
    sh = [col["maj@1"]] + [col[f"short-1@{k}"] for k in range(2, 7)]

    def env(bi, t):
        return np.nanmax([np.interp(t, Tm[bi, m], Am[bi, m], left=np.nan, right=np.nan) for m in (maj, sh)])

    rows = []
    for c in SHOW:
        d = np.array([Am[bi, col[c]] - env(bi, Tm[bi, col[c]]) for bi in range(len(w))])
        lo, hi = S.ci(d[1:])
        rows.append({"half": half, "policy": c, "acc": Am[0, col[c]], "tokens": Tm[0, col[c]],
                     "vs_envelope_pp": 100 * d[0], "lo": 100 * lo, "hi": 100 * hi})
    for a, bname in (("probe(q0.25)", "random(q0.25)"), ("probe(q0.5)", "random(q0.5)"), ("oracle", "random(q_oracle)")):
        d = Am[:, col[a]] - Am[:, col[bname]]
        dt = Tm[:, col[a]] - Tm[:, col[bname]]
        lo, hi = S.ci(d[1:])
        rows.append({"half": half, "policy": f"{a} minus {bname}", "acc": np.nan, "tokens": dt[0],
                     "vs_envelope_pp": 100 * d[0], "lo": 100 * lo, "hi": 100 * hi})
    res = pd.DataFrame(rows)
    res.to_csv(f"{OUT}/t5_envelope_{half}.csv", index=False)
    print(f"\n{half}: {len(pids)} problems; live at 4096: {live.sum()} traces, {100*q:.1f}% end wrong")
    print(res.round(3).to_string(index=False))
