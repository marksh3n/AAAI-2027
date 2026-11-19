# one cohort rule for all three models. DeepSeek's cohort keeps a problem only if its re-run (cohort) traces include
# a correct and a wrong source; Qwen3 and Ministral kept a problem if its six sources were mixed. this reruns X1 (D and
# readout capture), forward (re-run capture) and X2 (re-run AUROC) on Qwen3 and Ministral restricted to the DeepSeek
# rule ("pairs" problems), next to all cohort problems and the complement ("same": cohort traces all one outcome).
#   python src/analysis/consistent_rule.py <qwen outputs dir> <ministral outputs dir> <out dir>
import json, os, re, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(os.path.dirname(HERE), "probe_cmp")]
import determination as DT  # noqa: E402
import forward_capture as FC  # noqa: E402
import cheap_rollouts as CR  # noqa: E402
import stats as S  # noqa: E402

RESTART = re.compile(r"\b(wait|hold on|let me re|actually)\b", re.I)
QO, MO, OUT = sys.argv[1:4]
os.makedirs(OUT, exist_ok=True)


def load(O, layer):
    rows = []
    for line in open(f"{O}/sources.jsonl"):
        r = json.loads(line)
        for s in r["samples"]:
            rows.append({"trace_id": f"{r['problem_id']}_{s['sample_idx']}", "problem_id": str(r["problem_id"]),
                         "is_correct": bool(s["correct"]), "n_gen": s["n_gen"], "truncated": s["truncated"]})
    tr = pd.DataFrame(rows).set_index("trace_id")
    f = pd.read_parquet(f"{O}/feats/feats.parquet")
    meta = pd.DataFrame({"trace_id": f.trace_id, "problem_id": f.problem_id.astype(str), "pos": f.checkpoint,
                         "prefix": f.prefix_text, "restarts": [len(RESTART.findall(x)) / c * 1000
                                                               for x, c in zip(f.prefix_text, f.checkpoint)],
                         "row": f.row})
    acts = np.load(f"{O}/feats/acts_L{layer}.npy")[:, None, :]
    t = CR.load(f"{O}/bank.jsonl")
    t["problem_id"] = t.problem_id.astype(str)
    return tr, meta, acts, t


DT.LAYER_SLOT = 0
summary = []
for model, O, layer in (("qwen3", QO, 20), ("ministral", MO, 19)):
    tr, meta, acts, t = load(O, layer)
    y = (~tr.is_correct.astype(bool)).astype(int)
    both = t.assign(c=tr.loc[t.trace_id, "is_correct"].values).groupby("problem_id").c.nunique()
    pairs = set(both.index[both == 2])
    for name, keep in (("all", set(both.index)), ("pairs", pairs), ("same", set(both.index) - pairs)):
        s = t[t.problem_id.isin(keep)].reset_index(drop=True)
        print(f"\n==== {model} {name}: {len(s)} prefixes, {s.problem_id.nunique()} problems", flush=True)
        row = {"model": model, "cohort": name, "problems": s.problem_id.nunique(), "cuts": len(s)}
        if s.problem_id.nunique() < 5:  # too few problems for the grouped 5-fold probes: D only
            s = s[s.groupby("problem_id").trace_id.transform("size") >= 2].reset_index(drop=True)
            A, B, _ = DT.halves(s)
            fix, tot = DT.sums(s.problem_id.values, A, B), DT.total_sums(s)
            pids = fix.index.values
            dr = S.problem_draws(len(pids), 2000, 0)
            fx, tt = fix.loc[pids].values, tot.loc[pids].values
            lo, hi = S.ci((dr @ fx) / (dr @ tt))
            row.update(D=fx.sum() / tt.sum(), D_lo=lo, D_hi=hi)
            summary.append(row)
            continue
        x1 = DT.run(s, tr, meta, acts)
        x1.to_csv(f"{OUT}/x1_{model}_{name}.csv", index=False)
        fw = FC.run(s)
        fw.to_csv(f"{OUT}/forward_{model}_{name}.csv", index=False)
        d = x1[x1.what == "determined"].iloc[0]
        row.update(D=d.est, D_lo=d.lo, D_hi=d.hi)
        cap = x1[x1.what.str.startswith("capture:") & ~x1.what.str.contains("oracle")]
        row.update(max_free_capture=cap.est.max(), max_free_capture_hi=cap.hi.max())
        if name != "same":
            x2 = CR.run(s, y)
            x2.to_csv(f"{OUT}/x2_{model}_{name}.csv", index=False)
            full = x2[(x2.score == "correct") & (x2.K == 8) & (x2.L == 32768)].iloc[0]
            row.update(auroc_full=full.auroc, auroc_lo=full.lo, auroc_hi=full.hi, pair_problems=full.problems,
                       pairs=full.pairs)
        summary.append(row)
pd.DataFrame(summary).to_csv(f"{OUT}/summary.csv", index=False)
print(pd.DataFrame(summary).round(3).to_string(index=False))
