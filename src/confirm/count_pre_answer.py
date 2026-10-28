# how many live cuts and mixed problems each pre-answer rule keeps on the dev data. COUNTS ONLY: it reads
# prefixes, final answers and correct / wrong labels, never a monitor score, so no S7 outcome (AUROC or
# gain) is computed here. two populations: every dev trace, and the 87 problems of the dev dry run
# (problems with ceiling rollouts, if --rollouts is given).
#   python src/confirm/count_pre_answer.py --data data [--rollouts <recovery_ceiling>]
import argparse, json, os, sys
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "probe_cmp"))
import spec  # noqa: E402
import bundle_lib as B  # noqa: E402
import data as D  # noqa: E402

RULES = {"pre_answer": "stated answer (primary)", "pre_answer_strict": "strict value match",
         "pre_answer_think_boxed": "</think> and \\boxed only"}


def rollout_problems(path):
    out = set()
    for line in open(path):
        try:
            out.add(json.loads(line)["problem_id"])
        except (json.JSONDecodeError, KeyError):
            continue
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--registry", default=D.REGISTRY)
    ap.add_argument("--rollouts", default=None)
    a = ap.parse_args()
    store = D.Store(a.data, reg=D.load_registry(a.registry), names=["B_fixed"])
    tr = store.traces
    correct = (tr.is_correct.astype(bool) & ~tr.trunc.astype(bool)).astype(int)
    pops = {"all dev traces": set(tr.problem_id)}
    if a.rollouts:
        pops["dry-run problems (with rollouts)"] = rollout_problems(a.rollouts)
    rows = []
    for c in spec.CHECKPOINTS:
        tab = store.table[store.table.ckpt == str(c)]
        ids = tab.trace_id.tolist()
        flags = pd.DataFrame([B.cut_rules(t, tr.at[i, "answer"], True) for i, t in zip(ids, store.text(str(c), ids))])
        flags["trace_id"], flags["problem_id"] = ids, tr.loc[ids, "problem_id"].values
        flags["correct"] = correct.loc[ids].values
        for pname, probs in pops.items():
            f = flags[flags.problem_id.isin(probs)]
            for col in ["live", *RULES]:
                k = f[f[col]]
                rows.append({"population": pname, "checkpoint": c, "rule": "live (</think> only)" if col == "live" else RULES[col],
                             "cuts": len(k), "problems": k.problem_id.nunique(), "mixed_problems": mixed_problems(k)})
    out = pd.DataFrame(rows)
    for pname, g in out.groupby("population", sort=False):
        print(f"\n{pname}: live cuts kept / problems / mixed problems, per rule (labels only, no scores)")
        print(g.pivot_table(index="rule", columns="checkpoint", values=["cuts", "mixed_problems"], aggfunc="first")
              .reindex(["live (</think> only)", *RULES.values()]).to_string())


def mixed_problems(rows):
    g = rows.groupby("problem_id").correct.agg(lambda v: 0 < v.sum() < len(v))
    return int(g.sum())


if __name__ == "__main__":
    main()
