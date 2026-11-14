# fixed 50/50 problem split of the ceiling-rollout cohort: explore (Phase 3 choices) and holdout (sign check).
# stratified by MATH level, seed 0. written once, before any Phase 3 result.
#   python src/explore/make_split.py --rollouts data/recovery_ceiling --traces <traces_B.jsonl> --out results/split.json
import argparse, hashlib, json
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", required=True)
    ap.add_argument("--traces", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--population", choices=["rollouts", "traces"], default="rollouts",
                    help="problems with ceiling rollouts (87) or every dev problem with traces (367)")
    a = ap.parse_args()
    probs = set()
    for line in open(a.rollouts if a.population == "rollouts" else a.traces):
        try:
            probs.add(json.loads(line)["problem_id"])
        except (json.JSONDecodeError, KeyError):
            continue
    level = {}
    for line in open(a.traces):
        t = json.loads(line)
        level[t["problem_id"]] = t["level"]
    rng = np.random.default_rng(a.seed)
    explore, holdout = [], []
    for lv in sorted({level[p] for p in probs}):
        ids = sorted(p for p in probs if level[p] == lv)
        ids = [ids[i] for i in rng.permutation(len(ids))]
        h = len(ids) // 2 + (len(ids) % 2) * int(len(explore) > len(holdout))
        explore += ids[h:]
        holdout += ids[:h]
    allp = sorted(probs)
    out = {"seed": a.seed, "stratified_by": "level", "n_problems": len(allp),
           "problems_sha256": hashlib.sha256(json.dumps(allp).encode()).hexdigest(),
           "explore": sorted(explore), "holdout": sorted(holdout)}
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"explore {len(explore)}, holdout {len(holdout)}")


if __name__ == "__main__":
    main()
