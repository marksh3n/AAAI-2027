# problems file for a replication run (E0_MODEL set): the dev problems, i.e. the problems of traces_B.jsonl
# (MATH-500 levels 3-5), one row each, same problem_id as the dev traces so the models pair up. rows are in a
# seeded shuffle, so a sources run stopped by its budget is still a random subset.
#   python3 src/confirm/dev_problems.py --traces data/hf/traces_B.jsonl --out data/problems_dev.jsonl
import argparse, hashlib, json, random


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    probs = {}
    for line in open(a.traces):
        if not line.strip():
            continue
        t = json.loads(line)
        row = {"problem_id": str(t["problem_id"]), "problem": t["problem"], "gold": t["gold"],
               "level": str(t["level"]), "subject": t["subject"], "uid": t["uid"]}
        old = probs.setdefault(row["problem_id"], row)
        if old != row:
            raise SystemExit(f"problem {row['problem_id']} differs between its traces")
    rows = sorted(probs.values(), key=lambda r: int(r["problem_id"]))
    random.Random(a.seed).shuffle(rows)
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    sha = hashlib.sha256(open(a.out, "rb").read()).hexdigest()
    levels = {lv: sum(r["level"] == lv for r in rows) for lv in sorted({r["level"] for r in rows})}
    print(f"{len(rows)} problems {levels} -> {a.out}\nsha256 {sha}")


if __name__ == "__main__":
    main()
