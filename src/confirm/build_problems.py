# the problems file gpu_prep.py t0 reads: one jsonl row per id of the frozen id list, in list order, with the
# problem text and gold answer. ids are the planning package's "mathsha256:<sha256 of the NFKC-normalized,
# whitespace-collapsed problem text>" over qwedsacf/competition_math; gold = last \boxed{} of the reference
# solution. refuses an id list whose sha256 is not spec.SLOTS["problem_ids_csv_sha256"]. prints the file's sha256
# for spec.SLOTS["problems_file_sha256"]. downloads the dataset: run on Colab.
#   python src/confirm/build_problems.py --ids confirmation_after_pilot_1000.csv --out problems.jsonl
import argparse, csv, hashlib, json, os, sys, unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import spec  # noqa: E402
from generate import extract_boxed  # noqa: E402

DATASET = "qwedsacf/competition_math"
REVISION = "e839825f9ec5c6cfa585c654a59610969ec13993"  # the planning package's selection manifest


def problem_id(text):
    return "mathsha256:" + hashlib.sha256(" ".join(unicodedata.normalize("NFKC", text).split()).encode()).hexdigest()


def read_ids(path):
    rows = list(csv.DictReader(open(path, newline="")))
    rows.sort(key=lambda r: int(r["stage_order"]))
    return [(r["problem_id"], int(r["level"])) for r in rows]


def build(ids, dataset_rows):
    # ids: [(problem_id, level)] in list order; dataset_rows: dicts with problem, solution, level, type
    by_id = {}
    for r in dataset_rows:
        by_id.setdefault(problem_id(r["problem"]), []).append(r)
    out, missing, bad = [], [], []
    for pid, level in ids:
        hits = by_id.get(pid)
        if not hits:
            missing.append(pid)
            continue
        golds = {extract_boxed(h["solution"]) for h in hits}
        gold = golds.pop() if len(golds) == 1 else None
        if not gold or f"Level {level}" != hits[0]["level"]:
            bad.append(pid)
            continue
        out.append({"problem_id": pid, "problem": hits[0]["problem"], "gold": gold, "level": level,
                    "subject": hits[0]["type"]})
    if missing or bad or len({r["problem_id"] for r in out}) != len(ids):
        raise SystemExit(f"{len(missing)} ids not in {DATASET}, {len(bad)} with no single gold or a level mismatch, "
                         f"{len(ids) - len(set(p for p, _ in ids))} duplicate ids: {(missing + bad)[:5]}")
    return out


def write(rows, path):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n")
    return spec.file_sha(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sha = spec.file_sha(a.ids)
    if sha != spec.SLOTS["problem_ids_csv_sha256"]:
        raise SystemExit(f"{a.ids} has sha256 {sha}, spec.SLOTS['problem_ids_csv_sha256'] is "
                         f"{spec.SLOTS['problem_ids_csv_sha256']}")
    from datasets import load_dataset
    ds = load_dataset(DATASET, revision=REVISION, split="train")
    rows = build(read_ids(a.ids), ds)
    out_sha = write(rows, a.out)
    print(f"wrote {len(rows)} problems to {a.out}\nproblems_file_sha256 {out_sha}")


if __name__ == "__main__":
    main()
