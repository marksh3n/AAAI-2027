# classify wrong traces: ordinary error, degenerate loop, no answer, truncation.
# path is duplicated rather than imported from generate/extract so this runs on a bare
# cpu runtime without pulling in torch
import argparse, json, re
import pandas as pd
from sklearn.metrics import roc_auc_score

TRACES_PATH = "data/traces.jsonl"
OUT_PATH = "data/taxonomy.csv"
SAMPLE_PATH = "data/taxonomy_sample.txt"
SHINGLE = 20
LOOP_REP = 0.6
LOOP_COUNT = 10


def shingles(text):
    words = re.sub(r"\s+", " ", text).split()
    return [" ".join(words[i : i + SHINGLE]) for i in range(len(words) - SHINGLE + 1)]


def classify(t):
    # a loop is defined by repeated content, not by n_gen: length vs wrongness is the
    # thing under test, so thresholding on length here would beg the question
    if t["trunc"]:
        return "truncated"
    if t["answer"] is None:
        return "no_answer"
    if t["rep"] >= LOOP_REP or t["max_rep"] >= LOOP_COUNT:
        return "loop"
    return "ordinary"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=25)
    args = parser.parse_args()

    traces = [json.loads(line) for line in open(TRACES_PATH)]
    for t in traces:
        grams = shingles(t["text"].split("</think>")[0])
        counts = pd.Series(grams).value_counts() if grams else pd.Series(dtype=int)
        t["rep"] = 1 - len(counts) / len(grams) if grams else 0.0
        t["max_rep"] = int(counts.iloc[0]) if len(counts) else 0
        t["wrong"] = bool(t["trunc"] or not t["is_correct"])
        t["cat"] = classify(t) if t["wrong"] else "correct"

    df = pd.DataFrame(traces)[
        ["trace_id", "problem_id", "level", "cat", "wrong", "n_gen", "rep", "max_rep"]
    ]
    wrong = df[df.wrong]
    print(f"{len(wrong)} wrong of {len(df)}\n")
    print(
        wrong.groupby("cat")
        .agg(n=("n_gen", "size"), median_n_gen=("n_gen", "median"), median_rep=("rep", "median"))
        .sort_values("n", ascending=False)
        .to_string()
    )
    ok = df[~df.wrong]
    print(f"\ncorrect: n={len(ok)} median_n_gen={ok.n_gen.median():.0f} "
          f"median_rep={ok.rep.median():.3f}")

    # the number that decides the pilot: if length stops predicting wrongness once
    # degenerate failures are dropped, looping is the confound rather than error awareness
    print()
    for label, drop in [("all", []), ("-loop", ["loop"]),
                        ("-loop-no_answer-truncated", ["loop", "no_answer", "truncated"])]:
        sub = df[~df.cat.isin(drop)]
        auc = roc_auc_score(sub.wrong, sub.n_gen)
        print(f"length-only AUROC (n_gen) {label:26} {auc:.3f}  (n_wrong {int(sub.wrong.sum())})")

    df.to_csv(OUT_PATH, index=False)

    # heuristics will misfire, so dump head and tail of a sample to correct cat by hand
    sample = wrong.sample(min(args.n, len(wrong)), random_state=0).sort_values("cat")
    by_id = {t["trace_id"]: t for t in traces}
    with open(SAMPLE_PATH, "w") as f:
        for r in sample.itertuples():
            t = by_id[r.trace_id]
            f.write(f"=== {r.trace_id} | {r.cat} | level {r.level} | n_gen {r.n_gen} | "
                    f"rep {r.rep:.3f} | max_rep {r.max_rep}\n")
            f.write(f"gold {t['gold']!r} got {t['answer']!r}\n")
            f.write(t["text"][:1200] + "\n  [...]\n" + t["text"][-1200:] + "\n\n")
    print(f"\nwrote {OUT_PATH} and {SAMPLE_PATH} ({len(sample)} traces to hand-check)")


if __name__ == "__main__":
    main()
