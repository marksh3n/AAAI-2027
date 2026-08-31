# classify wrong traces: ordinary, loop, no answer, truncation.
# path duplicated, not imported: keeps this torch-free
import argparse, json, re
import pandas as pd
from sklearn.metrics import roc_auc_score

TRACES_PATH = "data/traces.jsonl"
OUT_PATH = "data/taxonomy.csv"
SAMPLE_PATH = "data/taxonomy_sample.txt"
SHINGLE = 20
RESTART = re.compile(r"\b(wait|hold on|let me re|actually)\b", re.I)


def shingles(text):
    words = re.sub(r"\s+", " ", text).split()
    return [" ".join(words[i : i + SHINGLE]) for i in range(len(words) - SHINGLE + 1)]


def classify(t, loop_rep):
    # rep is a fraction; raw counts scale with trace length
    if t["trunc"]:
        return "truncated"
    if t["answer"] is None:
        return "no_answer"
    if t["rep"] >= loop_rep:
        return "loop"
    return "ordinary"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=25)
    parser.add_argument("--chars", type=int, default=1200)
    parser.add_argument("--loop-rep", type=float, default=0.3)
    args = parser.parse_args()

    traces = [json.loads(line) for line in open(TRACES_PATH)]
    for t in traces:
        think = t["text"].split("</think>")[0]
        grams = shingles(think)
        counts = pd.Series(grams).value_counts() if grams else pd.Series(dtype=int)
        t["rep"] = 1 - len(counts) / len(grams) if grams else 0.0
        t["max_rep"] = int(counts.iloc[0]) if len(counts) else 0
        # per 1k tokens, and online-computable unlike final n_gen
        t["restarts"] = len(RESTART.findall(think)) / max(t["n_gen"], 1) * 1000
        t["wrong"] = bool(t["trunc"] or not t["is_correct"])
        t["cat"] = classify(t, args.loop_rep) if t["wrong"] else "correct"

    df = pd.DataFrame(traces)[
        ["trace_id", "problem_id", "level", "cat", "wrong", "n_gen", "rep", "max_rep", "restarts"]
    ]
    wrong, ok = df[df.wrong], df[~df.wrong]
    print(f"{len(wrong)} wrong of {len(df)}  (loop threshold rep >= {args.loop_rep})\n")
    print(
        df.groupby("cat")
        .agg(n=("n_gen", "size"), median_n_gen=("n_gen", "median"), median_rep=("rep", "median"),
             median_restarts=("restarts", "median"))
        .sort_values("n", ascending=False)
        .round(3)
        .to_string()
    )

    q = [0.25, 0.5, 0.75]
    print(f"\nn_gen q25/50/75   wrong {wrong.n_gen.quantile(q).astype(int).tolist()}"
          f"   correct {ok.n_gen.quantile(q).astype(int).tolist()}")

    # dropping long traces lowers length AUROC whatever the cause
    print(f"\n{'AUROC by subset':32}{'n_wrong':>8}{'n_gen':>8}{'restarts':>10}")
    for label, sub in [
        ("all", df),
        ("-loop", df[df.cat != "loop"]),
        ("-loop-no_answer-truncated", df[~df.cat.isin(["loop", "no_answer", "truncated"])]),
        ("wrong below their median", df[~df.wrong | (df.n_gen <= wrong.n_gen.median())]),
        ("wrong below their q25", df[~df.wrong | (df.n_gen <= wrong.n_gen.quantile(0.25))]),
    ]:
        print(f"{label:32}{int(sub.wrong.sum()):>8}"
              f"{roc_auc_score(sub.wrong, sub.n_gen):>8.3f}"
              f"{roc_auc_score(sub.wrong, sub.restarts):>10.3f}")

    # pick the loop threshold off this
    print("\nrep over wrong traces, sorted:")
    print(" ".join(f"{v:.3f}" for v in sorted(wrong.rep, reverse=True)))

    df.to_csv(OUT_PATH, index=False)

    # head and tail for hand-correcting cat
    sample = wrong.sample(min(args.n, len(wrong)), random_state=0).sort_values("cat")
    by_id = {t["trace_id"]: t for t in traces}
    with open(SAMPLE_PATH, "w") as f:
        for r in sample.itertuples():
            t = by_id[r.trace_id]
            f.write(f"=== {r.trace_id} | {r.cat} | level {r.level} | n_gen {r.n_gen} | "
                    f"rep {r.rep:.3f} | max_rep {r.max_rep} | restarts {r.restarts:.2f}\n")
            f.write(f"gold {t['gold']!r} got {t['answer']!r}\n")
            f.write(t["text"][: args.chars] + "\n  [...]\n" + t["text"][-args.chars :] + "\n\n")
    print(f"\nwrote {OUT_PATH} and {SAMPLE_PATH} ({len(sample)} traces to hand-check)")


if __name__ == "__main__":
    main()
