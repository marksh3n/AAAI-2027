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
RESTART = re.compile(r"\b(wait|hold on|let me re|actually)\b", re.I)


def shingles(text):
    words = re.sub(r"\s+", " ", text).split()
    return [" ".join(words[i : i + SHINGLE]) for i in range(len(words) - SHINGLE + 1)]


def all_boxed(text):
    # brace-matched like generate.extract_boxed, but every hit rather than the last:
    # the number of distinct candidates a trace cycles through is a crude flailing measure
    out, i = [], text.find("\\boxed{")
    while i >= 0:
        j, depth = i + 7, 1
        while j < len(text) and depth:
            depth += (text[j] == "{") - (text[j] == "}")
            j += 1
        if depth == 0:
            out.append(text[i + 7 : j - 1])
        i = text.find("\\boxed{", j)
    return out


def classify(t, loop_rep):
    # threshold on rep, which is a fraction, not on a raw repeat count: the same count
    # means very different things in a 2k and a 30k token trace
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
        t["cands"] = len(set(all_boxed(think)))
        t["restarts"] = len(RESTART.findall(think)) / max(t["n_gen"], 1) * 1000
        t["wrong"] = bool(t["trunc"] or not t["is_correct"])
        t["cat"] = classify(t, args.loop_rep) if t["wrong"] else "correct"

    df = pd.DataFrame(traces)[
        ["trace_id", "problem_id", "level", "cat", "wrong", "n_gen", "rep", "max_rep",
         "cands", "restarts"]
    ]
    wrong, ok = df[df.wrong], df[~df.wrong]
    print(f"{len(wrong)} wrong of {len(df)}  (loop threshold rep >= {args.loop_rep})\n")
    print(
        df.groupby("cat")
        .agg(n=("n_gen", "size"), median_n_gen=("n_gen", "median"), median_rep=("rep", "median"),
             median_cands=("cands", "median"), median_restarts=("restarts", "median"))
        .sort_values("n", ascending=False)
        .round(3)
        .to_string()
    )

    q = [0.25, 0.5, 0.75]
    print(f"\nn_gen q25/50/75   wrong {wrong.n_gen.quantile(q).astype(int).tolist()}"
          f"   correct {ok.n_gen.quantile(q).astype(int).tolist()}")

    # a loop classifier flags long traces, so dropping them lowers the length AUROC whether
    # or not looping is the cause. the assumption-free question is whether the shorter wrong
    # traces are still longer than correct ones, which bounds what any classifier could buy
    print()
    for label, sub in [
        ("all", df),
        ("-loop", df[df.cat != "loop"]),
        ("-loop-no_answer-truncated", df[~df.cat.isin(["loop", "no_answer", "truncated"])]),
        ("wrong below their median", df[~df.wrong | (df.n_gen <= wrong.n_gen.median())]),
        ("wrong below their q25", df[~df.wrong | (df.n_gen <= wrong.n_gen.quantile(0.25))]),
    ]:
        auc = roc_auc_score(sub.wrong, sub.n_gen)
        print(f"length-only AUROC (n_gen) {label:26} {auc:.3f}  (n_wrong {int(sub.wrong.sum())})")

    # pick the loop threshold off this rather than off a guess
    print("\nrep over wrong traces, sorted:")
    print(" ".join(f"{v:.3f}" for v in sorted(wrong.rep, reverse=True)))

    df.to_csv(OUT_PATH, index=False)

    # heuristics will misfire, so dump head and tail of a sample to correct cat by hand
    sample = wrong.sample(min(args.n, len(wrong)), random_state=0).sort_values("cat")
    by_id = {t["trace_id"]: t for t in traces}
    with open(SAMPLE_PATH, "w") as f:
        for r in sample.itertuples():
            t = by_id[r.trace_id]
            f.write(f"=== {r.trace_id} | {r.cat} | level {r.level} | n_gen {r.n_gen} | "
                    f"rep {r.rep:.3f} | max_rep {r.max_rep} | cands {r.cands}\n")
            f.write(f"gold {t['gold']!r} got {t['answer']!r}\n")
            f.write(t["text"][: args.chars] + "\n  [...]\n" + t["text"][-args.chars :] + "\n\n")
    print(f"\nwrote {OUT_PATH} and {SAMPLE_PATH} ({len(sample)} traces to hand-check)")


if __name__ == "__main__":
    main()
