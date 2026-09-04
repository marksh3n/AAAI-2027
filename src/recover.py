# resample continuations from extract.py's fixed-position prefixes and measure whether the
# model still reaches the correct final answer -- i.e. how recoverable a trajectory is at
# 128/512/1024+ tokens in. Consumes meta.parquet from `extract.py --fixed`, which already
# stores the decoded prefix text at each position (see extract.fixed(), the `prefix` column).
import argparse, json, os
import pandas as pd
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from generate import MODEL, answers_match, build_prompt, extract_boxed
from extract import POSITIONS, paths

OUT_DIR = "data"
# must match the --max-new the traces were generated with (generate.py's default): this is
# a budget on TRACE tokens, not total context. A prefix that already ate `pos` of it gets
# the remainder, not a fresh 32768.
TRACE_TOKEN_CAP = 32768
LLM_MAX_LEN = TRACE_TOKEN_CAP + 2048  # + room for the problem prompt itself
MIN_BUDGET = 512  # a prefix this close to the cap has almost no room left to recover in
CHUNK = 4  # each prompt still needs its own forward pass through up to ~32k prefix tokens,
           # even though n>1 shares the kv cache across a prompt's own samples


def recovery_path(tag):
    return f"{OUT_DIR}/recovery{f'_{tag}' if tag else ''}.jsonl"


def summary_path(tag):
    return f"{OUT_DIR}/recovery_summary{f'_{tag}' if tag else ''}.csv"


def add_args(parser):
    parser.add_argument("--positions", type=int, nargs="+", default=[128, 512, 1024],
                         help="must be a subset of extract.py's POSITIONS")
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--temp", type=float, default=0.7)
    parser.add_argument("--n", type=int, default=0,
                         help="cap on distinct traces to pull prefixes from, 0 = all")
    parser.add_argument("--wrong-only", action="store_true", default=True)
    parser.add_argument("--include-correct", dest="wrong_only", action="store_false",
                         help="also resample prefixes from traces that were already correct")
    parser.add_argument("--tag", default="", help="tag of the traces/meta this reads from")
    parser.add_argument("--out-tag", default="", help="tag for the recovery output; defaults to --tag")
    parser.add_argument("--seed", type=int, default=0)
    return parser


def load_prefixes(meta_path, positions, wrong_only):
    bad = [p for p in positions if p not in POSITIONS]
    if bad:
        raise ValueError(
            f"{bad} not in extract.py's POSITIONS {POSITIONS} -- meta.parquet has no prefix column for them"
        )
    df = pd.read_parquet(meta_path)
    df = df[df.pos.isin(positions)]
    if wrong_only:
        df = df[~df.is_correct]
    return df.reset_index(drop=True)


def already_done(path, samples):
    # a (trace_id, pos) pair only counts as done once it has its full sample budget, so an
    # interrupted run tops up a partially-sampled prefix instead of silently skipping it
    if not os.path.exists(path):
        return set()
    counts = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            key = (r["trace_id"], r["pos"])
            counts[key] = counts.get(key, 0) + 1
    return {k for k, n in counts.items() if n >= samples}


def select(df, args):
    done = already_done(recovery_path(args.out_tag or args.tag), args.samples)
    df = df[~df.apply(lambda r: (r.trace_id, r.pos) in done, axis=1)]
    if args.n:
        # sample whole traces, not individual rows, so a pilot still covers every
        # requested position for each trace it does include
        keep = df.trace_id.drop_duplicates().sample(
            min(args.n, df.trace_id.nunique()), random_state=args.seed
        )
        df = df[df.trace_id.isin(keep)]
    print(f"{len(df)} prefixes to sample ({len(done)} already done), "
          f"{df.trace_id.nunique()} traces, positions {sorted(df.pos.unique())}")
    return df.reset_index(drop=True)


def check_retokenization(tokenizer, prompt, prefix, pos):
    # meta.parquet stores `prefix` as DECODED TEXT (extract.fixed(), tokenizer.decode(ids[:pos])).
    # Re-tokenizing text is not guaranteed to reproduce the exact original token boundary at
    # the cut point -- BPE can merge or split differently right at the edge. This only reports
    # the drift, it does not correct it: a "pos=128" label is only as good as this staying ~0.
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(prompt + prefix, add_special_tokens=False)["input_ids"]
    return len(full_ids) - len(prompt_ids) - pos


def record(trace, pos, k, text, n_gen, budget):
    answer = extract_boxed(text)
    return {
        "trace_id": trace["trace_id"],
        "problem_id": trace["problem_id"],
        "level": trace["level"],
        "pos": pos,
        "sample_idx": k,
        "n_gen": n_gen,
        "trunc": n_gen >= budget or "</think>" not in text,
        "answer": answer,
        "is_correct": answers_match(answer, trace["gold"]),
        "text": text,
    }


def main():
    args = add_args(argparse.ArgumentParser()).parse_args()
    out_tag = args.out_tag or args.tag

    traces = {t["trace_id"]: t for t in map(json.loads, open("data/traces.jsonl"))}
    _, _, meta_path, _ = paths(args.tag)
    df = select(load_prefixes(meta_path, args.positions, args.wrong_only), args)
    if df.empty:
        return

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    llm = LLM(model=MODEL, dtype="float16", max_model_len=LLM_MAX_LEN, gpu_memory_utilization=0.90)

    out_file = open(recovery_path(out_tag), "a")
    drift_report, skipped = [], 0
    for start in range(0, len(df), CHUNK):
        batch = df.iloc[start : start + CHUNK]
        prompts, params_list, meta = [], [], []
        for r in batch.itertuples():
            trace = traces[r.trace_id]
            prompt = build_prompt(tokenizer, trace["problem"], trace.get("structured", False))
            drift_report.append(check_retokenization(tokenizer, prompt, r.prefix, r.pos))

            budget = TRACE_TOKEN_CAP - r.pos
            if budget < MIN_BUDGET:
                skipped += 1
                continue
            # raw continuation of the trace's own tokens, NOT re-templated: the model is
            # resuming a thought mid-stream, so nothing goes back through the chat template
            prompts.append(prompt + r.prefix)
            params_list.append(SamplingParams(
                temperature=args.temp, top_p=0.95, max_tokens=budget,
                n=args.samples, seed=args.seed,
            ))
            meta.append((trace, r.pos, budget))

        if not prompts:
            continue
        for (trace, pos, budget), out in zip(meta, llm.generate(prompts, params_list)):
            for k, o in enumerate(out.outputs):
                out_file.write(json.dumps(record(trace, pos, k, o.text, len(o.token_ids), budget)) + "\n")
        out_file.flush()
        print(f"{min(start + CHUNK, len(df))}/{len(df)}", flush=True)
    out_file.close()

    if skipped:
        print(f"skipped {skipped} prefixes with < {MIN_BUDGET} tokens of budget left")
    bad_drift = sum(abs(d) > 2 for d in drift_report)
    if bad_drift:
        print(f"warning: {bad_drift}/{len(drift_report)} prefixes retokenized "
              f"more than 2 tokens off their labeled position -- check check_retokenization()")

    summarise(out_tag)


def summarise(tag):
    path = recovery_path(tag)
    if not os.path.exists(path):
        return
    df = pd.DataFrame(json.loads(line) for line in open(path))
    if df.empty:
        return

    agg = (
        df.groupby(["trace_id", "problem_id", "level", "pos"])
        .agg(n_samples=("is_correct", "size"),
             n_recovered=("is_correct", "sum"),
             trunc_rate=("trunc", "mean"))
        .reset_index()
    )
    agg["recovery_rate"] = agg.n_recovered / agg.n_samples
    agg.to_csv(summary_path(tag), index=False)
    print(f"\n{len(agg)} prefixes, {len(df)} continuations -> {summary_path(tag)}")

    print(
        agg.groupby("pos")
        .agg(n_prefixes=("recovery_rate", "size"),
             mean_recovery=("recovery_rate", "mean"),
             median_recovery=("recovery_rate", "median"),
             pct_never_recovers=("n_recovered", lambda s: (s == 0).mean()),
             pct_always_recovers=("recovery_rate", lambda s: (s == 1).mean()))
        .round(3)
        .to_string()
    )
    print(
        "\nby level:\n"
        + agg.groupby(["level", "pos"])
        .recovery_rate.mean()
        .unstack("pos")
        .round(3)
        .to_string()
    )


if __name__ == "__main__":
    main()
