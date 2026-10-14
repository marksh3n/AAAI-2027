# r(t) = P(final answer correct | prefix), estimated by K continuations.
# (trace, pos) pairs come from meta.parquet; prefix tokens re-derived from the trace.
import argparse, hashlib, json, os
import pandas as pd
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from cache import StaleCache, changed_keys, set_aside, sha_obj, short
from generate import MODEL, answers_match, extract_boxed
from extract import POSITIONS, encode, paths

OUT_DIR = "data"
# budget on trace tokens, not context. a prefix already spent pos of it.
TRACE_TOKEN_CAP = 32768
# per-continuation cap. uncapped is ~390k tokens per prefix at n=12,
# and ~1GB kv per sequence. hitting it records capped, not wrong.
MAX_CONTINUATION = 8192
MIN_BUDGET = 512  # a prefix this close to the cap has almost no room left to recover in
PROMPT_HEADROOM = 1024


def recovery_path(tag):
    return f"{OUT_DIR}/recovery{f'_{tag}' if tag else ''}.jsonl"


def summary_path(tag):
    return f"{OUT_DIR}/recovery_summary{f'_{tag}' if tag else ''}.csv"


def config_path(tag):
    return f"{OUT_DIR}/recovery{f'_{tag}' if tag else ''}.config.json"


def add_args(parser):
    parser.add_argument("--positions", type=int, nargs="+", default=[128, 512, 1024],
                        help="must be a subset of extract.py's POSITIONS")
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--temp", type=float, default=0.7,
                        help="must match the traces' generation temperature; r(t) is "
                             "defined under the model's own sampling distribution")
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-continuation", type=int, default=MAX_CONTINUATION)
    parser.add_argument("--n", type=int, default=0,
                        help="cap on distinct traces to pull prefixes from, 0 = all")
    parser.add_argument("--wrong-only", action="store_true",
                        help="skip already-correct traces. off by default: the "
                             "within-problem contrast needs both classes")
    parser.add_argument("--chunk", type=int, default=8)
    parser.add_argument("--tag", default="", help="tag of the traces/meta this reads from")
    parser.add_argument("--out-tag", default="", help="tag for the recovery output; defaults to --tag")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rebuild", action="store_true",
                        help="set aside output written under a different config")
    return parser


def run_config(args):
    # fields that change what a continuation is. samples excluded: raising it tops up.
    cfg = {
        "model": MODEL,
        "temp": args.temp,
        "top_p": args.top_p,
        "max_continuation": args.max_continuation,
        "trace_token_cap": TRACE_TOKEN_CAP,
    }
    return cfg, sha_obj(cfg)


def check_config(tag, cfg, cfg_hash, rebuild=False):
    # rows reusable only when something on disk records what produced them
    jsonl, path = recovery_path(tag), config_path(tag)
    if not os.path.exists(jsonl):
        json.dump({"config_hash": cfg_hash, "config": cfg}, open(path, "w"), indent=2)
        return
    if not os.path.exists(path):
        if rebuild:
            set_aside(jsonl)
            json.dump({"config_hash": cfg_hash, "config": cfg}, open(path, "w"), indent=2)
            return
        raise StaleCache(
            f"{jsonl} exists, {path} does not: sampling config unknown.\n"
            "Use --rebuild or a new --out-tag."
        )
    old = json.load(open(path))
    if old.get("config_hash") != cfg_hash:
        if rebuild:
            set_aside(jsonl)
            json.dump({"config_hash": cfg_hash, "config": cfg}, open(path, "w"), indent=2)
            return
        diff = changed_keys(old.get("config", {}), cfg)
        raise StaleCache(
            f"{jsonl} was sampled under a different config ({diff}); mixing pools two "
            "distributions. Use --rebuild or a new --out-tag."
        )


def load_prefixes(meta_path, positions, wrong_only):
    bad = [p for p in positions if p not in POSITIONS]
    if bad:
        raise ValueError(
            f"{bad} not in extract.py's POSITIONS {POSITIONS} -- meta.parquet has no row for them"
        )
    df = pd.read_parquet(meta_path)
    df = df[df.pos.isin(positions)]
    if wrong_only:
        df = df[~df.is_correct]
    return df.reset_index(drop=True)


def sample_counts(path):
    # continuations per (trace_id, pos), so a resume tops up by the shortfall
    counts = {}
    if not os.path.exists(path):
        return counts
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            key = (r["trace_id"], r["pos"])
            counts[key] = counts.get(key, 0) + 1
    return counts


def select(df, args, counts):
    have = [counts.get(k, 0) for k in zip(df.trace_id, df.pos)]
    df = df.assign(have=have)
    df = df[df.have < args.samples]
    if args.n and len(df):
        # whole traces, so a pilot covers every position for each trace kept
        keep = df.trace_id.drop_duplicates().sample(
            min(args.n, df.trace_id.nunique()), random_state=args.seed
        )
        df = df[df.trace_id.isin(keep)]
    done = sum(1 for v in counts.values() if v >= args.samples)
    print(f"{len(df)} prefixes to sample ({done} already at full budget), "
          f"{df.trace_id.nunique()} traces, "
          f"positions {sorted(int(v) for v in df.pos.unique())}")
    return df.reset_index(drop=True)


def row_seed(base, trace_id, pos, have):
    # fresh stream per (prefix, top-up). python hash() is salted, so sha256.
    digest = hashlib.sha256(f"{base}:{trace_id}:{pos}:{have}".encode()).hexdigest()
    return int(digest[:8], 16)


def record(trace, pos, k, prefix, text, n_gen, budget):
    # scored trajectory is prefix + continuation: a deep prefix can already hold
    # </think> or the \boxed{}. only text is stored, prefix repeats per sample.
    full = prefix + text
    answer = extract_boxed(full)
    return {
        "trace_id": trace["trace_id"],
        "problem_id": trace["problem_id"],
        "level": trace["level"],
        "gold": trace["gold"],
        "pos": pos,
        "sample_idx": k,
        "n_gen": n_gen,
        "budget": budget,
        "capped": n_gen >= budget,
        "trunc": n_gen >= budget or "</think>" not in full,
        "answer": answer,
        "is_correct": answers_match(answer, trace["gold"]),
        "text": text,
    }


def main():
    args = add_args(argparse.ArgumentParser()).parse_args()
    out_tag = args.out_tag or args.tag
    cfg, cfg_hash = run_config(args)
    check_config(out_tag, cfg, cfg_hash, args.rebuild)
    print(f"config {short(cfg_hash)}: temp {args.temp}, top_p {args.top_p}, "
          f"continuation cap {args.max_continuation}", flush=True)

    traces_path, _, meta_path, _ = paths(args.tag)
    traces = {t["trace_id"]: t for t in map(json.loads, open(traces_path))}
    counts = sample_counts(recovery_path(out_tag))
    df = select(load_prefixes(meta_path, args.positions, args.wrong_only), args, counts)
    if df.empty:
        summarise(out_tag, traces)
        return

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    llm = LLM(model=MODEL, dtype="float16", gpu_memory_utilization=0.90,
              max_model_len=max(args.positions) + args.max_continuation + PROMPT_HEADROOM)

    ids_cache = {}
    out_file = open(recovery_path(out_tag), "a")
    skipped = 0
    for start in range(0, len(df), args.chunk):
        batch = df.iloc[start : start + args.chunk]
        prompts, params_list, meta = [], [], []
        for r in batch.itertuples():
            trace = traces[r.trace_id]
            if r.trace_id not in ids_cache:
                prompt_ids, trace_ids, _ = encode(tokenizer, trace)
                ids_cache[r.trace_id] = (prompt_ids, trace_ids)
            prompt_ids, trace_ids = ids_cache[r.trace_id]
            # numpy scalars are not JSON-serialisable; cast before any write
            pos, have = int(r.pos), int(r.have)
            prefix_ids = trace_ids[:pos]
            # meta.prefix is tokenizer.decode(trace_ids[:pos]). disagreement means
            # meta and traces.jsonl are from different generations.
            prefix = tokenizer.decode(prefix_ids)
            if prefix != r.prefix:
                raise StaleCache(
                    f"{r.trace_id} @ {pos}: prefix from {traces_path} != {meta_path}. "
                    "Re-run extract.py --fixed."
                )

            budget = int(min(args.max_continuation, TRACE_TOKEN_CAP - pos))
            if budget < MIN_BUDGET:
                skipped += 1
                continue
            need = args.samples - have
            # token ids, never re-templated, never round-tripped through text:
            # BPE can retokenize a decoded prefix to a different boundary.
            prompts.append({"prompt_token_ids": prompt_ids + prefix_ids})
            params_list.append(SamplingParams(
                temperature=args.temp, top_p=args.top_p, max_tokens=budget,
                n=need, seed=row_seed(args.seed, r.trace_id, pos, have),
            ))
            meta.append((trace, pos, prefix, budget, have))

        if not prompts:
            continue
        for (trace, pos, prefix, budget, have), out in zip(meta, llm.generate(prompts, params_list)):
            for k, o in enumerate(out.outputs):
                out_file.write(json.dumps(
                    record(trace, pos, have + k, prefix, o.text, len(o.token_ids), budget)) + "\n")
        out_file.flush()
        print(f"{min(start + args.chunk, len(df))}/{len(df)}", flush=True)
    out_file.close()

    if skipped:
        print(f"skipped {skipped} prefixes with < {MIN_BUDGET} tokens of budget left")
    summarise(out_tag, traces)


def within_problem_auroc(df, score, label):
    # fraction of within-problem (correct, wrong) pairs score orders right.
    # same statistic probe_fixed reports, so the two compare directly.
    wins = total = pairs_problems = 0
    for _, g in df.groupby("problem_id"):
        hi, lo = g[score][g[label] == 1].values, g[score][g[label] == 0].values
        if not len(hi) or not len(lo):
            continue
        d = hi[:, None] - lo[None, :]
        wins += (d > 0).sum() + 0.5 * (d == 0).sum()
        total += d.size
        pairs_problems += 1
    return (wins / total if total else float("nan")), int(total), pairs_problems


def summarise(tag, traces=None):
    path = recovery_path(tag)
    if not os.path.exists(path):
        return
    df = pd.DataFrame(json.loads(line) for line in open(path))
    if df.empty:
        return
    # an overlapping top-up would otherwise be counted twice
    df = df.drop_duplicates(["trace_id", "pos", "sample_idx"], keep="last")

    agg = (
        df.groupby(["trace_id", "problem_id", "level", "pos"])
        .agg(n_samples=("is_correct", "size"),
             n_correct=("is_correct", "sum"),
             n_capped=("capped", "sum"),
             trunc_rate=("trunc", "mean"))
        .reset_index()
    )
    agg["n_terminated"] = agg.n_samples - agg.n_capped
    # a capped continuation is neither correct nor wrong. counting it wrong biases r
    # down for long traces; dropping it biases up. report both and the cap rate.
    agg["recovery_rate"] = agg.n_correct / agg.n_samples
    agg["recovery_rate_terminated"] = agg.n_correct / agg.n_terminated.replace(0, pd.NA)
    agg["cap_rate"] = agg.n_capped / agg.n_samples
    agg.to_csv(summary_path(tag), index=False)
    print(f"\n{len(agg)} prefixes, {len(df)} continuations -> {summary_path(tag)}")

    print(
        agg.groupby("pos")
        .agg(n_prefixes=("recovery_rate", "size"),
             mean_r=("recovery_rate", "mean"),
             median_r=("recovery_rate", "median"),
             mean_r_terminated=("recovery_rate_terminated", "mean"),
             cap_rate=("cap_rate", "mean"),
             pct_never_recovers=("n_correct", lambda s: (s == 0).mean()),
             pct_always_recovers=("recovery_rate", lambda s: (s == 1).mean()))
        .round(3)
        .to_string()
    )
    print(
        "\nby level:\n"
        + agg.groupby(["level", "pos"]).recovery_rate.mean().unstack("pos").round(3).to_string()
    )

    if traces is None:
        return
    # r(t) is the Bayes-optimal prefix score, so this is the probe ceiling.
    # read next to probe_fixed's TR/TRH.
    agg["y"] = agg.trace_id.map(lambda t: int(bool(traces[t]["is_correct"])))
    print("\nwithin-problem AUROC of r(t) against the trace's realised outcome")
    print(f"{'pos':>6}{'auroc':>9}{'pairs':>8}{'problems':>10}")
    for pos, g in agg.groupby("pos"):
        auroc, pairs, probs = within_problem_auroc(g, "recovery_rate", "y")
        print(f"{pos:>6}{auroc:>9.3f}{pairs:>8}{probs:>10}")


if __name__ == "__main__":
    main()
