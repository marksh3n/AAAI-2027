# n continuations per prefix with vLLM, fed token ids only. resumable: one record per
# (prefix_id, round), written once all n samples for that prefix are back.
import argparse, json, os, time

from common import MODEL, OUT_DIR, TOP_P, answers_match, extract_boxed, prefix_seed, read_jsonl
from cache import StaleCache, changed_keys, sha_obj, short

VLLM_VERSION = "0.10.2"
LOG_EVERY = 50
SMOKE = {"limit": 2, "n": 2, "max_tokens_cap": 256}


def add_args(parser):
    parser.add_argument("--prefix-file", default=f"{OUT_DIR}/prefixes.jsonl")
    parser.add_argument("--out", default=None, help=f"default {OUT_DIR}/rollouts.jsonl")
    parser.add_argument("--temp", type=float, required=True,
                        help="tag-B generation temperature, see estimate_temp.py")
    parser.add_argument("--top-p", type=float, default=TOP_P)
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="dry run on the first N prefixes")
    parser.add_argument("--n-override-file", default=None,
                        help="jsonl of {prefix_id, n}: only these run, n extra samples each")
    parser.add_argument("--round", type=int, default=0)
    parser.add_argument("--max-tokens-cap", type=int, default=0, help="smoke only; 0 = none")
    parser.add_argument("--chunk", type=int, default=32, help="prefixes per llm.generate call")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true",
                        help=f"{SMOKE}, separate output, asserts samples differ")
    parser.add_argument("--allow-version-mismatch", action="store_true")
    parser.add_argument("--gpu-mem", type=float, default=0.90)
    return parser


def prefix_sha(rec):
    # binds an output row to the exact tokens it continued
    return sha_obj([rec["prompt_token_ids"], rec["prefix_gen_token_ids"]])


def check_config(path, cfg):
    # rows pool across runs only under one sampling config
    cfg_hash = sha_obj(cfg)
    if os.path.exists(path):
        old = json.load(open(path))
        if old["config_hash"] != cfg_hash:
            raise StaleCache(f"{path}: rollouts sampled under a different config "
                             f"({changed_keys(old['config'], cfg)}); use a new --out")
    else:
        json.dump({"config_hash": cfg_hash, "config": cfg}, open(path, "w"), indent=2)
    return cfg_hash


def todo(prefixes, done, round_, n, overrides=None, limit=0):
    # (record, n) still to sample in this round, prefix-file order. limit picks the
    # first N prefixes before skipping done ones, so a limited rerun is a no-op.
    if limit:
        prefixes = [r for r in prefixes if overrides is None or r["prefix_id"] in overrides][:limit]
    out = []
    for rec in prefixes:
        pid = rec["prefix_id"]
        if overrides is not None:
            if pid not in overrides:
                continue
            n_here = overrides[pid]
        else:
            n_here = n
        if (pid, round_) in done:
            continue
        out.append((rec, n_here))
    return out


def done_rows(path, prefixes):
    # {(prefix_id, round): sha}; a done prefix whose tokens changed is a stale file
    by_id = {r["prefix_id"]: r for r in prefixes}
    done = {}
    for row in read_jsonl(path):
        rec = by_id.get(row["prefix_id"])
        if rec is not None and prefix_sha(rec) != row["prefix_sha"]:
            raise StaleCache(f"{row['prefix_id']}: prefix tokens differ from those {path} "
                             "was sampled on; use a new --out")
        done[(row["prefix_id"], row["round"])] = row["prefix_sha"]
    return done


def score(prefix_text, gold, text, n_gen, finish_reason):
    truncated = finish_reason == "length"
    answer, from_prefix = None, False
    if not truncated:
        # a deep prefix can already hold the \boxed{}; scored on the whole trajectory
        full = prefix_text + text
        answer = extract_boxed(full)
        from_prefix = answer is not None and full.rfind("\\boxed{") < len(prefix_text)
    return {
        "text": text,
        "n_gen": n_gen,
        "finish_reason": finish_reason,
        "answer": answer,
        "answer_from_prefix": from_prefix,
        "correct": answers_match(answer, gold) if answer is not None else False,
        "truncated": truncated,
    }


def check_prompts(llm, make_params, recs):
    # proves prompt_token_ids are what vLLM feeds for the rendered string, BOS included
    first = {}
    for r in recs:
        first.setdefault(r["prompt_text"], r)
    outs = llm.generate(list(first), make_params(n=1, temperature=0.0, max_tokens=1, seed=0))
    for (text, r), o in zip(first.items(), outs):
        got = list(o.prompt_token_ids)
        if got != r["prompt_token_ids"]:
            raise StaleCache(
                f"{r['prefix_id']}: vLLM tokenizes prompt_text to {got[:4]}... "
                f"but the prefix file has {r['prompt_token_ids'][:4]}...; rebuild prefixes")
    print(f"prompt check: {len(first)} prompts match vLLM's own tokenization", flush=True)


def run(llm, make_params, work, args, out, cfg_hash):
    tokenizer = llm.get_tokenizer()
    start, tokens, done_n, last_log = time.time(), 0, 0, 0
    total = len(work)
    with open(out, "a") as f:
        for i in range(0, total, args.chunk):
            batch = work[i : i + args.chunk]
            prompts, params = [], []
            for rec, n in batch:
                prompts.append({"prompt_token_ids": rec["prompt_token_ids"] + rec["prefix_gen_token_ids"]})
                cap = rec["max_tokens"]
                if args.max_tokens_cap:
                    cap = min(cap, args.max_tokens_cap)
                params.append(make_params(n=n, temperature=args.temp, top_p=args.top_p,
                                          max_tokens=cap,
                                          seed=prefix_seed(args.seed, rec["prefix_id"], args.round)))
            for (rec, n), p, out_ in zip(batch, params, llm.generate(prompts, params)):
                prefix_text = tokenizer.decode(rec["prefix_gen_token_ids"])
                conts = [score(prefix_text, rec["gold_answer"], o.text, len(o.token_ids),
                               o.finish_reason) for o in out_.outputs]
                tokens += sum(c["n_gen"] for c in conts)
                f.write(json.dumps({
                    "prefix_id": rec["prefix_id"],
                    "round": args.round,
                    "n": n,
                    "seed": p.seed,
                    "max_tokens": p.max_tokens,
                    "prefix_sha": prefix_sha(rec),
                    "config_hash": cfg_hash,
                    "source": rec["source"],
                    "problem_id": rec["problem_id"],
                    "trace_id": rec["trace_id"],
                    "checkpoint": rec["checkpoint"],
                    "cut_token_idx": rec["cut_token_idx"],
                    "continuations": conts,
                }) + "\n")
            f.flush()
            done_n += len(batch)
            if done_n - last_log >= LOG_EVERY or done_n == total:
                last_log = done_n
                el = time.time() - start
                rate = tokens / el if el else 0.0
                per_prefix = tokens / done_n
                eta = (total - done_n) * per_prefix / rate / 3600 if rate else float("nan")
                seqs = sum(n for _, n in work[:done_n])
                print(f"{done_n}/{total} prefixes | {rate:,.0f} gen tok/s | "
                      f"mean continuation {tokens / seqs:,.0f} tok | "
                      f"~{eta:.2f} h left for {total - done_n} prefixes", flush=True)


def main():
    args = add_args(argparse.ArgumentParser()).parse_args()
    if args.smoke:
        for k, v in SMOKE.items():
            setattr(args, k, v)
        args.out = args.out or f"{OUT_DIR}/smoke.jsonl"
    args.out = args.out or f"{OUT_DIR}/rollouts.jsonl"
    if args.n_override_file and args.round < 1:
        raise SystemExit("a top-up needs --round >= 1 so it draws a fresh seed stream")

    import vllm
    print(f"vllm {vllm.__version__}", flush=True)
    if vllm.__version__ != VLLM_VERSION and not args.allow_version_mismatch:
        raise SystemExit(f"tag-B used vllm {VLLM_VERSION}; pass --allow-version-mismatch to go on")
    from vllm import LLM, SamplingParams

    prefixes = read_jsonl(args.prefix_file)
    overrides = None
    if args.n_override_file:
        overrides = {r["prefix_id"]: int(r["n"]) for r in read_jsonl(args.n_override_file)}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    cfg = {"model": MODEL, "temp": args.temp,
           "top_p": args.top_p, "max_tokens_cap": args.max_tokens_cap}
    cfg_hash = check_config(f"{args.out}.config.json", cfg)
    done = done_rows(args.out, prefixes)
    work = todo(prefixes, done, args.round, args.n, overrides, args.limit)
    print(f"config {short(cfg_hash)}: temp {args.temp}, top_p {args.top_p}, round {args.round}; "
          f"{len(work)} prefixes to sample, {len(done)} (prefix, round) rows already in {args.out}",
          flush=True)
    if not work:
        return

    max_len = max(len(r["prompt_token_ids"]) + len(r["prefix_gen_token_ids"])
                  + (min(r["max_tokens"], args.max_tokens_cap) if args.max_tokens_cap else r["max_tokens"])
                  for r, _ in work)
    llm = LLM(model=MODEL, dtype="float16", enable_prefix_caching=True,
              max_model_len=max_len, gpu_memory_utilization=args.gpu_mem)
    check_prompts(llm, SamplingParams, [r for r, _ in work])
    run(llm, SamplingParams, work, args, args.out, cfg_hash)

    if args.smoke:
        ids = {r["prefix_id"] for r, _ in work}
        for row in read_jsonl(args.out):
            if row["prefix_id"] in ids and row["round"] == args.round:
                texts = {c["text"] for c in row["continuations"]}
                if len(texts) < 2:
                    raise AssertionError(f"{row['prefix_id']}: all {row['n']} continuations are "
                                         "identical; seeding is broken")
        print("smoke ok: continuations differ within each prefix", flush=True)


if __name__ == "__main__":
    main()
