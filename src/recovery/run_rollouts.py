# n continuations per prefix with vLLM, fed token ids only. one request and one output
# row per continuation, written the moment it finishes, so a disconnect loses only the
# continuations still running. resume re-requests only the missing (prefix, round, sample).
import argparse, json, os, time

from common import MODEL, OUT_DIR, TOP_P, answers_match, extract_boxed, prefix_seed, read_jsonl
from cache import StaleCache, changed_keys, sha_obj, short

VLLM_VERSION = "0.10.2"
LOG_SECONDS = 120
# rows are one continuation each; bumping this refuses files in an older layout
ROW_FORMAT = 2
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
    parser.add_argument("--inflight", type=int, default=256,
                        help="continuations on the engine at once (vLLM's max_num_seqs is 256)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true",
                        help=f"{SMOKE}, separate output, asserts samples differ")
    parser.add_argument("--allow-version-mismatch", action="store_true")
    parser.add_argument("--gpu-mem", type=float, default=0.90)
    return parser


def prefix_sha(rec):
    # binds an output row to the exact tokens it continued
    return sha_obj([rec["prompt_token_ids"], rec["prefix_gen_token_ids"]])


def sample_seed(base, prefix_id, round_, idx):
    # what vLLM gives child idx of an n>1 request seeded prefix_seed
    return prefix_seed(base, prefix_id, round_) + idx


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
    # (record, sample idx) still to sample in this round, prefix-file order. limit picks
    # the first N prefixes before skipping done ones, so a limited rerun is a no-op.
    if overrides is not None:
        prefixes = [r for r in prefixes if r["prefix_id"] in overrides]
    if limit:
        prefixes = prefixes[:limit]
    out = []
    for rec in prefixes:
        pid = rec["prefix_id"]
        n_here = overrides[pid] if overrides is not None else n
        out += [(rec, i) for i in range(n_here) if (pid, round_, i) not in done]
    return out


def done_rows(path, prefixes):
    # {(prefix_id, round, sample_idx)}; a done prefix whose tokens changed is a stale file
    by_id = {r["prefix_id"]: r for r in prefixes}
    done = set()
    for row in read_jsonl(path):
        rec = by_id.get(row["prefix_id"])
        if rec is not None and prefix_sha(rec) != row["prefix_sha"]:
            raise StaleCache(f"{row['prefix_id']}: prefix tokens differ from those {path} "
                             "was sampled on; use a new --out")
        done.add((row["prefix_id"], row["round"], row["sample_idx"]))
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
    # up to --inflight continuations on the engine. requests stream token deltas, so the
    # throughput log counts every generated token, not only finished continuations.
    engine, tokenizer = llm.llm_engine, llm.get_tokenizer()
    queue, live = list(work), {}  # live: request id -> [rec, idx, params, text parts, n tokens]
    prefix_text = {}
    total, done_n, tokens = len(work), 0, 0
    start = last_t = time.time()
    last_tokens, done_tokens = 0, 0
    with open(out, "a") as f:
        while queue or live:
            while queue and len(live) < args.inflight:
                rec, idx = queue.pop(0)
                cap = rec["max_tokens"]
                if args.max_tokens_cap:
                    cap = min(cap, args.max_tokens_cap)
                p = make_params(n=1, temperature=args.temp, top_p=args.top_p, max_tokens=cap,
                                seed=sample_seed(args.seed, rec["prefix_id"], args.round, idx))
                rid = f"{rec['prefix_id']}#{args.round}#{idx}"
                engine.add_request(rid, {"prompt_token_ids": rec["prompt_token_ids"]
                                         + rec["prefix_gen_token_ids"]}, p)
                live[rid] = [rec, idx, p, [], 0]
            for o in engine.step():
                slot = live[o.request_id]
                c = o.outputs[0] if o.outputs else None
                if c is not None:
                    slot[3].append(c.text)
                    slot[4] += len(c.token_ids)
                    tokens += len(c.token_ids)
                if not o.finished:
                    continue
                rec, idx, p, parts, n_gen = live.pop(o.request_id)
                pid = rec["prefix_id"]
                if pid not in prefix_text:
                    prefix_text[pid] = tokenizer.decode(rec["prefix_gen_token_ids"])
                row = {
                    "prefix_id": pid,
                    "round": args.round,
                    "sample_idx": idx,
                    "seed": p.seed,
                    "max_tokens": p.max_tokens,
                    "prefix_sha": prefix_sha(rec),
                    "config_hash": cfg_hash,
                    "source": rec["source"],
                    "problem_id": rec["problem_id"],
                    "trace_id": rec["trace_id"],
                    "checkpoint": rec["checkpoint"],
                    "cut_token_idx": rec["cut_token_idx"],
                    **score(prefix_text[pid], rec["gold_answer"], "".join(parts), n_gen,
                            c.finish_reason if c is not None else None),
                }
                f.write(json.dumps(row) + "\n")
                f.flush()
                done_n += 1
                done_tokens += n_gen
            now = time.time()
            if now - last_t >= LOG_SECONDS or (done_n == total and not live and not queue):
                window = (tokens - last_tokens) / (now - last_t) if now > last_t else 0.0
                overall = tokens / (now - start) if now > start else 0.0
                # remaining work guessed from finished continuations; early on, short ones
                # finish first, so this runs low until the long ones start landing
                mean_len = done_tokens / done_n if done_n else float("nan")
                left = max((total - done_n) * mean_len - (tokens - done_tokens), 0.0)
                eta = left / overall / 3600 if overall and done_n else float("nan")
                print(f"{done_n}/{total} continuations | {window:,.0f} gen tok/s now, "
                      f"{overall:,.0f} overall | {len(live)} running | {tokens / 1e6:.2f}M tokens | "
                      f"mean finished length {mean_len:,.0f} | ~{eta:.2f} h left", flush=True)
                last_t, last_tokens = now, tokens


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
    from vllm.sampling_params import RequestOutputKind

    def make_params(**kw):
        # stream token deltas so every generated token is counted as it lands
        sp = SamplingParams(**kw)
        sp.output_kind = RequestOutputKind.DELTA
        return sp

    prefixes = read_jsonl(args.prefix_file)
    overrides = None
    if args.n_override_file:
        overrides = {r["prefix_id"]: int(r["n"]) for r in read_jsonl(args.n_override_file)}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    cfg = {"model": MODEL, "temp": args.temp, "top_p": args.top_p,
           "max_tokens_cap": args.max_tokens_cap, "row_format": ROW_FORMAT}
    cfg_hash = check_config(f"{args.out}.config.json", cfg)
    done = done_rows(args.out, prefixes)
    work = todo(prefixes, done, args.round, args.n, overrides, args.limit)
    print(f"config {short(cfg_hash)}: temp {args.temp}, top_p {args.top_p}, round {args.round}; "
          f"{len(work)} continuations to sample over {len({r['prefix_id'] for r, _ in work})} "
          f"prefixes, {len(done)} already in {args.out}", flush=True)
    if not work:
        return

    max_len = max(len(r["prompt_token_ids"]) + len(r["prefix_gen_token_ids"])
                  + (min(r["max_tokens"], args.max_tokens_cap) if args.max_tokens_cap else r["max_tokens"])
                  for r, _ in work)
    llm = LLM(model=MODEL, dtype="float16", enable_prefix_caching=True,
              max_model_len=max_len, gpu_memory_utilization=args.gpu_mem)
    check_prompts(llm, SamplingParams, [r for r, _ in work])
    run(llm, make_params, work, args, args.out, cfg_hash)

    if args.smoke:
        texts = {}
        for row in read_jsonl(args.out):
            if row["round"] == args.round:
                texts.setdefault(row["prefix_id"], set()).add(row["text"])
        for pid in {r["prefix_id"] for r, _ in work}:
            if len(texts.get(pid, ())) < 2:
                raise AssertionError(f"{pid}: its continuations are all identical; seeding is broken")
        print("smoke ok: continuations differ within each prefix", flush=True)


if __name__ == "__main__":
    main()
