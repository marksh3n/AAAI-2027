# n continuations per prefix with vLLM, fed token ids only. one engine request per prefix
# with SamplingParams(n) so the n samples share the prompt's KV, on a queue refilled from
# the prefix file in order (a ceiling prefix file is already priority ordered). each
# prefix's n continuations are written as one row the moment the last one finishes, so a
# stop or a preemption loses only prefixes still running; resume skips (prefix, round)
# rows already present. a problem's prefixes are admitted together: a budget or time
# stop never leaves a problem half started.
import argparse, json, os, time
from collections import Counter, defaultdict

from common import MODEL, OUT_DIR, TOP_P, prefix_seed, read_jsonl
from cache import StaleCache, changed_keys, sha_obj, short
from generate import answers_match, extract_boxed

VLLM_VERSION = "0.10.2"
LOG_SECONDS = 60
# one row per prefix holding its n samples; bumping this refuses files in an older layout
ROW_FORMAT = 3
SMOKE = {"limit": 2, "n": 2, "max_tokens_cap": 256}
# all-in $/h per GPU; --rate-usd-hr overrides
RATES = {"h100": 4.40, "h200": 4.99, "t4": 1.05, "l4": 1.00, "a10g": 1.10}
# vLLM max_num_seqs per GPU: the H200 has ~4.6M tokens of KV cache for this model and sat
# at 39% with 190 sequences in its benchmark; the H100 has about half that
MAX_SEQS = {"h100": 256, "h200": 384, "t4": 64, "l4": 64, "a10g": 64}
MARGIN_USD = 2.0  # stop starting problems this far below the budget
HARD_MARGIN_USD = 0.5  # drop in-flight work and exit this far below it
STARTUP_S = 240  # container start and model load, billed before this process's clock
PLAN_TOK_S = 5000  # throughput assumed until a real one is measured
PLAN_WARMUP_S = 180


def add_args(parser):
    parser.add_argument("--prefix-file", default=f"{OUT_DIR}/prefixes.jsonl")
    parser.add_argument("--out", default=None, help=f"default {OUT_DIR}/rollouts.jsonl")
    parser.add_argument("--temp", type=float, default=None,
                        help="tag-B generation temperature (0.7), see check_temp.py")
    parser.add_argument("--top-p", type=float, default=TOP_P)
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="dry run on the first N prefixes")
    parser.add_argument("--n-override-file", default=None,
                        help="jsonl of {prefix_id, n}: only these run, n extra samples each")
    parser.add_argument("--round", type=int, default=0)
    parser.add_argument("--max-tokens-cap", type=int, default=0, help="smoke only; 0 = none")
    parser.add_argument("--max-num-seqs", type=int, default=0, help="vLLM max_num_seqs; 0 = MAX_SEQS[--gpu]")
    parser.add_argument("--inflight", type=int, default=0,
                        help="unfinished samples handed to the engine (a prefix goes in whole, "
                             "so up to n-1 over); 0 = max-num-seqs + 64, so the scheduler always "
                             "has waiting work")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true",
                        help=f"{SMOKE}, separate output, asserts samples differ")
    parser.add_argument("--allow-version-mismatch", action="store_true")
    parser.add_argument("--gpu-mem", type=float, default=0.92)
    parser.add_argument("--tiers", type=int, nargs="+", default=None, help="only these tiers")
    parser.add_argument("--gpu", default="h100", choices=sorted(RATES), help="for the $/h rate")
    parser.add_argument("--rate-usd-hr", type=float, default=None, help="default: RATES[--gpu]")
    parser.add_argument("--budget-usd", type=float, default=0.0,
                        help=f"stop starting problems ${MARGIN_USD} before this; 0 = no budget")
    parser.add_argument("--max-minutes", type=float, default=0.0,
                        help="stop after N minutes, dropping unfinished prefixes (they rerun on resume)")
    parser.add_argument("--status", action="store_true", help="print progress and spend, no GPU")
    parser.add_argument("--ledger-key", default=None,
                        help="spend is summed over runs with this key (a Modal retry reuses it)")
    parser.add_argument("--save-token-ids", default="none", choices=["none", "sha", "full"],
                        help="per sample: nothing (ceiling run), sha256 of the generated ids, or the ids")
    parser.add_argument("--revision", default=None, help="model and tokenizer revision to pin (confirmation)")
    parser.add_argument("--model", default=MODEL, help="HF repo; another model is a replication run")
    parser.add_argument("--dtype", default="float16", help="vLLM dtype; bf16 models need bfloat16")
    parser.add_argument("--vllm-version", default=VLLM_VERSION, help="the vLLM the run must use")
    parser.add_argument("--engine-args", default="",
                        help="JSON of extra vllm.LLM keyword arguments (e.g. the Mistral formats); part of the config")
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


def todo(prefixes, done, round_, n, overrides=None, limit=0, tiers=None):
    # (record, n) for each prefix with no row yet in this round, prefix-file order. limit
    # picks the first N prefixes before skipping done ones, so a limited rerun is a no-op.
    if overrides is not None:
        prefixes = [r for r in prefixes if r["prefix_id"] in overrides]
    if tiers:
        prefixes = [r for r in prefixes if r.get("tier") in tiers]
    if limit:
        prefixes = prefixes[:limit]
    return [(r, overrides[r["prefix_id"]] if overrides is not None else n)
            for r in prefixes if (r["prefix_id"], round_) not in done]


def repair_tail(path):
    # a container killed mid-write can leave a torn last line; cut it so appends stay valid
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return 0
    with open(path, "rb+") as f:
        data = f.read()
        if data.endswith(b"\n"):
            return 0
        keep = data.rfind(b"\n") + 1
        f.truncate(keep)
        return len(data) - keep


def done_rows(path, prefixes):
    # {(prefix_id, round)}; a done prefix whose tokens changed is a stale file
    by_id = {r["prefix_id"]: r for r in prefixes}
    done = set()
    for row in read_jsonl(path):
        rec = by_id.get(row["prefix_id"])
        if rec is not None and prefix_sha(rec) != row["prefix_sha"]:
            raise StaleCache(f"{row['prefix_id']}: prefix tokens differ from those {path} "
                             "was sampled on; use a new --out")
        done.add((row["prefix_id"], row["round"]))
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


def token_record(ids, i, mode):
    # the generated ids of sample i (confirmation sources), or their sha256 (rollout banks)
    if mode == "none" or ids is None:
        return {}
    got = [int(x) for x in ids[i]]
    if mode == "full":
        return {"token_ids": got}
    return {"token_ids_sha256": sha_obj(got)}


def check_prompts(llm, make_params, recs):
    # proves prompt_token_ids are what vLLM feeds for the rendered string, BOS included. records with
    # bos == "one" (confirmation) are fed as ids, never as a string: there the check is that the
    # ids carry exactly one leading BOS and that vLLM echoes them unchanged
    one = [r for r in recs if r.get("bos") in ("one", "template")]
    if one:
        bos = llm.get_tokenizer().bos_token_id
        firsts = {}
        for r in one:
            firsts.setdefault(tuple(r["prompt_token_ids"]), r)
        outs = llm.generate([{"prompt_token_ids": list(k)} for k in firsts],
                            make_params(n=1, temperature=0.0, max_tokens=1, seed=0))
        for (ids, r), o in zip(firsts.items(), outs):
            strict = r["bos"] == "one" and (ids[0] != bos or ids[1] == bos)
            double = bos is not None and list(ids[:2]) == [bos, bos]
            if list(o.prompt_token_ids) != list(ids) or strict or double:
                raise StaleCache(f"{r['prefix_id']}: prompt ids break BOS rule {r['bos']!r} or vLLM changed them")
        print(f"prompt check: {len(firsts)} id prompts echoed unchanged", flush=True)
        recs = [r for r in recs if r.get("bos") not in ("one", "template")]
        if not recs:
            return
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


class Budget:
    # dollars spent by this run, carried across Modal retries of it through a ledger file
    # keyed by --ledger-key (one key per launch; a retry reuses it)
    def __init__(self, args, path, clock=time.time):
        self.rate, self.limit, self.clock = args.rate_usd_hr, args.budget_usd, clock
        self.path, self.key = path, args.ledger_key
        self.start = clock()
        self.prior = 0.0
        if self.key and path and os.path.exists(path):
            self.prior = sum(a["usd"] for a in json.load(open(path)).get(self.key, []))
        self.attempt = None

    def spent(self):
        return self.prior + (STARTUP_S + self.clock() - self.start) * self.rate / 3600

    def save(self):
        if not (self.key and self.path):
            return
        data = json.load(open(self.path)) if os.path.exists(self.path) else {}
        runs = data.setdefault(self.key, [])
        entry = {"started": self.start, "usd": self.spent() - self.prior, "rate": self.rate}
        if self.attempt is None:
            self.attempt = len(runs)
            runs.append(entry)
        else:
            runs[self.attempt] = entry
        tmp = f"{self.path}.tmp"
        json.dump(data, open(tmp, "w"), indent=1)
        os.replace(tmp, self.path)


def engine_stats(llm):
    # vLLM's prometheus snapshot, summed over label sets; {} when stats are off (tests)
    try:
        metrics = llm.get_metrics()
    except Exception:
        return {}
    out = defaultdict(float)
    for m in metrics:
        v = getattr(m, "value", None)
        if isinstance(v, (int, float)):
            out[m.name.split(":")[-1].removesuffix("_total")] += v
    return out


def est_tokens(rec, n):
    # planning tokens for n samples of a prefix; files without estimates plan on the cap
    return n * rec.get("est_cont_tokens", rec["max_tokens"])


def units_of(work):
    # consecutive work items of one (tier, problem): admitted all or nothing
    units = []
    for item in work:
        key = (item[0].get("tier"), item[0]["problem_id"])
        if not units or units[-1][0] != key:
            units.append((key, []))
        units[-1][1].append(item)
    return [u for _, u in units]


class Live:
    # one prefix on the engine: its n samples stream in as token deltas, each tagged with
    # its child index, and each frees its slot the moment it finishes
    def __init__(self, rec, params, est, keep_ids=False):
        self.rec, self.params, self.est = rec, params, est
        self.ids = defaultdict(list) if keep_ids else None
        self.text = defaultdict(list)
        self.tokens = Counter()
        self.finish = {}

    def add(self, c):
        # returns 1 when this delta finishes a sample
        self.text[c.index].append(c.text)
        self.tokens[c.index] += len(c.token_ids)
        if self.ids is not None:
            self.ids[c.index].extend(c.token_ids)
        if c.finish_reason is not None and c.index not in self.finish:
            self.finish[c.index] = c.finish_reason
            return 1
        return 0


def tail_seconds(max_tokens, rate, peak, max_num_seqs):
    # the budget gate's worst case for a new problem: its longest sample runs to max_tokens at one sequence's
    # share of the throughput. the share is rate / sequences running at once: the most vLLM has reported so
    # far, else max_num_seqs. a long-trace model is KV-bound well below max_num_seqs (Ministral: 30-90 of
    # 256), and dividing by 256 there overstated the tail about 3x and stopped its sources at $1.88 of $10
    return max_tokens / (rate / max(peak or max_num_seqs, 1))


def own_id(live, rid):
    # our request id for an engine output. newer vLLM may add a suffix to the id it reports; ours end in
    # "#<round>" and prefix ids hold no "#", so a key that starts the reported id is the only match
    if rid in live:
        return rid
    return next((k for k in live if rid.startswith(k)), None)


def run(llm, make_params, work, args, out, cfg_hash, budget=None, clock=time.time):
    # work: [(record, n)]. returns (stop reason or None, measured gen tok/s or None).
    # live_seqs counts unfinished samples, so the engine is refilled as soon as any sample
    # ends, not when a prefix's slowest one does
    engine, tokenizer = llm.llm_engine, llm.get_tokenizer()
    units = units_of(work)
    live, pending = {}, []  # live: request id -> Live
    next_unit, live_seqs, stop = 0, 0, None
    gen = 0  # generated tokens, counted from the deltas
    start = last_t = clock()
    last_gen = 0
    n_prefixes, n_done = len(work), 0

    peak = [0]  # most sequences vLLM has run at once so far (its own metric); 0 until the first log line

    def tok_s(now):
        return gen / (now - start) if now - start >= PLAN_WARMUP_S and gen > 0 else None

    def gate(unit, now):
        # reason not to start this problem, or None
        if args.max_minutes and now - start >= args.max_minutes * 60:
            return f"--max-minutes {args.max_minutes} reached"
        if not (budget and budget.limit):
            return None
        rate = tok_s(now) or PLAN_TOK_S
        # planned tokens still owed by prefixes admitted or promised, minus what they made so far
        committed = sum(max(x.est - sum(x.tokens.values()), 0) for x in live.values()) \
            + sum(est_tokens(r, n) for r, n in pending)
        new = sum(est_tokens(r, n) for r, n in unit)
        tail_s = tail_seconds(max(r["max_tokens"] for r, _ in unit), rate, peak[0], args.max_num_seqs)
        eta = budget.spent() + ((committed + new) / rate + tail_s) * budget.rate / 3600
        if eta >= budget.limit - MARGIN_USD:
            return (f"budget: ${budget.spent():.2f} spent + ${eta - budget.spent():.2f} to finish "
                    f"what is running and this problem >= ${budget.limit:.2f} - ${MARGIN_USD:.2f}")
        return None

    with open(out, "a") as f:
        while True:
            now = clock()
            while live_seqs < args.inflight:
                if not pending:
                    if stop or next_unit == len(units):
                        break
                    stop = gate(units[next_unit], now)
                    if stop:
                        print(f"not starting new problems: {stop}", flush=True)
                        break
                    pending = list(units[next_unit])
                    next_unit += 1
                rec, n = pending.pop(0)
                cap = rec["max_tokens"]
                if args.max_tokens_cap:
                    cap = min(cap, args.max_tokens_cap)
                p = make_params(n=n, temperature=args.temp, top_p=args.top_p, max_tokens=cap,
                                seed=prefix_seed(args.seed, rec["prefix_id"], args.round))
                rid = f"{rec['prefix_id']}#{args.round}"
                engine.add_request(rid, {"prompt_token_ids": rec["prompt_token_ids"]
                                         + rec["prefix_gen_token_ids"]}, p)
                live[rid] = Live(rec, p, est_tokens(rec, n), getattr(args, "save_token_ids", "none") != "none")
                live_seqs += n
            if not live:
                break

            outs = engine.step()
            if not outs and not getattr(engine, "has_unfinished_requests", lambda: True)():
                # the engine has nothing left but these never came back: fail instead of spinning
                raise RuntimeError(f"vLLM is idle but {len(live)} prefixes got no final output "
                                   f"(request ids such as {list(live)[:2]})")
            for o in outs:
                rid = own_id(live, o.request_id)
                if rid is None:
                    continue
                slot = live[rid]
                for c in o.outputs:
                    gen += len(c.token_ids)
                    live_seqs -= slot.add(c)
                if not o.finished:
                    continue
                del live[rid]
                rec, p = slot.rec, slot.params
                live_seqs -= p.n - len(slot.finish)  # never leak a slot
                pid = rec["prefix_id"]
                prefix_text = tokenizer.decode(rec["prefix_gen_token_ids"])
                # vLLM gives child i of a seeded n>1 request seed + i
                samples = [{"sample_idx": i, "seed": p.seed + i,
                            **score(prefix_text, rec["gold_answer"], "".join(slot.text[i]),
                                    slot.tokens[i], slot.finish.get(i)),
                            **token_record(slot.ids, i, getattr(args, "save_token_ids", "none"))}
                           for i in range(p.n)]
                row = {
                    "prefix_id": pid,
                    "round": args.round,
                    "n": p.n,
                    "seed": p.seed,
                    "max_tokens": p.max_tokens,
                    "prefix_sha": prefix_sha(rec),
                    "config_hash": cfg_hash,
                    "source": rec["source"],
                    "tier": rec.get("tier"),
                    "problem_id": rec["problem_id"],
                    "trace_id": rec["trace_id"],
                    "checkpoint": rec["checkpoint"],
                    "cut_token_idx": rec["cut_token_idx"],
                    "samples": samples,
                }
                f.write(json.dumps(row) + "\n")
                f.flush()
                n_done += 1

            now = clock()
            hard = None
            if args.max_minutes and now - start >= args.max_minutes * 60:
                hard = f"--max-minutes {args.max_minutes} reached"
            elif budget and budget.limit and budget.spent() >= budget.limit - HARD_MARGIN_USD:
                hard = f"hard budget cap: ${budget.spent():.2f} spent of ${budget.limit:.2f}"
            if now - last_t >= LOG_SECONDS or hard or not live:
                stats = engine_stats(llm)
                peak[0] = max(peak[0], int(stats.get("num_requests_running", 0)))
                window = (gen - last_gen) / (now - last_t) if now > last_t else 0.0
                kv = stats.get("kv_cache_usage_perc", stats.get("gpu_cache_usage_perc", float("nan")))
                money = ""
                if budget:
                    money = f" | ${budget.spent():.2f} spent" + (f" of ${budget.limit:.2f}" if budget.limit else "")
                print(f"[{(now - start) / 60:5.1f} min] {window:,.0f} gen tok/s now, "
                      f"{gen / max(now - start, 1e-9):,.0f} avg | "
                      f"running {stats.get('num_requests_running', live_seqs):.0f} "
                      f"waiting {stats.get('num_requests_waiting', 0):.0f} | KV {kv:.0%} | "
                      f"preemptions {stats.get('num_preemptions', 0):.0f} | "
                      f"prefixes {n_done}/{n_prefixes} written, {len(live)} running | "
                      f"{gen / 1e6:.2f}M tokens{money}", flush=True)
                last_t, last_gen = now, gen
                if budget:
                    budget.save()
            if hard:
                # finished prefixes are all written; the ones still running are re-run on resume
                if live and hasattr(engine, "abort_request"):
                    engine.abort_request(list(live))
                print(f"stopping now: {hard}; dropping {len(live)} prefixes still running", flush=True)
                stop = hard
                break
    elapsed = clock() - start
    return stop, (gen / elapsed if elapsed > 0 and gen > 0 else None)


def progress(prefixes, done, round_, n, tok_s, rate):
    # per tier: prefixes and planned tokens done / left, and what the rest would cost
    by_tier = defaultdict(Counter)
    for r in prefixes:
        t = r.get("tier", "-")
        key = "done" if (r["prefix_id"], round_) in done else "left"
        by_tier[t][key] += 1
        by_tier[t][f"{key}_tok"] += est_tokens(r, n)
    lines, cum = [], 0.0
    for t in sorted(by_tier, key=str):
        c = by_tier[t]
        cum += c["left_tok"]
        line = (f"  tier {t}: {c['done']} done, {c['left']} left "
                f"(~{c['left_tok'] / 1e6:.1f}M tokens left)")
        if tok_s:
            h = c["left_tok"] / tok_s / 3600
            hc = cum / tok_s / 3600
            line += f" | this tier ~{h:.2f} h ${h * rate:.2f}; through it ~{hc:.2f} h ${hc * rate:.2f}"
        lines.append(line)
    return "\n".join(lines)


def main():
    args = add_args(argparse.ArgumentParser()).parse_args()
    if args.smoke:
        for k, v in SMOKE.items():
            setattr(args, k, v)
        args.out = args.out or f"{OUT_DIR}/smoke.jsonl"
    args.out = args.out or f"{OUT_DIR}/rollouts.jsonl"
    if args.rate_usd_hr is None:
        args.rate_usd_hr = RATES[args.gpu]
    if not args.max_num_seqs:
        args.max_num_seqs = MAX_SEQS[args.gpu]
    if not args.inflight:
        args.inflight = args.max_num_seqs + 64
    if args.n_override_file and args.round < 1:
        raise SystemExit("a top-up needs --round >= 1 so it draws a fresh seed stream")

    if args.status:
        prefixes = read_jsonl(args.prefix_file)
        done = done_rows(args.out, prefixes)
        print(f"{len(done)}/{len(prefixes)} prefixes done in {args.out}")
        print(progress(prefixes, done, args.round, args.n, None, args.rate_usd_hr))
        ledger = f"{args.out}.ledger.json"
        for key, runs in (json.load(open(ledger)).items() if os.path.exists(ledger) else []):
            print(f"  spend {key}: ${sum(r['usd'] for r in runs):.2f} over {len(runs)} attempt(s)")
        return

    if args.temp is None:
        raise SystemExit("--temp is required: the tag-B sampling temperature, 0.7")
    import vllm
    print(f"vllm {vllm.__version__}", flush=True)
    if vllm.__version__ != args.vllm_version and not args.allow_version_mismatch:
        raise SystemExit(f"this run needs vllm {args.vllm_version}; pass --allow-version-mismatch to go on")
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import RequestOutputKind

    def make_params(**kw):
        # each sample streams its token deltas, so a finished sample is seen at once
        sp = SamplingParams(**kw)
        sp.output_kind = RequestOutputKind.DELTA
        return sp

    prefixes = read_jsonl(args.prefix_file)
    overrides = None
    if args.n_override_file:
        overrides = {r["prefix_id"]: int(r["n"]) for r in read_jsonl(args.n_override_file)}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    cfg = {"model": args.model, "temp": args.temp, "top_p": args.top_p,
           "max_tokens_cap": args.max_tokens_cap, "row_format": ROW_FORMAT}
    # only when set, so the ceiling run's config hash is unchanged
    if args.save_token_ids != "none":
        cfg["save_token_ids"] = args.save_token_ids
    if args.dtype != "float16":
        cfg["dtype"] = args.dtype
    engine_args = json.loads(args.engine_args) if args.engine_args else {}
    if engine_args:
        cfg["engine_args"] = engine_args
    if "VLLM_USE_FLASHINFER_SAMPLER" in os.environ:
        # which top-p kernel draws the samples; only recorded when set, so older config hashes are unchanged
        cfg["flashinfer_sampler"] = os.environ["VLLM_USE_FLASHINFER_SAMPLER"]
    if args.revision:
        # confirmation: sources and banks must match on all of this, so the engine goes in too
        cfg["revision"] = args.revision
        cfg["vllm"] = vllm.__version__
    cfg_hash = check_config(f"{args.out}.config.json", cfg)
    torn = repair_tail(args.out)
    if torn:
        print(f"cut a torn {torn}-byte last line from {args.out}", flush=True)
    done = done_rows(args.out, prefixes)
    work = todo(prefixes, done, args.round, args.n, overrides, args.limit, args.tiers)
    budget = Budget(args, f"{args.out}.ledger.json")
    print(f"config {short(cfg_hash)}: temp {args.temp}, top_p {args.top_p}, n {args.n}, round "
          f"{args.round}; {len(work)} prefixes to sample, {len(done)} already in {args.out}; "
          f"{args.gpu} at ${args.rate_usd_hr}/h, budget "
          f"{f'${args.budget_usd:.2f}' if args.budget_usd else 'none'}"
          + (f", ${budget.prior:.2f} already spent by earlier attempts" if budget.prior else ""),
          flush=True)
    print(progress(prefixes, done, args.round, args.n, None, args.rate_usd_hr), flush=True)
    if not work:
        return

    max_len = max(len(r["prompt_token_ids"]) + len(r["prefix_gen_token_ids"])
                  + (min(r["max_tokens"], args.max_tokens_cap) if args.max_tokens_cap else r["max_tokens"])
                  for r, _ in work)
    pin = {"revision": args.revision, "tokenizer_revision": args.revision} if args.revision else {}
    llm = LLM(model=args.model, dtype=args.dtype, max_model_len=max_len, seed=args.seed, **pin,
              enable_prefix_caching=True, enable_chunked_prefill=True,
              gpu_memory_utilization=args.gpu_mem, max_num_seqs=args.max_num_seqs,
              disable_log_stats=False, **engine_args)
    check_prompts(llm, SamplingParams, [r for r, _ in work])
    stop, tok_s = run(llm, make_params, work, args, args.out, cfg_hash, budget)
    budget.save()

    done = done_rows(args.out, prefixes)
    print(f"\n{'stopped: ' + stop if stop else 'all requested prefixes done'}")
    print(f"measured {tok_s or float('nan'):,.0f} gen tok/s; ${budget.spent():.2f} spent "
          f"this run (incl. ~{STARTUP_S // 60} min startup)")
    print("progress, projected from tag-B lengths at the measured rate:")
    print(progress(prefixes, done, args.round, args.n, tok_s, args.rate_usd_hr), flush=True)

    if args.smoke:
        for row in read_jsonl(args.out):
            if row["round"] == args.round and len({x["text"] for x in row["samples"]}) < 2:
                raise AssertionError(f"{row['prefix_id']}: its continuations are all identical; "
                                     "seeding is broken")
        print("smoke ok: continuations differ within each prefix", flush=True)


if __name__ == "__main__":
    main()
