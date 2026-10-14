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
ROW_FORMAT = 4
SMOKE = {"limit": 2, "n": 2, "max_tokens_cap": 256}
# all-in $/h per GPU; --rate-usd-hr overrides
RATES = {"h100": 4.40, "h200": 4.99, "t4": 1.05}
# vLLM max_num_seqs per GPU: the H200 has ~4.6M tokens of KV cache for this model and sat
# at 39% with 190 sequences in its benchmark; the H100 has about half that
MAX_SEQS = {"h100": 256, "h200": 384, "t4": 64}
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
    return parser


def prefix_sha(rec):
    # binds an output row to the exact tokens it continued
    return sha_obj([rec["prompt_token_ids"], rec["prefix_gen_token_ids"]])


def prefix_definition_sha(rec):
    # Token equality alone does not establish an identical experiment: the
    # continuation budget and gold answer determine its sampling and grading.
    return sha_obj({"prefix_sha": prefix_sha(rec), "max_tokens": rec["max_tokens"],
                    "gold_answer": rec["gold_answer"]})


def validate_prefixes(prefixes):
    """Check the IDs actually fed to vLLM, without certifying historical tokens."""
    seen = set()
    for rec in prefixes:
        pid = rec.get("prefix_id")
        if not isinstance(pid, str) or not pid or pid in seen:
            raise StaleCache(f"Missing/duplicate prefix_id: {pid!r}")
        seen.add(pid)
        for name in ("prompt_token_ids", "prefix_gen_token_ids"):
            ids = rec.get(name)
            if not isinstance(ids, list) or any(type(i) is not int or i < 0 for i in ids):
                raise StaleCache(f"{pid}: invalid {name}")
        if not rec["prompt_token_ids"]:
            raise StaleCache(f"{pid}: empty prompt")
        cut = rec.get("cut_token_idx")
        if type(cut) is not int or cut != len(rec["prefix_gen_token_ids"]):
            raise StaleCache(f"{pid}: cut_token_idx must equal the number of kept generated tokens")
        ckpt = rec.get("checkpoint")
        if not isinstance(ckpt, str) or not ckpt:
            raise StaleCache(f"{pid}: missing checkpoint")
        if ckpt.startswith("p") and ckpt[1:].isdigit() and int(ckpt[1:]) != cut:
            raise StaleCache(f"{pid}: fixed checkpoint disagrees with its token cut")
        if ckpt == "t0" and (cut or rec.get("trace_id") is not None):
            raise StaleCache(f"{pid}: t0 must have zero generated tokens and no trace_id")
        if ckpt != "t0" and not isinstance(rec.get("trace_id"), str):
            raise StaleCache(f"{pid}: trace checkpoint needs a trace_id")
        if rec.get("problem_id") is None or type(rec.get("max_tokens")) is not int or rec["max_tokens"] <= 0:
            raise StaleCache(f"{pid}: invalid problem_id or continuation budget")
        if not isinstance(rec.get("gold_answer"), str) or not rec["gold_answer"].strip():
            raise StaleCache(f"{pid}: missing or invalid gold answer for continuation grading")
    return prefixes


def read_prefixes(path):
    # Unlike append-only rollout output, the completed prefix definition file must
    # not silently lose an incomplete last record.
    with open(path) as handle:
        prefixes = [json.loads(line) for line in handle if line.strip()]
    return validate_prefixes(prefixes)


def check_paths(prefix_path, output_path):
    # Sidecars must not replace either the prefix file or the output file.
    names = [prefix_path, output_path, f"{output_path}.config.json", f"{output_path}.ledger.json"]
    resolved = [os.path.realpath(p) for p in names]
    if len(set(resolved)) != len(resolved):
        raise StaleCache("Prefix, rollout, config and ledger paths must be distinct")


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
        # A valid final JSON record without newline is complete, not a torn write.
        try:
            json.loads(data[keep:])
        except (json.JSONDecodeError, UnicodeDecodeError):
            # Preserve the fragment for investigation before recovering appendability.
            backup = f"{path}.torn_tail.{__import__('hashlib').sha256(data[keep:]).hexdigest()[:16]}.bin"
            if not os.path.exists(backup):
                with open(backup, "wb") as audit:
                    audit.write(data[keep:])
        else:
            f.seek(0, os.SEEK_END)
            f.write(b"\n")
            return 0
        f.truncate(keep)
        return len(data) - keep


def done_rows(path, prefixes, config_hash=None, base_seed=None, max_tokens_cap=None):
    # {(prefix_id, round)}. A configured resume requires the new budget/gold
    # binding; config_hash=None permits read-only legacy status inspection only.
    by_id = {r["prefix_id"]: r for r in validate_prefixes(prefixes)}
    done, fingerprints = set(), {}
    for row in read_jsonl(path):
        rec = by_id.get(row.get("prefix_id"))
        if rec is None:
            raise StaleCache(f"Unknown prefix in existing rollouts: {row.get('prefix_id')!r}")
        if prefix_sha(rec) != row.get("prefix_sha"):
            raise StaleCache(f"{row['prefix_id']}: prefix tokens differ from those {path} "
                             "was sampled on; use a new --out")
        definition = row.get("prefix_definition_sha")
        if definition is None:
            if config_hash is not None:
                raise StaleCache(f"{row['prefix_id']}: legacy row lacks its continuation-budget/"
                                 "gold-answer binding; use a new --out, do not retrofit a manifest")
        elif definition != prefix_definition_sha(rec):
            raise StaleCache(f"{row['prefix_id']}: continuation budget or gold answer changed; "
                             "use a new --out")
        if definition is not None and row.get("gold_answer") != rec["gold_answer"]:
            raise StaleCache(f"{row['prefix_id']}: saved gold answer conflicts with prefix file")
        if max_tokens_cap is not None:
            expected_cap = min(rec["max_tokens"], max_tokens_cap) if max_tokens_cap else rec["max_tokens"]
            if type(row.get("max_tokens")) is not int or row["max_tokens"] != expected_cap:
                raise StaleCache(f"{row['prefix_id']}: saved effective continuation budget differs")
        for field in ("problem_id", "trace_id", "checkpoint", "cut_token_idx"):
            if row.get(field) != rec.get(field):
                raise StaleCache(f"{row['prefix_id']}: saved {field} conflicts with prefix file")
        if type(row.get("round")) is not int or row["round"] < 0:
            raise StaleCache(f"{row['prefix_id']}: invalid round")
        if config_hash is not None and row.get("config_hash") != config_hash:
            raise StaleCache(f"{row['prefix_id']}: saved sampling config differs")
        if base_seed is not None and row.get("seed") != prefix_seed(base_seed, row["prefix_id"], row["round"]):
            raise StaleCache(f"{row['prefix_id']}: saved seed differs from requested base seed")
        samples, n = row.get("samples"), row.get("n")
        if not isinstance(samples, list) or type(n) is not int or n <= 0 or len(samples) != n:
            raise StaleCache(f"{row['prefix_id']}: incomplete sample group")
        if any(not isinstance(s, dict) or type(s.get("sample_idx")) is not int for s in samples):
            raise StaleCache(f"{row['prefix_id']}: invalid sample index")
        if sorted(s["sample_idx"] for s in samples) != list(range(n)):
            raise StaleCache(f"{row['prefix_id']}: duplicate/missing sample index")
        if any(type(s.get("correct")) is not bool or type(s.get("truncated")) is not bool for s in samples):
            raise StaleCache(f"{row['prefix_id']}: nonboolean outcome/truncation label")
        if "prompt_token_ids" in row or "prefix_gen_token_ids" in row:
            if row.get("prompt_token_ids") != rec["prompt_token_ids"] or row.get("prefix_gen_token_ids") != rec["prefix_gen_token_ids"]:
                raise StaleCache(f"{row['prefix_id']}: recorded input IDs differ")
        key = (row["prefix_id"], row["round"])
        fingerprint = sha_obj(row)
        if key in fingerprints and fingerprints[key] != fingerprint:
            raise StaleCache(f"Conflicting repeated rollout row: {key}")
        fingerprints[key] = fingerprint
        done.add(key)
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
    # Sanity check the explicit token inputs, including intentional historical BOS
    # choices. Re-tokenizing prompt_text here would override an explicit policy.
    first = {}
    for r in recs:
        first.setdefault(tuple(r["prompt_token_ids"]), r)
    outs = llm.generate([{"prompt_token_ids": list(ids)} for ids in first],
                        make_params(n=1, temperature=0.0, max_tokens=1, seed=0))
    if len(outs) != len(first):
        raise StaleCache("vLLM prompt check returned an incomplete result")
    for r, o in zip(first.values(), outs):
        got = list(o.prompt_token_ids)
        if got != r["prompt_token_ids"]:
            raise StaleCache(
                f"{r['prefix_id']}: vLLM reports prompt IDs {got[:4]}... "
                f"but the prefix file has {r['prompt_token_ids'][:4]}...; rebuild prefixes")
    print(f"prompt check: {len(first)} explicit prompts match vLLM inputs; historical generation remains separately audited", flush=True)


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
    def __init__(self, rec, params, est):
        self.rec, self.params, self.est = rec, params, est
        self.text = defaultdict(list)
        self.tokens = Counter()
        self.finish = {}

    def add(self, c):
        # returns 1 when this delta finishes a sample
        self.text[c.index].append(c.text)
        self.tokens[c.index] += len(c.token_ids)
        if c.finish_reason is not None and c.index not in self.finish:
            self.finish[c.index] = c.finish_reason
            return 1
        return 0


def run(llm, make_params, work, args, out, cfg_hash, budget=None, clock=time.time):
    # work: [(record, n)]. returns (stop reason or None, measured gen tok/s or None).
    # live_seqs counts unfinished samples, so the engine is refilled as soon as any sample
    # ends, not when a prefix's slowest one does
    validate_prefixes([r for r, _ in work])
    engine, tokenizer = llm.llm_engine, llm.get_tokenizer()
    units = units_of(work)
    live, pending = {}, []  # live: request id -> Live
    next_unit, live_seqs, stop = 0, 0, None
    gen = 0  # generated tokens, counted from the deltas
    start = last_t = clock()
    last_gen = 0
    n_prefixes, n_done = len(work), 0

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
        per_seq = rate / max(live_seqs, args.max_num_seqs, 1)
        tail_s = max(r["max_tokens"] for r, _ in unit) / per_seq
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
                live[rid] = Live(rec, p, est_tokens(rec, n))
                live_seqs += n
            if not live:
                break

            for o in engine.step():
                slot = live.get(o.request_id)
                if slot is None:
                    continue
                for c in o.outputs:
                    gen += len(c.token_ids)
                    live_seqs -= slot.add(c)
                if not o.finished:
                    continue
                del live[o.request_id]
                rec, p = slot.rec, slot.params
                live_seqs -= p.n - len(slot.finish)  # never leak a slot
                pid = rec["prefix_id"]
                prefix_text = tokenizer.decode(rec["prefix_gen_token_ids"])
                # vLLM gives child i of a seeded n>1 request seed + i
                samples = [{"sample_idx": i, "seed": p.seed + i,
                            **score(prefix_text, rec["gold_answer"], "".join(slot.text[i]),
                                    slot.tokens[i], slot.finish.get(i))}
                           for i in range(p.n)]
                row = {
                    "prefix_id": pid,
                    "round": args.round,
                    "n": p.n,
                    "seed": p.seed,
                    "max_tokens": p.max_tokens,
                    "prefix_sha": prefix_sha(rec),
                    "prefix_definition_sha": prefix_definition_sha(rec),
                    "gold_answer": rec["gold_answer"],
                    "prompt_token_ids": rec["prompt_token_ids"],
                    "prefix_gen_token_ids": rec["prefix_gen_token_ids"],
                    "input_capture": "recorded_at_rollout_submission",
                    "source_token_provenance": rec.get("token_input_provenance", "legacy_unverified"),
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
    check_paths(args.prefix_file, args.out)
    if args.rate_usd_hr is None:
        args.rate_usd_hr = RATES[args.gpu]
    if not args.max_num_seqs:
        args.max_num_seqs = MAX_SEQS[args.gpu]
    if not args.inflight:
        args.inflight = args.max_num_seqs + 64
    if args.n_override_file and args.round < 1:
        raise SystemExit("a top-up needs --round >= 1 so it draws a fresh seed stream")

    if args.status:
        prefixes = read_prefixes(args.prefix_file)
        done = done_rows(args.out, prefixes)
        print(f"{len(done)}/{len(prefixes)} prefixes done in {args.out}")
        print("Status only: legacy rows can be counted without certifying their budget/gold-answer binding.")
        print(progress(prefixes, done, args.round, args.n, None, args.rate_usd_hr))
        ledger = f"{args.out}.ledger.json"
        for key, runs in (json.load(open(ledger)).items() if os.path.exists(ledger) else []):
            print(f"  spend {key}: ${sum(r['usd'] for r in runs):.2f} over {len(runs)} attempt(s)")
        return

    if args.temp is None:
        raise SystemExit("--temp is required: the tag-B sampling temperature, 0.7")
    import vllm
    print(f"vllm {vllm.__version__}", flush=True)
    if vllm.__version__ != VLLM_VERSION and not args.allow_version_mismatch:
        raise SystemExit(f"tag-B used vllm {VLLM_VERSION}; pass --allow-version-mismatch to go on")
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import RequestOutputKind

    def make_params(**kw):
        # each sample streams its token deltas, so a finished sample is seen at once
        sp = SamplingParams(**kw)
        sp.output_kind = RequestOutputKind.DELTA
        return sp

    prefixes = read_prefixes(args.prefix_file)
    overrides = None
    if args.n_override_file:
        overrides = {r["prefix_id"]: int(r["n"]) for r in read_jsonl(args.n_override_file)}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    cfg = {"model": MODEL, "temp": args.temp, "top_p": args.top_p,
           "max_tokens_cap": args.max_tokens_cap, "row_format": ROW_FORMAT}
    cfg_hash = check_config(f"{args.out}.config.json", cfg)
    torn = repair_tail(args.out)
    if torn:
        print(f"cut a torn {torn}-byte last line from {args.out}", flush=True)
    done = done_rows(args.out, prefixes, cfg_hash, args.seed, args.max_tokens_cap)
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
    llm = LLM(model=MODEL, dtype="float16", max_model_len=max_len, seed=args.seed,
              enable_prefix_caching=True, enable_chunked_prefill=True,
              gpu_memory_utilization=args.gpu_mem, max_num_seqs=args.max_num_seqs,
              disable_log_stats=False)
    check_prompts(llm, SamplingParams, [r for r, _ in work])
    stop, tok_s = run(llm, make_params, work, args, args.out, cfg_hash, budget)
    budget.save()

    done = done_rows(args.out, prefixes, cfg_hash, args.seed, args.max_tokens_cap)
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
