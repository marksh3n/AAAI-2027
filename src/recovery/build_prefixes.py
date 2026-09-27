# prefixes of tag-B traces at fixed checkpoints, one record per prefix.
# the schema is generic so edited / injected prefixes can share the runner later.
import argparse, bisect, json, os
from collections import Counter, defaultdict
import numpy as np

from common import (MAX_NEW, OUT_DIR, canonical, encode_trace, load_tokenizer, out_path,
                    prompt_ids, build_prompt)

FRACTIONS = {"f25": 0.25, "f50": 0.50, "f75": 0.75}
THINK_END = "</think>"
# ceiling design: the probe_cmp checkpoints, cut at exactly that many tokens (as the probes
# read them, no step snapping), plus one t0 restart per problem
CEILING_POSITIONS = (512, 2048, 4096)


def cut_points(text, offsets):
    # token counts k where ids[:k] ends a step: the last kept token closes a "\n\n"
    # (".\n\n" counts), or token k is </think>. never mid-number.
    ks = [k for k in range(1, len(offsets) + 1) if text[: offsets[k - 1][1]].endswith("\n\n")]
    think = text.find(THINK_END)
    if think >= 0:
        ks += [k for k, (a, _) in enumerate(offsets) if a == think]
    return sorted(set(ks))


def snap(points, requested):
    # largest step boundary <= requested, 0 when there is none
    i = bisect.bisect_right(points, requested)
    return points[i - 1] if i else 0


def token_at(offsets, char):
    # tokens that end at or before char, i.e. the cut that stops right before it
    ends = [b for _, b in offsets]
    return bisect.bisect_right(ends, char)


def paragraphs(text):
    pos = 0
    for para in text.split("\n\n"):
        yield pos, para
        pos += len(para) + 2


def first_mention(text, answer, match):
    # char start of the first paragraph whose math matches the trace's own final answer
    for pos, para in paragraphs(text):
        if para.strip() and match(answer, para):
            return pos
    return None


def pre_final(text):
    # start of the paragraph holding the final \boxed{}, or </think> if that is earlier
    boxed = text.rfind("\\boxed{")
    if boxed < 0:
        return None
    start = text.rfind("\n\n", 0, boxed)
    start = 0 if start < 0 else start + 2
    think = text.find(THINK_END)
    return think if 0 <= think < start else start


def math_match(answer, para):
    from math_verify import parse, verify
    return bool(verify(parse("$" + answer + "$"), parse(para)))


def pass_bin(rate):
    return "low" if rate <= 1 / 3 else "high" if rate >= 2 / 3 else "mid"


def select(traces, eligible, n_problems, seed):
    # mixed problems stratified by level x pass rate, drawn round-robin across strata,
    # then one eligible correct and one eligible wrong trace per problem
    by_problem = defaultdict(list)
    for t in traces:
        by_problem[t["problem_id"]].append(t)
    pools = defaultdict(list)
    for pid, ts in sorted(by_problem.items()):
        ok = [t for t in ts if eligible[t["trace_id"]]]
        if any(t["is_correct"] for t in ok) and any(not t["is_correct"] for t in ok):
            rate = sum(t["is_correct"] for t in ts) / len(ts)
            pools[(ts[0]["level"], pass_bin(rate))].append(pid)
    rng = np.random.default_rng(seed)
    pools = {k: list(rng.permutation(v)) for k, v in sorted(pools.items())}
    picked = []
    while len(picked) < n_problems and any(pools.values()):
        for key in sorted(pools):
            if pools[key] and len(picked) < n_problems:
                picked.append(int(pools[key].pop(0)))
    out = []
    for pid in sorted(picked):
        ok = sorted((t for t in by_problem[pid] if eligible[t["trace_id"]]),
                    key=lambda t: t["trace_id"])
        for label in (True, False):
            pool = [t for t in ok if t["is_correct"] == label]
            out.append(pool[rng.integers(len(pool))])
    strata = Counter((by_problem[p][0]["level"], pass_bin(
        sum(t["is_correct"] for t in by_problem[p]) / len(by_problem[p]))) for p in picked)
    return out, strata


def checkpoints(text, answer, offsets, match):
    # requested token index per checkpoint; None when it cannot be located
    n = len(offsets)
    want = {name: int(q * n) for name, q in FRACTIONS.items()}
    fm = first_mention(text, answer, match)
    want["first_mention"] = None if fm is None else token_at(offsets, fm)
    pf = pre_final(text)
    want["pre_final"] = None if pf is None else token_at(offsets, pf)
    return want


def record(prefix_id, trace, problem, ckpt, cut, requested, p_ids, gen_ids, pass_k, pass_n):
    return {
        "prefix_id": prefix_id,
        "source": "truncation",
        "problem_id": problem["problem_id"],
        "level": problem["level"],
        "trace_id": trace and trace["trace_id"],
        "trace_label": trace and ("correct" if trace["is_correct"] else "wrong"),
        "trace_final_answer": trace and trace["answer"],
        "trace_gen_len": trace and trace["_len"],
        "gold_answer": problem["gold"],
        "problem_pass_k": pass_k,
        "problem_pass_n": pass_n,
        "checkpoint": ckpt,
        "cut_token_idx": cut,
        "requested_token_idx": requested,
        "prompt_text": problem["_prompt"],
        "prompt_token_ids": p_ids,
        "prefix_gen_token_ids": gen_ids,
        "max_tokens": MAX_NEW - cut,
        "edit_meta": None,
    }


def ceiling_population(traces, lens, min_len):
    # probe_cmp's fixed population: untruncated traces whose retokenized length reaches
    # min_len, then problems with a correct and a wrong trace among them. every trace kept.
    ok = [t for t in traces if not t["trunc"] and t["n_gen"] < MAX_NEW and lens[t["trace_id"]] >= min_len]
    by_problem = defaultdict(list)
    for t in ok:
        by_problem[t["problem_id"]].append(t)
    return {pid: sorted(ts, key=lambda t: t["trace_id"]) for pid, ts in sorted(by_problem.items())
            if len({t["is_correct"] for t in ts}) == 2}


def est_cont_tokens(all_traces, cut, cap):
    # expected continuation length from cut: the problem's own tag-B traces (truncated ones
    # count at the cap) that ran past cut. only a planning number, never used as a cap.
    left = [min(t["n_gen"], MAX_NEW) - cut for t in all_traces if t["n_gen"] > cut]
    return int(min(np.mean(left), cap)) if left else cap


def tier_order(units, target, seed):
    # units: {problem_id: (pass rate, est tokens)}. problems drawn round-robin across pass
    # bins in a seeded random order; tier 1 is the shortest prefix of that order reaching
    # target tokens, tier 2 the rest. returns [(problem_id, tier)] in run order.
    rng = np.random.default_rng(seed)
    pools = defaultdict(list)
    for pid in sorted(units):
        pools[pass_bin(units[pid][0])].append(pid)
    pools = {k: [int(x) for x in rng.permutation(v)] for k, v in sorted(pools.items())}
    order = []
    while any(pools.values()):
        for key in sorted(pools):
            if pools[key]:
                order.append(pools[key].pop(0))
    out, total, tier = [], 0, 1
    for pid in order:
        if tier == 1 and total >= target:
            tier = 2
        out.append((pid, tier))
        total += units[pid][1]
    return out


def build_ceiling(args, tokenizer, traces):
    lens, enc = {}, {}
    for t in traces:
        if not t["trunc"] and t["n_gen"] < MAX_NEW:
            enc[t["trace_id"]] = encode_trace(tokenizer, t["text"])[0]
        lens[t["trace_id"]] = len(enc[t["trace_id"]]) if t["trace_id"] in enc else 0
    pop = ceiling_population(traces, lens, max(args.positions))
    n_traces = sum(map(len, pop.values()))
    drift = [t["trace_id"] for ts in pop.values() for t in ts if not canonical(t, enc[t["trace_id"]])]
    print(f"population: {n_traces} traces over {len(pop)} mixed problems "
          f"({sum(t['is_correct'] for ts in pop.values() for t in ts)} correct); "
          f"{len(drift)} fail the retokenization count check (kept, flagged canonical=false)")

    by_problem = defaultdict(list)
    for t in traces:
        by_problem[t["problem_id"]].append(t)
    per_problem, units = {}, {}
    for pid, ts in pop.items():
        problem = dict(ts[0])
        problem["_prompt"] = build_prompt(tokenizer, problem["problem"])
        p_ids = prompt_ids(tokenizer, problem["problem"])
        everyone = by_problem[pid]
        pass_k, pass_n = sum(t["is_correct"] for t in everyone), len(everyone)
        recs = [record(f"ceil/{pid}/t0", None, problem, "t0", 0, 0, p_ids, [], pass_k, pass_n)]
        for trace in ts:
            ids = enc[trace["trace_id"]]
            trace = dict(trace, _len=len(ids))
            for pos in args.positions:
                gen = ids[:pos]
                # a cut inside a multi-byte character decodes to U+FFFD at the end
                decoded = tokenizer.decode(gen).rstrip("\ufffd")
                if not trace["text"].startswith(decoded):
                    raise AssertionError(f"{trace['trace_id']} p{pos}: decode(ids[:{pos}]) is not "
                                         f"a prefix of the stored text: ...{decoded[-60:]!r}")
                recs.append(record(f"ceil/{trace['trace_id']}/p{pos}", trace, problem, f"p{pos}",
                                   pos, pos, p_ids, gen, pass_k, pass_n))
        for r in recs:
            r["max_tokens"] = args.max_continuation
            r["canonical"] = r["trace_id"] not in drift
            r["est_cont_tokens"] = est_cont_tokens(everyone, r["cut_token_idx"], args.max_continuation)
        per_problem[pid] = recs
        units[pid] = (pass_k / pass_n, args.n * sum(r["est_cont_tokens"] for r in recs))

    records = []
    for rank, (pid, tier) in enumerate(tier_order(units, args.tier1_tokens, args.seed)):
        for r in per_problem[pid]:
            records.append(dict(r, tier=tier, order=len(records), problem_rank=rank))
    return records


def ceiling_report(records, n):
    print(f"\n{len(records)} prefixes, est. tokens at n={n} (from tag-B lengths of the same problem):")
    for key in ("checkpoint", "tier"):
        c, tok = Counter(), Counter()
        for r in records:
            c[r[key]] += 1
            tok[r[key]] += n * r["est_cont_tokens"]
        for k in sorted(c, key=str):
            print(f"  {key} {k}: {c[k]} prefixes, {tok[k] / 1e6:.1f}M tokens")
    for tier in (1, 2):
        pids = {r["problem_id"] for r in records if r["tier"] == tier}
        print(f"  tier {tier}: {len(pids)} problems, pass bins "
              f"{dict(Counter(pass_bin(next(r['problem_pass_k'] / r['problem_pass_n'] for r in records if r['problem_id'] == p)) for p in pids))}")
    print(f"  total {n * sum(r['est_cont_tokens'] for r in records) / 1e6:.1f}M tokens")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--design", choices=["checkpoints", "ceiling"], default="checkpoints")
    parser.add_argument("--positions", type=int, nargs="+", default=list(CEILING_POSITIONS),
                        help="ceiling: exact token cuts; the population must reach the largest")
    parser.add_argument("--max-continuation", type=int, default=MAX_NEW, help="ceiling: max_tokens")
    parser.add_argument("--n", type=int, default=8, help="ceiling: samples per prefix, for estimates")
    parser.add_argument("--tier1-tokens", type=float, default=12.5e6, help="ceiling: tier 1 size")
    parser.add_argument("--data-dir", default=None, help="where traces_<tag>.jsonl lives")
    parser.add_argument("--tag", default="B")
    parser.add_argument("--n-problems", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=f"{OUT_DIR}/prefixes.jsonl")
    parser.add_argument("--force", action="store_true", help="overwrite an existing --out")
    args = parser.parse_args()
    if os.path.exists(args.out) and not args.force:
        raise SystemExit(f"{args.out} exists; rollouts are keyed to it. --force to overwrite")

    tokenizer = load_tokenizer()
    path = out_path(args.tag)
    if args.data_dir:
        path = os.path.join(args.data_dir, os.path.basename(path))
    traces = [json.loads(line) for line in open(path)]
    if args.design == "ceiling":
        records = build_ceiling(args, tokenizer, traces)
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        ceiling_report(records, args.n)
        print(f"-> {args.out}")
        return
    enc, eligible, why = {}, {}, Counter()
    for t in traces:
        if t["trunc"]:
            why["truncated"] += 1
        elif t["answer"] is None:
            why["no final answer"] += 1
        else:
            ids, offsets = encode_trace(tokenizer, t["text"])
            if canonical(t, ids):
                enc[t["trace_id"]] = (ids, offsets)
            else:
                why["retokenization drift"] += 1
        eligible[t["trace_id"]] = t["trace_id"] in enc
    print(f"{sum(eligible.values())}/{len(traces)} traces eligible; excluded {dict(why)}")

    chosen, strata = select(traces, eligible, args.n_problems, args.seed)
    print(f"{len(chosen) // 2} problems by (level, pass bin): {dict(sorted(strata.items()))}")

    by_problem = defaultdict(list)
    for t in traces:
        by_problem[t["problem_id"]].append(t)
    records, dropped, shared = [], Counter(), 0
    for i in range(0, len(chosen), 2):
        problem = dict(chosen[i])
        problem["_prompt"] = build_prompt(tokenizer, problem["problem"])
        p_ids = prompt_ids(tokenizer, problem["problem"])
        ts = by_problem[problem["problem_id"]]
        pass_k, pass_n = sum(t["is_correct"] for t in ts), len(ts)
        records.append(record(f"trunc/{problem['problem_id']}/t0", None, problem, "t0", 0, 0,
                              p_ids, [], pass_k, pass_n))
        for trace in chosen[i : i + 2]:
            ids, offsets = enc[trace["trace_id"]]
            trace = dict(trace, _len=len(ids))
            points = cut_points(trace["text"], offsets)
            seen = set()
            for ckpt, req in checkpoints(trace["text"], trace["answer"], offsets, math_match).items():
                if req is None:
                    dropped[f"{ckpt}: not found"] += 1
                    print(f"  {trace['trace_id']} {ckpt}: not found, skipped")
                    continue
                cut = snap(points, req)
                if cut == 0:
                    dropped[f"{ckpt}: snapped to 0"] += 1
                    print(f"  {trace['trace_id']} {ckpt}: snapped to 0 (req {req}), skipped")
                    continue
                gen = ids[:cut]
                decoded = tokenizer.decode(gen)
                if not trace["text"].startswith(decoded):
                    raise AssertionError(
                        f"{trace['trace_id']} {ckpt}: decode(ids[:{cut}]) is not a prefix of the "
                        f"stored text: ...{decoded[-60:]!r}")
                shared += cut in seen
                seen.add(cut)
                records.append(record(f"trunc/{trace['trace_id']}/{ckpt}", trace, problem, ckpt,
                                      cut, req, p_ids, gen, pass_k, pass_n))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"\n{len(records)} prefixes -> {args.out}")
    print(f"per checkpoint: {dict(Counter(r['checkpoint'] for r in records))}")
    if dropped:
        print(f"dropped: {dict(dropped)}")
    if shared:
        print(f"{shared} checkpoints share a cut with another on the same trace (kept)")


if __name__ == "__main__":
    main()
