# gate for the ceiling run: which of a few temperatures best explains the sampled tokens of
# original traces from the ceiling population, under vLLM's sampler (logits / T, then top-p:
# keep a token while the mass ranked above it is < p, renormalise). teacher-forced with HF.
# also re-runs the rollout runner's prompt-token-id check (two BOS) through vLLM itself.
import argparse, gc, json, os
import numpy as np
import torch

from common import MODEL, TOP_P, canonical, encode_trace, load_tokenizer, read_jsonl

# contamination floor: a token fp16 noise puts just outside the nucleus costs log(eps)
# under that T instead of -inf, so one edge token cannot veto a temperature
EPS = 1e-3
N_BOOT = 2000


def nucleus_loglik(logits, targets, temps, top_p, eps=EPS):
    # logits [P, V], targets [P] -> ([len(temps), P] loglik, [len(temps), P] in-nucleus)
    logits = logits.float()
    z, order = logits.sort(dim=-1, descending=True)  # order is the same at every T
    rank = (order == targets[:, None]).float().argmax(dim=-1, keepdim=True)
    ll, inside = [], []
    for T in temps:
        logp = torch.log_softmax(z / T, dim=-1)
        p = logp.exp()
        keep = (p.cumsum(-1) - p) < top_p
        log_mass = torch.logsumexp(logp.masked_fill(~keep, float("-inf")), dim=-1)
        lp_t = logp.gather(1, rank).squeeze(1)
        in_t = keep.gather(1, rank).squeeze(1)
        nucleus = torch.where(in_t, (lp_t - log_mass).exp(), torch.zeros_like(lp_t))
        ll.append(torch.log((1 - eps) * nucleus + eps * lp_t.exp()))
        inside.append(in_t)
    return torch.stack(ll), torch.stack(inside)


def plain_loglik(logits, targets, temps):
    # no top-p: a cross-check that does not depend on nucleus edges
    logits = logits.float()
    return torch.stack([torch.log_softmax(logits / T, dim=-1).gather(1, targets[:, None]).squeeze(1)
                        for T in temps])


def trace_curves(model, device, p_ids, ids, temps, top_p, chunk):
    # summed loglik per T for one trace, plus out-of-nucleus counts per T
    seq = torch.tensor([p_ids + ids], device=device)
    with torch.inference_mode():
        h = model.model(seq).last_hidden_state[0]
        start = len(p_ids) - 1  # position i predicts token i + 1
        nuc = torch.zeros(len(temps), dtype=torch.float64)
        plain = torch.zeros(len(temps), dtype=torch.float64)
        outside = torch.zeros(len(temps), dtype=torch.long)
        for a in range(0, len(ids), chunk):
            b = min(a + chunk, len(ids))
            logits = model.lm_head(h[start + a : start + b])
            tgt = torch.tensor(ids[a:b], device=device)
            ll, inside = nucleus_loglik(logits, tgt, temps, top_p)
            nuc += ll.sum(1).double().cpu()
            plain += plain_loglik(logits, tgt, temps).sum(1).double().cpu()
            outside += (~inside).sum(1).cpu()
    return nuc.numpy(), plain.numpy(), outside.numpy()


def pick_traces(prefixes, traces, n, seed):
    # half correct, half wrong, from the traces the ceiling prefix file continues
    in_file = {r["trace_id"] for r in prefixes if r["trace_id"]}
    pool = [t for t in traces if t["trace_id"] in in_file]
    rng = np.random.default_rng(seed)
    out = []
    for label in (True, False):
        side = [t for t in pool if t["is_correct"] == label]
        out += [side[i] for i in rng.permutation(len(side))]
    return pool, out


def verdict(temps, nuc, target, seed):
    # best T by summed loglik, and how often a bootstrap over traces agrees
    rng = np.random.default_rng(seed)
    best = temps[nuc.sum(0).argmax()]
    wins = np.mean([temps[nuc[rng.integers(0, len(nuc), len(nuc))].sum(0).argmax()] == target
                    for _ in range(N_BOOT)])
    return best, float(wins)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", required=True, help="traces_B.jsonl")
    parser.add_argument("--prefix-file", required=True, help="the ceiling prefix file")
    parser.add_argument("--n-traces", type=int, default=20, help="half correct, half wrong")
    parser.add_argument("--max-tokens-per-trace", type=int, default=4096)
    parser.add_argument("--temps", type=float, nargs="+", default=[0.6, 0.7, 0.8, 1.0])
    parser.add_argument("--target", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=TOP_P)
    parser.add_argument("--chunk", type=int, default=256, help="positions per logits slice")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None, help="optional json of the result")
    parser.add_argument("--skip-vllm", action="store_true", help="skip the vLLM prompt check")
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM
    tokenizer = load_tokenizer()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float16 if device == "cuda" else torch.float32).to(device).eval()
    temps = np.array(args.temps)
    prefixes = read_jsonl(args.prefix_file)
    by_trace = {r["trace_id"]: r for r in prefixes if r["trace_id"]}
    pool, candidates = pick_traces(prefixes, read_jsonl(args.traces), args.n_traces, args.seed)
    print(f"{len(pool)} population traces; temps {list(temps)}, top_p {args.top_p}, "
          f"<= {args.max_tokens_per_trace} tokens per trace, device {device}", flush=True)

    rows, nuc, plain, outside, bos = [], [], [], [], []
    want = {True: args.n_traces // 2, False: args.n_traces - args.n_traces // 2}
    for trace in candidates:
        if want[trace["is_correct"]] == 0:
            continue
        ids, _ = encode_trace(tokenizer, trace["text"])
        if not canonical(trace, ids):
            continue  # retokenized ids would not be the sampled ones
        want[trace["is_correct"]] -= 1
        p_ids = by_trace[trace["trace_id"]]["prompt_token_ids"]
        bos.append(p_ids[:2] == [tokenizer.bos_token_id] * 2)
        ids = ids[: args.max_tokens_per_trace]
        a, b, c = trace_curves(model, device, p_ids, ids, temps, args.top_p, args.chunk)
        nuc.append(a), plain.append(b), outside.append(c)
        rows.append({"trace_id": trace["trace_id"], "correct": trace["is_correct"], "tokens": len(ids),
                     "loglik": dict(zip(map(str, temps), a.round(1).tolist()))})
        print(f"  {len(rows):>2} {trace['trace_id']:>7} {'correct' if trace['is_correct'] else 'wrong':>7} "
              f"{len(ids):>5} tok | best T {temps[a.argmax()]}", flush=True)
    nuc, plain, outside = np.array(nuc), np.array(plain), np.array(outside)
    n_tok = sum(r["tokens"] for r in rows)

    print(f"\n{len(rows)} traces ({sum(r['correct'] for r in rows)} correct), {n_tok} sampled tokens")
    print(f"{'T':>5} {'logL (top-p)':>14} {'per token':>10} {'vs target':>10} {'logL no top-p':>14} "
          f"{'outside nucleus':>16}")
    it = int(np.abs(temps - args.target).argmin())
    for i, T in enumerate(temps):
        print(f"{T:>5.2f} {nuc[:, i].sum():>14.1f} {nuc[:, i].sum() / n_tok:>10.4f} "
              f"{nuc[:, i].sum() - nuc[:, it].sum():>+10.1f} {plain[:, i].sum():>14.1f} "
              f"{int(outside[:, i].sum()):>8} ({outside[:, i].sum() / n_tok:.2%})")
    best, wins = verdict(temps, nuc, args.target, args.seed)
    best_plain = temps[plain.sum(0).argmax()]
    for label in (True, False):
        m = np.array([r["correct"] == label for r in rows])
        if m.any():
            print(f"  {'correct' if label else 'wrong'} traces alone: best T {temps[nuc[m].sum(0).argmax()]}")
    clear = best == args.target and wins >= 0.95
    print(f"\nbest T {best} (no top-p: {best_plain}); T={args.target} wins {wins:.1%} of "
          f"{N_BOOT} bootstrap resamples of traces")
    print(f"prompt ids open with two BOS on {sum(bos)}/{len(bos)} traces")
    print("VERDICT: " + (f"T={args.target} is clearly best" if clear else
                         f"T={args.target} is NOT clearly best: stop, the ceiling would be sampled "
                         "at the wrong temperature"), flush=True)

    prompt_ok = None
    if not args.skip_vllm:
        # the runner's own check, through vLLM's own tokenization of prompt_text
        del model
        gc.collect()
        torch.cuda.empty_cache()
        from vllm import LLM, SamplingParams
        import run_rollouts as rr
        recs = [by_trace[r["trace_id"]] for r in rows]
        llm = LLM(model=MODEL, dtype="float16", max_model_len=4096, gpu_memory_utilization=0.8,
                  enforce_eager=True)
        rr.check_prompts(llm, SamplingParams, recs)
        prompt_ok = True
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump({"temps": temps.tolist(), "best": float(best), "best_no_top_p": float(best_plain),
                   "target_bootstrap_wins": wins, "clear": bool(clear), "tokens": n_tok,
                   "outside_nucleus": outside.sum(0).tolist(), "two_bos": sum(bos),
                   "vllm_prompt_check": prompt_ok, "traces": rows}, open(args.out, "w"), indent=1)
    if not clear:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
