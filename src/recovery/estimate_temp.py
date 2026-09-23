# MLE of the sampling temperature behind tag-B, per generation batch.
# teacher-forces traces and scores each sampled token under vLLM's sampler at T:
# logits / T, then top-p (keep a token while the mass ranked above it is < p), renormalise.
import argparse, json, os
import numpy as np
import torch

from common import (FIRST_BATCH_LINES, MAX_NEW, MODEL, TOP_P, canonical, encode_trace,
                    load_tokenizer, out_path, prompt_ids)

# contamination floor: a token fp16 noise puts just outside the nucleus costs log(eps)
# under that T instead of -inf, so one edge token cannot veto a whole region of T
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
        # position i predicts token i + 1
        start = len(p_ids) - 1
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


def summarise(name, temps, nuc, plain, outside, n_tokens, seed):
    # nuc/plain [traces, T]; CI from a bootstrap over traces (tokens within one are dependent)
    rng = np.random.default_rng(seed)
    t_hat = temps[nuc.sum(0).argmax()]
    boots = [temps[nuc[rng.integers(0, len(nuc), len(nuc))].sum(0).argmax()] for _ in range(N_BOOT)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    i6, i7 = np.abs(temps - 0.6).argmin(), np.abs(temps - 0.7).argmin()
    d = nuc.sum(0)[i7] - nuc.sum(0)[i6]
    i_hat = nuc.sum(0).argmax()
    print(f"{name:>10} | {len(nuc):>3} traces {n_tokens:>7} tok | T_hat {t_hat:.2f} "
          f"[{lo:.2f}, {hi:.2f}] | no-top-p T_hat {temps[plain.sum(0).argmax()]:.2f} | "
          f"logL(0.7)-logL(0.6) {d:+.1f} | out-of-nucleus at T_hat {int(outside[:, i_hat].sum())}")
    return {"batch": name, "traces": len(nuc), "tokens": n_tokens, "t_hat": float(t_hat),
            "ci": [float(lo), float(hi)], "t_hat_no_top_p": float(temps[plain.sum(0).argmax()]),
            "dlogL_07_minus_06": float(d)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="B")
    parser.add_argument("--n-traces", type=int, default=20, help="per batch")
    parser.add_argument("--max-tokens-per-trace", type=int, default=4096)
    parser.add_argument("--grid", type=float, nargs=3, default=[0.3, 1.2, 0.01],
                        metavar=("LO", "HI", "STEP"))
    parser.add_argument("--top-p", type=float, default=TOP_P)
    parser.add_argument("--chunk", type=int, default=256, help="positions per logits slice")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None, help="optional json of the per-batch results")
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM
    tokenizer = load_tokenizer()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float16 if device == "cuda" else torch.float32).to(device).eval()
    lo, hi, step = args.grid
    temps = np.round(np.arange(lo, hi + step / 2, step), 4)

    traces = [json.loads(line) for line in open(out_path(args.tag))]
    batches = {"first80": traces[:FIRST_BATCH_LINES], "rest": traces[FIRST_BATCH_LINES:]}
    rng = np.random.default_rng(args.seed)
    results = []
    print(f"grid {temps[0]}..{temps[-1]} ({len(temps)} points), top_p {args.top_p}, "
          f"<= {args.max_tokens_per_trace} tokens per trace, device {device}", flush=True)
    for name, pool in batches.items():
        pool = [t for t in pool if not t["trunc"] and t["n_gen"] < MAX_NEW]
        nuc, plain, outside, n_tok = [], [], [], 0
        for t in rng.permutation(len(pool)):
            if len(nuc) == args.n_traces:
                break
            trace = pool[t]
            ids, _ = encode_trace(tokenizer, trace["text"])
            if not canonical(trace, ids):
                continue  # retokenized ids would not be the sampled ones
            ids = ids[: args.max_tokens_per_trace]
            a, b, c = trace_curves(model, device, prompt_ids(tokenizer, trace["problem"]), ids,
                                   temps, args.top_p, args.chunk)
            nuc.append(a), plain.append(b), outside.append(c)
            n_tok += len(ids)
            print(f"  {name} {len(nuc)}/{args.n_traces} {trace['trace_id']} "
                  f"({len(ids)} tok) T_hat so far {temps[np.sum(nuc, 0).argmax()]:.2f}", flush=True)
        results.append(summarise(name, temps, np.array(nuc), np.array(plain), np.array(outside),
                                 n_tok, args.seed))
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(results, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
