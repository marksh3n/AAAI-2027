# per-token confidence of the source prefixes the rollout banks continued, for DeepConf-style readouts (Fu et al.
# 2025, arXiv 2508.15260). one teacher-forced pass of the model over prompt + the first MAX_CUT generated tokens;
# per generated token i (the distribution that produced it, at prompt_len + i - 1):
#   entropy   -sum_j P(j) log P(j)                      full vocabulary
#   conf      -(1/K) sum_{j in top K} log P(j), K = 20  DeepConf token confidence (their vLLM call returns top 20)
#   logp      log P(sampled token)
#   mass_above  probability mass, at the sampler's T 0.7, of tokens strictly more likely than the sampled one: the
#             token is inside the top-p 0.95 nucleus iff mass_above < 0.95. a check that the ids are the sampled ones
# P is the raw model distribution (temperature 1, before the sampler's T 0.7 and top-p), the same for every model:
# what vLLM V1 returns as logprobs and what DeepConf scores. windows and summaries are computed later on CPU.
#   cuts  Qwen3 / Ministral: a cuts.jsonl (gpu_prep cuts), whose ids are the sampled ids (saved by run_rollouts)
#   ds    DeepSeek: sampled ids were never stored. the generated text is re-tokenized as build_prefixes did, and each
#         cut is checked three ways: n_gen == len(ids) + 1 (count), sha of (2-BOS prompt, ids[:cut]) == the
#         rollout row's prefix_sha (the bank continued exactly these ids), and the nucleus test above. the prompt
#         is scored with the BOS count the source was sampled with (tag-B lines < 480: one, else two)
#   python src/replication/confidence.py cuts --cuts cuts.jsonl --out <dir>
#   python src/replication/confidence.py ds --traces traces_B.jsonl --cohort ds_cohort.jsonl --out <dir>
import argparse, json, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path[:0] = [HERE, SRC, os.path.join(SRC, "recovery")]
import spec  # noqa: E402

MAX_CUT = 4096
K = 20
SAMPLER_T, SAMPLER_TOP_P = 0.7, 0.95
CHUNK = 1024
FIRST_BATCH_LINES = 480  # tag-B lines 0-479: the pilot, sampled with one BOS


def load_model():
    import torch
    import transformers as T
    cls = T.AutoModelForImageTextToText if spec.MULTIMODAL else T.AutoModelForCausalLM
    return cls.from_pretrained(spec.MODEL, revision=spec.MODEL_REVISION,
                               dtype=getattr(torch, spec.DTYPE)).to("cuda").eval()


def token_stats(model, prompt, gen, device="cuda", chunk=CHUNK):
    # per generated token: entropy, conf, logp (T 1) and mass_above (T 0.7); chunked forward with a KV cache
    import torch
    ids = list(prompt) + list(gen)
    P = len(prompt)
    out = {k: np.zeros(len(gen), np.float32) for k in ("entropy", "conf", "logp", "mass_above")}
    cache = None
    with torch.no_grad():
        for start in range(0, len(ids), chunk):
            piece = torch.tensor([ids[start:start + chunk]], device=device)
            o = model(input_ids=piece, past_key_values=cache, use_cache=True)
            cache = o.past_key_values
            # logits at absolute position q predict token q + 1; generated token i sits at P + i
            lo, hi = max(start, P - 1), min(start + piece.shape[1], P + len(gen) - 1)
            if lo >= hi:
                continue
            lg = o.logits[0, lo - start:hi - start].float()
            tgt = torch.tensor(ids[lo + 1:hi + 1], device=device)
            lp = torch.log_softmax(lg, -1)
            p07 = torch.softmax(lg / SAMPLER_T, -1)
            pt = p07.gather(1, tgt[:, None])
            sl = slice(lo + 1 - P, hi + 1 - P)
            out["entropy"][sl] = (-(lp.exp() * lp).sum(-1)).cpu().numpy()
            out["conf"][sl] = (-lp.topk(K, -1).values.mean(-1)).cpu().numpy()
            out["logp"][sl] = lp.gather(1, tgt[:, None])[:, 0].cpu().numpy()
            out["mass_above"][sl] = (p07 * (p07 > pt)).sum(-1).cpu().numpy()
            del o, lg, lp, p07
    return out


def save(out_dir, rows, arrays, extra):
    os.makedirs(out_dir, exist_ok=True)
    idx = pd.DataFrame(rows)
    np.savez_compressed(os.path.join(out_dir, "tokens.npz"),
                        **{k: np.concatenate([a[k] for a in arrays]) for k in arrays[0]})
    idx.to_parquet(os.path.join(out_dir, "index.parquet"))
    import torch, transformers
    json.dump({"model": spec.MODEL, "revision": spec.MODEL_REVISION, "dtype": spec.DTYPE, "k": K,
               "distribution": "raw model distribution, temperature 1 (entropy, conf, logp)",
               "nucleus_check": f"mass_above at T {SAMPLER_T}; in nucleus iff < {SAMPLER_TOP_P}",
               "max_cut": MAX_CUT, "torch": torch.__version__, "transformers": transformers.__version__, **extra},
              open(os.path.join(out_dir, "meta.json"), "w"), indent=1)
    v = np.concatenate([a["mass_above"] for a in arrays]) >= SAMPLER_TOP_P
    print(f"wrote {len(idx)} traces, {sum(len(a['conf']) for a in arrays)} tokens -> {out_dir}; "
          f"{v.sum()} tokens outside the T {SAMPLER_T} / top-p {SAMPLER_TOP_P} nucleus ({v.mean():.2e})", flush=True)


def cmd_cuts(a):
    # one record per trace: the longest cut's ids (the bank cuts here are all at 4096)
    recs = {}
    for r in map(json.loads, open(a.cuts)):
        if r["trace_id"] not in recs or r["cut_token_idx"] > recs[r["trace_id"]]["cut_token_idx"]:
            recs[r["trace_id"]] = r
    model = load_model()
    rows, arrays = [], []
    for k, r in enumerate(recs.values()):
        gen = r["prefix_gen_token_ids"][:MAX_CUT]
        s = token_stats(model, r["prompt_token_ids"], gen)
        rows.append({"trace_id": r["trace_id"], "problem_id": str(r["problem_id"]), "n_tokens": len(gen),
                     "cuts": [r["cut_token_idx"]], "ids_source": "sampled (stored)", "bos": r.get("bos")})
        arrays.append(s)
        if k % 50 == 0:
            print(f"{k + 1}/{len(recs)} traces", flush=True)
    save(a.out, rows, arrays, {"source": "cuts", "cuts_file": a.cuts})


def cmd_ds(a):
    from transformers import AutoTokenizer
    from cache import sha_obj
    from common import MAX_NEW, canonical, encode_trace
    from generate import build_prompt
    tok = AutoTokenizer.from_pretrained(spec.MODEL, revision=spec.TOKENIZER_REVISION)
    cohort = [json.loads(x) for x in open(a.cohort)]
    by_trace = {}
    for c in cohort:
        by_trace.setdefault(c["trace_id"], []).append(c)
    traces = {t["trace_id"]: t for t in map(json.loads, open(a.traces)) if t["trace_id"] in by_trace}
    model = load_model()
    rows, arrays = [], []
    for k, (tid, cs) in enumerate(by_trace.items()):
        t = traces[tid]
        ids = encode_trace(tok, t["text"])[0]
        p2 = tok(build_prompt(tok, t["problem"], t.get("structured", False)))["input_ids"]
        if p2[:2] != [tok.bos_token_id] * 2 or p2[2] == tok.bos_token_id:
            raise SystemExit(f"{tid}: prompt opens {p2[:3]}, not two BOS as build_prefixes asserted")
        pilot = cs[0]["line"] < FIRST_BATCH_LINES
        prompt = p2[1:] if pilot else p2
        sha_ok = {int(c["cut"]): sha_obj([p2, ids[:c["cut"]]]) == c["prefix_sha"] for c in cs}
        gen = ids[:MAX_CUT]
        s = token_stats(model, prompt, gen)
        rows.append({"trace_id": tid, "problem_id": str(t["problem_id"]), "n_tokens": len(gen),
                     "cuts": sorted(sha_ok), "ids_source": "re-tokenized text", "bos": "one" if pilot else "two",
                     "canonical": bool(canonical(t, ids)), "n_gen": t["n_gen"], "retok_len": len(ids),
                     "prefix_sha_ok": all(sha_ok.values()), "prefix_sha_bad_cuts": [c for c, ok in sha_ok.items() if not ok]})
        arrays.append(s)
        if k % 50 == 0:
            print(f"{k + 1}/{len(by_trace)} traces", flush=True)
    idx = pd.DataFrame(rows)
    print(f"count check (n_gen == retokenized + 1): {idx.canonical.sum()}/{len(idx)} pass; "
          f"prefix_sha == rollout rows: {idx.prefix_sha_ok.sum()}/{len(idx)} traces; pilot (one BOS): "
          f"{(idx.bos == 'one').sum()}", flush=True)
    save(a.out, rows, arrays, {"source": "ds", "traces_file": a.traces, "cohort_file": a.cohort, "max_new": MAX_NEW})


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("cuts")
    p.add_argument("--cuts", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("ds")
    p.add_argument("--traces", required=True)
    p.add_argument("--cohort", required=True)
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    {"cuts": cmd_cuts, "ds": cmd_ds}[a.cmd](a)


if __name__ == "__main__":
    main()
