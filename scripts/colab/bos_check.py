# which prompt generated the source traces: one BOS or two. the source prompt ids were never stored.
# the template text is fixed (the model's tokenizer_config last changed 2025-02-09), so only the
# leading BOS can differ: vLLM tokenizes a string prompt with special tokens on, and whether that
# adds a second BOS depends on the transformers version (5.x loads Qwen2Tokenizer and adds none).
# the rollouts were built with two (build_prefixes asserts it; run_rollouts checks vLLM agrees).
# teacher-forced: the variant that generated a trace has the higher expected log-likelihood.
#   python bos_check.py --prefixes <lenfilter_prefixes.csv> --traces data/traces_B.jsonl
import argparse, json, os, sys
import numpy as np
import pandas as pd
import torch
import transformers
from scipy.stats import binomtest

ap = argparse.ArgumentParser()
ap.add_argument("--prefixes", required=True, help="lenfilter.py's per-prefix csv: the 448 population traces")
ap.add_argument("--traces", required=True)
ap.add_argument("--repo", default=".")
ap.add_argument("--n", type=int, default=30, help="traces per generation block")
ap.add_argument("--max-tokens-per-trace", type=int, default=2048)
ap.add_argument("--out", default=None)
a = ap.parse_args()
sys.path.insert(0, os.path.join(a.repo, "src", "recovery"))
from common import MODEL, TOP_P, build_prompt, canonical, encode_trace, load_tokenizer  # noqa: E402
from check_temp import trace_curves  # noqa: E402
from transformers import AutoModelForCausalLM  # noqa: E402

T = 0.7
tok = load_tokenizer()
bos = tok.bos_token_id
traces = [json.loads(line) for line in open(a.traces)]
line_of = {t["trace_id"]: i for i, t in enumerate(traces)}
pop = sorted(set(pd.read_csv(a.prefixes).trace_id.dropna()), key=line_of.get)
probe = build_prompt(tok, traces[line_of[pop[0]]]["problem"])
head = tok(probe)["input_ids"][:3]
print(f"this env: transformers {transformers.__version__}, {type(tok).__name__}; default tokenization of a "
      f"prompt opens {head} -> {sum(1 for x in head[:2] if x == bos)} BOS (bos id {bos})", flush=True)


def variants(problem):
    one = tok(build_prompt(tok, problem), add_special_tokens=False)["input_ids"]
    assert one[0] == bos and one[1] != bos, one[:3]  # the template text itself carries one BOS
    return {"one BOS": one, "two BOS": [bos] + one}


blocks = {"lines 0-431": (0, 432), "lines 432-479": (432, 480), "lines 480+": (480, len(traces))}
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).to("cuda").eval()
rows = []
for name, (lo, hi) in blocks.items():
    cand = [t for t in pop if lo <= line_of[t] < hi]
    picked = 0
    for j in np.unique(np.linspace(0, len(cand), min(a.n * 2, len(cand)), endpoint=False).astype(int)):
        if picked == a.n:
            break
        t = traces[line_of[cand[j]]]
        ids, _ = encode_trace(tok, t["text"])
        if not canonical(t, ids):
            continue
        ids = ids[: a.max_tokens_per_trace]
        r = {"block": name, "line": line_of[t["trace_id"]], "trace_id": t["trace_id"], "tokens": len(ids)}
        for v, p_ids in variants(t["problem"]).items():
            nuc, plain, out = trace_curves(model, "cuda", p_ids, ids, np.array([T]), TOP_P, 256)
            r[f"nuc {v}"], r[f"plain {v}"], r[f"outside {v}"] = float(nuc[0]), float(plain[0]), int(out[0])
        r["d_nuc"] = r["nuc two BOS"] - r["nuc one BOS"]
        r["d_plain"] = r["plain two BOS"] - r["plain one BOS"]
        rows.append(r)
        picked += 1
        print(f"  {name} {picked:>2} line {r['line']:>4} {r['trace_id']:>7} {len(ids):>5} tok | "
              f"logL(two BOS) - logL(one BOS): top-p {r['d_nuc']:+8.2f}, no top-p {r['d_plain']:+8.2f}", flush=True)

df = pd.DataFrame(rows)
print("\nlogL(two BOS) - logL(one BOS) at T=0.7; positive favours two BOS (the rollouts' prompt)")
summ = []
for name, g in list(df.groupby("block", sort=False)) + [("all", df)]:
    k = int((g.d_nuc > 0).sum())
    summ.append({"block": name, "traces": len(g), "tokens": int(g.tokens.sum()), "sum d (top-p)": g.d_nuc.sum(),
                 "per token": g.d_nuc.sum() / g.tokens.sum(), "sum d (no top-p)": g.d_plain.sum(),
                 "traces favouring two BOS": f"{k}/{len(g)}",
                 "sign test p": binomtest(k, len(g)).pvalue if len(g) else np.nan})
print(pd.DataFrame(summ).round(4).to_string(index=False))
if a.out:
    df.to_csv(a.out, index=False)
    print("wrote", a.out)
