# MLE of the sampling temperature behind traces_B, per generation block. the notebook shows the
# pilot resuming with "8/80 problems left" under --temp 0.7, so lines 0-431 (72 problems) come from
# an earlier invocation whose flags are not on record (generate.py's default --temp is 0.6),
# lines 432-479 from that visible --temp 0.7 resume, and 480+ from run_bg gen_B ... --temp 0.7.
# teacher-forced under vLLM's sampler (logits / T, then top-p 0.95), check_temp.py's scoring.
# traces are spread evenly over each block's lines, so a sub-block at another T cannot hide.
#   python temp_blocks.py --traces data/traces_B.jsonl [--n 20] [--out temp_blocks.json]
import argparse, json, os, sys
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--traces", required=True)
ap.add_argument("--repo", default=".")
ap.add_argument("--n", type=int, default=20, help="traces per block")
ap.add_argument("--max-tokens-per-trace", type=int, default=4096)
ap.add_argument("--out", default=None)
a = ap.parse_args()
sys.path.insert(0, os.path.join(a.repo, "src", "recovery"))
from common import MAX_NEW, MODEL, TOP_P, canonical, encode_trace, load_tokenizer, prompt_ids  # noqa: E402
from check_temp import trace_curves  # noqa: E402
from transformers import AutoModelForCausalLM  # noqa: E402

temps = np.round(np.arange(0.30, 1.2 + 1e-9, 0.01), 2)  # the grid estimate_temp.py used
i6, i7 = int(np.abs(temps - 0.6).argmin()), int(np.abs(temps - 0.7).argmin())
traces = [json.loads(line) for line in open(a.traces)]
assert len(traces) == 2202, len(traces)
blocks = {"lines 0-431": range(0, 432), "lines 432-479": range(432, 480), "lines 480+": range(480, len(traces))}
for name, rng_ in blocks.items():
    pids = [traces[i]["problem_id"] for i in rng_]
    print(f"{name}: {len(rng_)} traces, {len(set(pids))} problems, problem_id {pids[0]}..{pids[-1]}", flush=True)
# a block boundary must not split a problem's 6 samples
for b in (432, 480):
    assert traces[b - 1]["problem_id"] != traces[b]["problem_id"], f"line {b} splits a problem"

tok = load_tokenizer()
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).to("cuda").eval()
rng = np.random.default_rng(0)
print(f"\ngrid {temps[0]}..{temps[-1]}, top_p {TOP_P}, <= {a.max_tokens_per_trace} tokens per trace", flush=True)
results = {}
for name, lines in blocks.items():
    pool = [i for i in lines if not traces[i]["trunc"] and traces[i]["n_gen"] < MAX_NEW]
    starts = np.unique(np.linspace(0, len(pool), min(a.n, len(pool)), endpoint=False).astype(int))
    taken, rows, nuc, plain = set(), [], [], []
    for s in starts:
        for j in range(s, len(pool)):  # first canonical trace at or after the slot
            i = pool[j]
            if i in taken:
                continue
            t = traces[i]
            ids, _ = encode_trace(tok, t["text"])
            if canonical(t, ids):
                break
        else:
            continue
        taken.add(i)
        ids = ids[: a.max_tokens_per_trace]
        x, y, _ = trace_curves(model, "cuda", prompt_ids(tok, t["problem"]), ids, temps, TOP_P, 256)
        nuc.append(x), plain.append(y)
        rows.append({"line": i, "trace_id": t["trace_id"], "tokens": len(ids), "t_hat": float(temps[x.argmax()]),
                     "dlogL_07_minus_06": float(x[i7] - x[i6])})
        print(f"  {name} {len(rows):>2} line {i:>4} {t['trace_id']:>7} {len(ids):>5} tok | T_hat {temps[x.argmax()]:.2f} "
              f"| logL(0.7)-logL(0.6) {x[i7] - x[i6]:+8.1f}", flush=True)
    nuc, plain = np.array(nuc), np.array(plain)
    boots = [temps[nuc[rng.integers(0, len(nuc), len(nuc))].sum(0).argmax()] for _ in range(2000)]
    d = nuc[:, i7] - nuc[:, i6]
    results[name] = {"traces": len(rows), "tokens": int(sum(r["tokens"] for r in rows)),
                     "t_hat": float(temps[nuc.sum(0).argmax()]), "ci": [float(v) for v in np.percentile(boots, [2.5, 97.5])],
                     "t_hat_no_top_p": float(temps[plain.sum(0).argmax()]), "dlogL_07_minus_06": float(d.sum()),
                     "traces_favouring_07": int((d > 0).sum()), "per_trace": rows}

print("\nblock          traces  tokens  T_hat [95% CI]       no-top-p T_hat  logL(0.7)-logL(0.6)  traces favouring 0.7")
for name, r in results.items():
    print(f"{name:<14} {r['traces']:>6} {r['tokens']:>7}  {r['t_hat']:.2f} [{r['ci'][0]:.2f}, {r['ci'][1]:.2f}]"
          f"   {r['t_hat_no_top_p']:>13.2f}  {r['dlogL_07_minus_06']:>+19.1f}  {r['traces_favouring_07']:>6}/{r['traces']}")
if a.out:
    json.dump(results, open(a.out, "w"), indent=1)
    print("wrote", a.out)
