# full-depth re-extraction for the layer sweep: every hidden-state entry (embedding + 28 decoder blocks,
# layer l = hidden_states[l + 1], l = -1..27) at every fixed position 128..32768, plus each finished
# trace's last token ("end"). the leading BOS matches how each trace was generated (bos_check.py):
# one for the first run (traces_B lines 0-479), two for the main run (480+). extract.py's own
# encode / fixed_positions / activations are reused, so token indexing is unchanged (state at pos-1).
# one part file per trace, so a crash costs one trace; parts are reused on a rerun.
#   python extract_full.py --traces data/traces_B.jsonl --out /content/e0_full [--limit 3]
import argparse, json, os, shutil, sys, time
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--traces", required=True)
ap.add_argument("--out", required=True, help="local dir for parts and the assembled arrays")
ap.add_argument("--copy-to", default=None, help="also copy the assembled arrays here (e.g. Drive)")
ap.add_argument("--repo", default=".")
ap.add_argument("--pilot-lines", type=int, default=480, help="traces_B lines generated with one BOS")
ap.add_argument("--limit", type=int, default=0, help="first N traces only (smoke test)")
a = ap.parse_args()
sys.path.insert(0, os.path.join(a.repo, "src"))
import extract as E  # noqa: E402
from taxonomy import RESTART  # noqa: E402

E.LAYERS = list(range(-1, 28))  # activations() reads this module global
tok, model, device = E.load_model()
assert model.config.num_hidden_layers == 28, model.config.num_hidden_layers
assert device == "cuda", "needs the GPU runtime"
bos = tok.bos_token_id
traces = [json.loads(line) for line in open(a.traces)]
assert len(traces) == 2202, len(traces)
if a.limit:
    traces = traces[: a.limit]
parts = os.path.join(a.out, "parts")
os.makedirs(parts, exist_ok=True)


def plan(i, t):
    prompt_ids, ids, _ = E.encode(tok, t)
    assert prompt_ids[0] == bos and prompt_ids[1] != bos, prompt_ids[:3]  # template text carries one BOS
    n_bos = 1 if i < a.pilot_lines else 2
    prompt_ids = [bos] * (n_bos - 1) + prompt_ids
    positions, token_idx, feed = E.fixed_positions(ids)
    has_end = not t["trunc"] and "</think>" in t["text"]
    if has_end:
        positions, token_idx, feed = positions + [-1], token_idx + [len(ids) - 1], ids
    return prompt_ids, ids, positions, token_idx, feed, n_bos


t0, done_tok = time.time(), 0
total_tok = sum(min(t["n_gen"], max(E.POSITIONS) + 1) for t in traces)  # planning only
print(f"{len(traces)} traces, ~{total_tok / 1e6:.1f}M tokens to feed; layers {E.LAYERS[0]}..{E.LAYERS[-1]} "
      f"({len(E.LAYERS)}), positions {E.POSITIONS} + end", flush=True)
for i, t in enumerate(traces):
    path = os.path.join(parts, f"{i}.npz")
    if os.path.exists(path):
        with np.load(path) as z:
            if str(z["trace_id"]) == t["trace_id"] and z["acts"].shape[1] == len(E.LAYERS):
                continue
    prompt_ids, ids, positions, token_idx, feed, n_bos = plan(i, t)
    h = E.activations(model, device, prompt_ids, feed, token_idx)
    assert h.shape == (len(token_idx), len(E.LAYERS), 1536), h.shape
    tmp = path + ".tmp.npz"
    np.savez(tmp, acts=h, trace_id=t["trace_id"], positions=np.array(positions), n_bos=n_bos, n_ids=len(ids))
    os.replace(tmp, path)
    done_tok += len(feed)
    if i % 25 == 0 or i == len(traces) - 1:
        el = time.time() - t0
        rate = done_tok / el if el else 0
        print(f"{i + 1}/{len(traces)} | {rate:,.0f} tok/s | elapsed {el / 60:.0f} min | "
              f"~{(total_tok - done_tok) / max(rate, 1) / 60:.0f} min left (upper bound)", flush=True)

# assemble: one row per (trace, checkpoint); acts written straight to a memmap
rows, n_rows = [], 0
for i, t in enumerate(traces):
    with np.load(os.path.join(parts, f"{i}.npz")) as z:
        n_rows += len(z["positions"])
acts = np.lib.format.open_memmap(os.path.join(a.out, "acts_full.npy"), mode="w+", dtype=np.float16,
                                 shape=(n_rows, len(E.LAYERS), 1536))
r = 0
for i, t in enumerate(traces):
    with np.load(os.path.join(parts, f"{i}.npz")) as z:
        h, positions, n_bos = z["acts"], z["positions"].tolist(), int(z["n_bos"])
    _, ids, _ = E.encode(tok, t)
    for j, pos in enumerate(positions):
        prefix = t["text"] if pos == -1 else tok.decode(ids[:pos])
        n_tok = len(ids) if pos == -1 else pos
        rows.append({"trace_id": t["trace_id"], "problem_id": t["problem_id"], "line": i, "n_bos": n_bos,
                     "ckpt": "end" if pos == -1 else str(pos), "pos": n_tok, "is_correct": bool(t["is_correct"]) and not t["trunc"],
                     "trunc": bool(t["trunc"]), "n_gen": t["n_gen"], "prefix": prefix,
                     "restarts": len(RESTART.findall(prefix)) / max(n_tok, 1) * 1000, "row": r})
        acts[r] = h[j]
        r += 1
acts.flush()
meta = pd.DataFrame(rows)
meta.to_parquet(os.path.join(a.out, "meta_full.parquet"))
json.dump({"layers": E.LAYERS, "layer_convention": "hidden_states[l+1]", "positions": E.POSITIONS,
           "token_index": "pos-1; end = last trace token", "bos": f"1 for lines < {a.pilot_lines}, else 2",
           "rows": n_rows, "traces": len(traces)}, open(os.path.join(a.out, "full_config.json"), "w"), indent=1)
print(f"assembled {n_rows} rows -> {a.out}/acts_full.npy {acts.shape}, meta_full.parquet", flush=True)
print(meta.groupby("ckpt", sort=False).agg(traces=("trace_id", "size")).T.to_string(), flush=True)
if a.copy_to:
    os.makedirs(a.copy_to, exist_ok=True)
    for f in ("acts_full.npy", "meta_full.parquet", "full_config.json"):
        shutil.copy(os.path.join(a.out, f), os.path.join(a.copy_to, f))
    print("copied to", a.copy_to, flush=True)
