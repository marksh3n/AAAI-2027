# representation test, step 1 (GPU; a T4 will do). for each trace of the ceiling population, one
# forward pass over its first max(CUTS) generated tokens, keeping one layer's residual stream at each
# cut two ways: the last token (state at index pos-1, as extract.py) and the mean over the trailing
# WINDOW tokens (indices pos-WINDOW .. pos-1). the checkpoint-to-checkpoint change is built from these
# by reps.py, so all three representations come from one extraction under one prompt.
# the leading BOS matches how each trace was generated (bos_check.py, 2026-10-06): one for traces_B
# lines < 480 (the pilot), two for 480+. extract.py's encode gives the template's own single BOS.
# one part per trace, reused on a rerun when it holds the same trace, layers and cuts.
#   python src/probe_cmp/extract_reps.py --rollouts <recovery_ceiling> --traces data/traces_B.jsonl \
#       --out <dir> [--layers 20] [--check 3] [--limit N]
import argparse, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

CUTS = [128, 256, 512, 1024, 2048, 4096]  # probe_cmp's checkpoints up to the deepest rollout cut
WINDOW = 128
PILOT_LINES = 480  # traces_B lines generated with one BOS


def population(rollouts_path):
    # the traces the ceiling rolled out from: probe_cmp's fixed population over 128..4096 + end
    ids = set()
    for line in open(rollouts_path):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue  # a torn last line
        if r.get("trace_id"):
            ids.add(r["trace_id"])
    return ids


def window_bounds(prompt_len, cut, window):
    # absolute positions [lo, hi) of the trailing window ending at generated-token index cut-1
    return prompt_len + cut - window, prompt_len + cut


def reps(model, device, prompt_ids, trace_ids, layers, cuts, window, chunk):
    # chunked forward with a kv cache (causal, so identical to one pass); per layer and cut, the last
    # token's state and a running sum over the trailing window
    import torch
    ids = prompt_ids + trace_ids[: max(cuts)]
    P = len(prompt_ids)
    last = np.zeros((len(cuts), len(layers), model.config.hidden_size), np.float32)
    sums = np.zeros_like(last)
    cache = None
    with torch.no_grad():
        for start in range(0, len(ids), chunk):
            piece = torch.tensor([ids[start: start + chunk]], device=device)
            out = model.model(piece, past_key_values=cache, use_cache=True, output_hidden_states=True)
            cache = out.past_key_values
            end = start + piece.shape[1]
            for li, layer in enumerate(layers):
                h = out.hidden_states[layer + 1][0].float()
                for ci, cut in enumerate(cuts):
                    lo, hi = window_bounds(P, cut, window)
                    a, b = max(lo, start), min(hi, end)
                    if a < b:
                        sums[ci, li] += h[a - start: b - start].sum(0).cpu().numpy()
                    if start <= hi - 1 < end:
                        last[ci, li] = h[hi - 1 - start].cpu().numpy()
            del out
    del cache
    return last, sums / window


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", required=True, help="the ceiling rollouts: which traces to extract")
    ap.add_argument("--traces", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", type=int, nargs="+", default=[20])
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--check", type=int, default=3,
                    help="re-extract the last token of the first N traces with extract.activations and compare")
    ap.add_argument("--limit", type=int, default=0, help="first N population traces only (smoke test)")
    a = ap.parse_args()
    import extract as E  # needs torch and transformers: imported here so tests can load this module

    traces = [json.loads(line) for line in open(a.traces)]
    assert len(traces) == 2202, len(traces)
    want = population(a.rollouts)
    todo = [(i, t) for i, t in enumerate(traces) if t["trace_id"] in want]
    assert len(todo) == len(want), f"{len(want) - len(todo)} rollout traces are not in {a.traces}"
    if a.limit:
        todo = todo[: a.limit]
    tok, model, device = E.load_model()
    assert device == "cuda", "needs the GPU runtime"
    bos = tok.bos_token_id
    parts = os.path.join(a.out, "parts")
    os.makedirs(parts, exist_ok=True)
    print(f"{len(todo)} traces, layers {a.layers}, cuts {CUTS}, window {WINDOW}", flush=True)

    t0, fed = time.time(), 0
    for k, (i, t) in enumerate(todo):
        path = os.path.join(parts, f"{i}.npz")
        if os.path.exists(path):
            with np.load(path) as z:
                if (str(z["trace_id"]) == t["trace_id"] and z["layers"].tolist() == a.layers
                        and z["cuts"].tolist() == CUTS and int(z["window"]) == WINDOW):
                    continue
        prompt_ids, ids, _ = E.encode(tok, t)
        assert prompt_ids[0] == bos and prompt_ids[1] != bos, prompt_ids[:3]  # the template's one BOS
        assert len(ids) >= max(CUTS), f"{t['trace_id']}: {len(ids)} tokens < {max(CUTS)}"
        n_bos = 1 if i < PILOT_LINES else 2
        prompt_ids = [bos] * (n_bos - 1) + prompt_ids
        last, mean = reps(model, device, prompt_ids, ids, a.layers, CUTS, WINDOW, a.chunk)
        if k < a.check:
            # independent path: extract.activations at index pos-1 must give the same last-token state
            E.LAYERS = a.layers
            ref = E.activations(model, device, prompt_ids, ids[: max(CUTS)], [c - 1 for c in CUTS])
            cos = E.cosines(ref, last.astype(np.float16))
            assert cos.min() >= 0.99, f"{t['trace_id']}: last-token mismatch vs extract.activations, min cos {cos.min():.4f}"
            print(f"  check {t['trace_id']} ({n_bos} BOS): last token matches extract.activations, "
                  f"min cos {cos.min():.5f}", flush=True)
        tmp = path + ".tmp.npz"
        np.savez(tmp, last=last.astype(np.float16), mean=mean.astype(np.float16), trace_id=t["trace_id"],
                 line=i, n_bos=n_bos, layers=np.array(a.layers), cuts=np.array(CUTS), window=WINDOW)
        os.replace(tmp, path)
        fed += len(prompt_ids) + max(CUTS)
        if k % 25 == 0 or k == len(todo) - 1:
            el = time.time() - t0
            print(f"{k + 1}/{len(todo)} | {fed / max(el, 1e-9):,.0f} tok/s | {el / 60:.1f} min", flush=True)

    # assemble in population order
    ids_, last, mean, lines, nb = [], [], [], [], []
    for i, t in todo:
        with np.load(os.path.join(parts, f"{i}.npz")) as z:
            assert str(z["trace_id"]) == t["trace_id"]
            ids_.append(t["trace_id"]), last.append(z["last"]), mean.append(z["mean"])
            lines.append(i), nb.append(int(z["n_bos"]))
    dst = os.path.join(a.out, "reps.npz")
    np.savez(dst, trace_id=np.array(ids_), line=np.array(lines), n_bos=np.array(nb), last=np.stack(last),
             mean=np.stack(mean), layers=np.array(a.layers), cuts=np.array(CUTS), window=WINDOW)
    json.dump({"layers": a.layers, "layer_convention": "hidden_states[l+1]", "cuts": CUTS, "window": WINDOW,
               "last": "state at generated-token index pos-1", "mean": f"mean over indices pos-{WINDOW}..pos-1",
               "bos": f"1 for traces_B lines < {PILOT_LINES}, else 2", "traces": len(ids_)},
              open(os.path.join(a.out, "reps_config.json"), "w"), indent=1)
    print(f"wrote {dst}: last/mean {np.stack(last).shape} (trace, cut, layer, dim)", flush=True)


if __name__ == "__main__":
    main()
