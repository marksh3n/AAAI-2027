# GPU only (Colab T4). re-extracts a few stored rows of every registered source with the model
# and names the (token offset, layer mapping) that reproduces them. an off-by-one shows up as
# "matches index pos" instead of "pos-1"; a block-index mix-up as "hidden_states[l]".
# it reuses extract.py's prompt and tokenization, so it catches cache, offset and row-order
# bugs, not a bug that extraction and this check would share.
#   python src/probe_cmp/reextract_check.py --data data --out <dir> [--n 3]
import argparse, json, os, sys
import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # src/, for extract.py
import extract  # noqa: E402
import data as D  # noqa: E402

MIN_COS = 0.99  # fp16 and chunking move the last bits; a wrong token or layer moves the direction
CHUNK = 1024


def all_states(model, device, prompt_ids, ids, picks):
    # every hidden_states entry at each trace index in picks, chunked with a kv cache
    seq = prompt_ids + ids
    want = {len(prompt_ids) + i: i for i in picks}
    got, cache = {}, None
    with torch.no_grad():
        for start in range(0, len(seq), CHUNK):
            piece = torch.tensor([seq[start: start + CHUNK]], device=device)
            out = model.model(piece, past_key_values=cache, use_cache=True, output_hidden_states=True)
            cache = out.past_key_values
            for pos, i in want.items():
                if start <= pos < start + piece.shape[1]:
                    got[i] = torch.stack([h[0, pos - start] for h in out.hidden_states]).float().cpu().numpy()
            del out
    return got


def expected_index(src, trace, row, ids, offsets):
    if src["token_index"] == "pos-1":
        return int(row[src["pos_col"]]) - 1
    if src["token_index"] == "answer_end":
        _, token_idx = extract.segment(trace["text"], offsets)
        return int(token_idx[-1])
    raise ValueError(src["token_index"])


def cos(a, b):
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="data/probe_cmp")
    ap.add_argument("--n", type=int, default=3, help="random rows per source")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if not torch.cuda.is_available():
        sys.exit("no GPU: run this on a Colab GPU runtime; row-level alignment stays unverified")
    reg = D.load_registry()
    traces = {t["trace_id"]: t for t in map(json.loads, open(os.path.join(a.data, reg["traces"])))}
    tok, model, device = extract.load_model()
    rows_out, ok_all = [], True
    for src in reg["sources"]:
        ap_, mp = os.path.join(a.data, src["acts"]), os.path.join(a.data, src["meta"])
        if not (os.path.exists(ap_) and os.path.exists(mp)):
            print(f"{src['name']}: not present, skipped")
            continue
        if src["layer_convention"] != "hidden_states[l+1]":
            sys.exit(f"{src['name']}: unknown layer convention {src['layer_convention']}")
        meta, acts = pd.read_parquet(mp), np.load(ap_, mmap_mode="r")
        rng = np.random.default_rng(a.seed)
        print(f"\n{src['name']} ({src['token_index']}):")
        print(f"{'row':>7} {'trace':>9}  " + "  ".join(f"{k:>18}" for k in ("token index", "hs[l+1] min cos", "hs[l] min cos")))
        for r in sorted(rng.choice(len(meta), min(a.n, len(meta)), replace=False)):
            row = meta.iloc[r]
            trace = traces[row.trace_id]
            prompt_ids, ids, offsets = extract.encode(tok, trace)
            e = expected_index(src, trace, row, ids, offsets)
            if src["token_index"] == "pos-1":
                names = ["pos-2", "pos-1", "pos", "pos+1"]  # pos-1 is the registered index
            else:
                names = ["end-1", "end", "end+1"]
            want = names[1]
            cands = {n: e + k - 1 for k, n in enumerate(names) if 0 <= e + k - 1 < len(ids)}
            if want not in cands:
                print(f"{r:>7} {row.trace_id:>9}  registered index {e} is outside the "
                      f"{len(ids)}-token retokenized trace   <-- MISMATCH")
                ok_all = False
                rows_out.append({"source": src["name"], "row": int(r), "trace_id": row.trace_id,
                                 "expected_index": e, "n_tokens": len(ids), "passed": False})
                continue
            states = all_states(model, device, prompt_ids, ids[: max(cands.values()) + 1], list(cands.values()))
            stored = np.asarray(acts[r], np.float32)
            grid, per_layer = {}, {}
            for k, i in cands.items():
                h = states[i]
                per_layer[k] = [cos(stored[j], h[l + 1]) for j, l in enumerate(src["layers"])]
                plus1 = min(per_layer[k])
                block = min(cos(stored[j], h[l]) for j, l in enumerate(src["layers"]) if l >= 0)
                grid[k] = (plus1, block)
                print(f"{r:>7} {row.trace_id:>9}  {k:>18}  {plus1:>18.4f}  {block:>18.4f}")
            match = [(k, m) for k, (p1, b) in grid.items() for m, v in (("hs[l+1]", p1), ("hs[l]", b)) if v >= MIN_COS]
            # identical neighbouring tokens (e.g. repeated newlines) can match twice; the registered
            # index still has to be among the matches
            passed = (want, "hs[l+1]") in match
            ok_all &= passed
            name = " and ".join(f"index {k} with {m}" for k, m in match) if match else "nothing"
            print(f"{'':>18}-> matches {name}; registered: index {want} with hs[l+1]"
                  f"{'' if passed else '   <-- MISMATCH'}")
            if not passed:
                low = [f"L{l}: {c:.3f}" for l, c in zip(src["layers"], per_layer[want]) if c < MIN_COS]
                print(f"{'':>18}   at the registered index, layers below {MIN_COS}: {', '.join(low)}")
            rows_out.append({"source": src["name"], "row": int(r), "trace_id": row.trace_id, "expected_index": e,
                             "grid": {k: {"hs[l+1]": v[0], "hs[l]": v[1]} for k, v in grid.items()},
                             "per_layer_cos_at_registered": dict(zip(map(str, src["layers"]), per_layer[want])),
                             "match": match, "passed": passed})
    summary = (f"{sum(r['passed'] for r in rows_out)}/{len(rows_out)} sampled rows reproduce at the registered "
               f"token index with hidden_states[l+1] (min per-layer cosine >= {MIN_COS})")
    print("\n" + summary + ("" if ok_all else "  -- NOT VERIFIED, do not trust results from these files"))
    os.makedirs(a.out, exist_ok=True)
    json.dump({"passed": bool(ok_all and rows_out), "summary": summary, "rows": rows_out},
              open(os.path.join(a.out, "reextract_check.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
