# one-cell check of an acts/meta pair before it enters the analysis: shape, layers, checkpoint
# positions, every alignment assert, a layer-convention sanity check, and the fixed-population
# size for each candidate checkpoint set once the pair joins the registered ones.
#   python src/probe_cmp/validate.py                              # every registered pair
#   python src/probe_cmp/validate.py --source B_fixed
#   python src/probe_cmp/validate.py --acts acts_M.npy --meta meta_M.parquet --kind position \
#       --pos-col pos --layers -1 0 4 8 12 16 20 24 27 --prefix tokenize --name M_mid
import argparse, copy, json, os, sys
import numpy as np
import pandas as pd

import data as D

REQUIRED_LAYERS = [16, 20]


def adhoc_entry(a):
    e = {"name": a.name, "acts": a.acts, "meta": a.meta, "kind": a.kind, "layers": a.layers,
         "layer_convention": a.layer_convention, "prefix": a.prefix,
         "token_index": "pos-1" if a.kind == "position" else "answer_end"}
    if a.kind == "position":
        e["pos_col"] = a.pos_col
        if a.name_col:
            e["name_col"] = a.name_col
    return e


def manifest_layers(meta_path):
    # extract.py writes meta*.manifest.json next to the parquet; its config names the layers
    m = meta_path.replace(".parquet", ".manifest.json")
    if not os.path.exists(m):
        return None
    return json.load(open(m)).get("config", {}).get("layers")


def check_one(src, data_dir, traces, reg, ref_norms):
    a, m = os.path.join(data_dir, src["acts"]), os.path.join(data_dir, src["meta"])
    print(f"\n=== {src['name']}: {src['acts']} + {src['meta']}")
    if not (os.path.exists(a) and os.path.exists(m)):
        print("  files not present")
        return False
    meta, acts = pd.read_parquet(m), np.load(a, mmap_mode="r")
    print(f"  acts {acts.shape} {acts.dtype}; meta {meta.shape}, columns {list(meta.columns)}")
    print(f"  layers (registry) {src['layers']}, convention {src['layer_convention']}, "
          f"token index {src['token_index']}, prefix {src['prefix']}")
    ml = manifest_layers(m)
    if ml is not None:
        print(f"  layers (manifest) {ml}" + ("" if ml == src["layers"] else "   <-- DIFFERS from registry"))
    missing = [l for l in REQUIRED_LAYERS if l not in src["layers"]]
    if missing:
        print(f"  missing required layers {missing}")
    if src["kind"] == "position":
        vc = meta[src["pos_col"]].value_counts().sort_index()
        print("  checkpoints (tokens observed: rows) " + ", ".join(f"{int(k)}: {v}" for k, v in vc.items()))
        if src.get("name_col"):
            print(f"  labels from {src['name_col']}: {sorted(meta[src['name_col']].unique())}")
    else:
        print(f"  end checkpoint: {len(meta)} traces")
    results = D.check_pair(src, meta, acts, traces, reg["hidden_dim"])
    for name, ok, detail in results:
        print(f"  {'ok  ' if ok else 'FAIL'} {name}" + (f" ({detail})" if detail and not ok else ""))
    ok = all(r[1] for r in results) and not missing
    norms = D.layer_norms(acts) if acts.ndim == 3 else None
    if norms is not None and ref_norms is not None:
        ref_src, ref = ref_norms
        shared = [l for l in src["layers"] if l in ref_src["layers"]]
        print("  median |h| per layer vs " + ref_src["name"] + " (a layer-mapping shift shows as a ratio far from 1):")
        for l in shared:
            r = norms[src["layers"].index(l)] / ref[ref_src["layers"].index(l)]
            print(f"    L{l:>3}: {norms[src['layers'].index(l)]:8.1f} vs {ref[ref_src['layers'].index(l)]:8.1f}"
                  f"  ratio {r:5.2f}" + ("   <-- check convention" if not 0.67 <= r <= 1.5 else ""))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--registry", default=D.REGISTRY)
    ap.add_argument("--source", action="append", help="registered source name(s); default all")
    ap.add_argument("--acts")
    ap.add_argument("--meta")
    ap.add_argument("--kind", choices=["position", "end"], default="position")
    ap.add_argument("--pos-col", default="pos")
    ap.add_argument("--name-col", default=None)
    ap.add_argument("--layers", type=int, nargs="+", default=[-1, 0, 4, 8, 12, 16, 20, 24, 27])
    ap.add_argument("--layer-convention", default="hidden_states[l+1]")
    ap.add_argument("--prefix", default="tokenize", help="meta:<col> | tokenize | trace")
    ap.add_argument("--name", default="new")
    ap.add_argument("--min-problems", type=int, default=40)
    a = ap.parse_args()

    reg = D.load_registry(a.registry)
    traces = D.load_traces(a.data, reg)
    todo = [s for s in reg["sources"] if not a.source or s["name"] in a.source]
    extra = None
    if a.acts or a.meta:
        if not (a.acts and a.meta):
            sys.exit("--acts and --meta go together")
        extra = adhoc_entry(a)
        todo = [extra]
    ref = None
    for s in reg["sources"]:  # reference norms from the first registered pair present
        p = os.path.join(a.data, s["acts"])
        if os.path.exists(p) and os.path.exists(os.path.join(a.data, s["meta"])):
            ref = (s, D.layer_norms(np.load(p, mmap_mode="r")))
            break
    all_ok = True
    for s in todo:
        ok = check_one(s, a.data, traces, reg, ref if ref and ref[0]["name"] != s["name"] else None)
        all_ok &= ok

    if not all_ok:
        sys.exit("\nNOT USABLE: fix the failures above before registering or running on this pair")
    reg2 = copy.deepcopy(reg)
    if extra:
        reg2["sources"].append(extra)
        print("\nregistry entry to append to src/probe_cmp/registry.json:")
        print(json.dumps(extra, indent=2))
    store = D.Store(a.data, reg=reg2, log=lambda *_: None)
    print(f"\ncheckpoints with every present source: {store.ckpts}")
    print("fixed population per candidate checkpoint set (eligible, present at all, mixed problems):")
    for c in D.candidate_sets(store):
        flag = "" if c["mixed_problems"] >= a.min_problems else f"   <-- under {a.min_problems}"
        print(f"  {','.join(c['checkpoints']):<60} traces {c['traces']:>4}  mixed problems "
              f"{c['mixed_problems']:>3}  pairs {c['pairs']:>4}{flag}")
    print("\nall checks passed")


if __name__ == "__main__":
    main()
