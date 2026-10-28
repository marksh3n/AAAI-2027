# confirmation outputs -> the bundle analyze.py reads. CPU (tokenizer only), after every GPU step.
#   --sources   run_rollouts output of the t0 records, round 0, --save-token-ids full
#   --bank      run_rollouts output: t0 records round 1 (K_T0 each) and the live cuts round 0 (K_PREFIX)
#   --feats     gpu_prep.py extract output (layer-20 states, decoded prefixes, live / pre-answer flags)
#   --monitors  fit_monitors.py output; scores are written into the bundle with each frozen threshold
#   python src/confirm/bundle.py --sources S --bank R --t0 T --feats F --monitors M --out <bundle>
import argparse, json, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import spec  # noqa: E402
import bundle_lib as B  # noqa: E402
from gpu_prep import read_rows, sources_table  # noqa: E402


def bank_samples(rows):
    out = []
    for r in rows:
        t0 = r["checkpoint"] == "t0"
        if t0 and r["round"] != 1:
            continue  # round 0 of a t0 record is a source trace, never a bank sample
        cut = 0 if t0 else int(r["cut_token_idx"])
        if not t0 and r["max_tokens"] != spec.MAX_NEW - cut:
            raise SystemExit(f"{r['prefix_id']}: max_tokens {r['max_tokens']} != {spec.MAX_NEW} - {cut}")
        for s in r["samples"]:
            out.append({"problem_id": r["problem_id"], "trace_id": "" if t0 else r["trace_id"], "checkpoint": cut,
                        "sample_idx": s["sample_idx"], "correct": int(bool(s["correct"])), "n_gen": s["n_gen"],
                        "answer": s["answer"]})
    return pd.DataFrame(out)


def same_engine(sources, bank, smoke=False):
    # run_rollouts writes <out>.config.json; sources and bank must agree on everything that samples
    cs, cb = (json.load(open(f"{p}.config.json"))["config"] for p in (sources, bank))
    drop = lambda c: {k: v for k, v in c.items() if k != "save_token_ids"}
    want = {"model": spec.MODEL, "temp": spec.TEMPERATURE, "top_p": spec.TOP_P, "revision": spec.MODEL_REVISION,
            "vllm": spec.VLLM_VERSION}
    bad = [k for k, v in want.items() if cs.get(k) != v or cb.get(k) != v]
    if drop(cs) != drop(cb) or (bad and not smoke) or (not smoke and (cs["max_tokens_cap"] or cb["max_tokens_cap"])):
        raise SystemExit(f"sources and bank were not sampled under the frozen engine: {cs} vs {cb} (spec {want})")


def main():
    ap = argparse.ArgumentParser()
    for k in ("sources", "bank", "t0", "feats", "monitors", "out"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--kind", default="confirmation", choices=["confirmation", "smoke"])
    a = ap.parse_args()
    same_engine(a.sources, a.bank, smoke=a.kind == "smoke")
    src = sources_table(a.sources)
    bank = bank_samples(read_rows(a.bank))
    feats = pd.read_parquet(os.path.join(a.feats, "feats.parquet"))
    acts = np.load(os.path.join(a.feats, f"acts_L{spec.LAYER}.npy")).astype(np.float32)
    want = spec.JADEN["samples_per_problem"]
    if a.kind == "confirmation" and want is not None and (src.groupby("problem_id").size() != want).any():
        raise SystemExit(f"not every problem has {want} source traces")

    src_keys, vote_rows = {}, []
    t0 = bank[bank.checkpoint == 0]
    for p, s in src.groupby("problem_id"):
        v = t0[t0.problem_id == p].sort_values("sample_idx")
        keys = B.cluster_keys(list(s.answer) + list(v.answer), list(s.correct.astype(bool)) + list(v.correct.astype(bool)))
        src_keys.update(dict(zip(s.trace_id, keys[: len(s)])))
        vote_rows += [{"problem_id": p, "sample_idx": int(i), "answer_key": k, "correct": int(c), "n_gen": int(n)}
                      for i, k, c, n in zip(v.sample_idx, keys[len(s):], v.correct, v.n_gen)]
    sources = src[["trace_id", "problem_id", "n", "correct"]].assign(answer_key=src.trace_id.map(src_keys))

    full = pd.DataFrame([{"trace_id": t, "checkpoint": c} for t in src.trace_id for c in sorted(feats.checkpoint.unique())])
    flags = ["live", *B.PRE_COLS]
    cuts = full.merge(feats[["trace_id", "checkpoint", *flags]], how="left", on=["trace_id", "checkpoint"])
    cuts[flags] = cuts[flags].fillna(False).astype(bool)

    monitors, man = B.load_monitors(a.monitors)
    parts = []
    for c, g in feats.groupby("checkpoint"):
        parts.append(B.score_rows(monitors, int(c), g.trace_id.tolist(), acts[g.row.values], g.prefix_text.tolist(),
                                  np.full(len(g), float(c))))
    prov = {"sources": spec.file_sha(a.sources), "bank": spec.file_sha(a.bank), "t0": spec.file_sha(a.t0),
            "feats": json.load(open(os.path.join(a.feats, "feats.json"))),
            "monitors_manifest": spec.file_sha(os.path.join(a.monitors, "manifest.json")),
            "source_config_hashes": sorted(src.config_hash.unique())}
    meta = B.write(a.out, a.kind, prov, sources, cuts, B.bank_table(bank.drop(columns="answer")), pd.DataFrame(vote_rows),
                   pd.concat(parts, ignore_index=True))
    print(json.dumps({k: v for k, v in meta.items() if k not in ("files", "provenance")}, indent=1))


if __name__ == "__main__":
    main()
