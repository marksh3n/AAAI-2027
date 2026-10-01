# audit recovery_<tag>.jsonl from recover.py: token alignment, settings, truncation.
# ends with a verdict on whether it can serve as pilot data for recovery v2.
import argparse, json, os
import pandas as pd

from common import MAX_NEW, build_prompt, canonical, encode_trace, load_tokenizer, prompt_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="B")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--gen-temp", type=float, default=0.7, help="tag-B temperature to compare to")
    args = parser.parse_args()
    d, tag = args.data_dir, args.tag

    traces = {t["trace_id"]: t for t in map(json.loads, open(f"{d}/traces_{tag}.jsonl"))}
    meta = pd.read_parquet(f"{d}/meta_{tag}.parquet")
    rec = pd.DataFrame(map(json.loads, open(f"{d}/recovery_{tag}.jsonl")))
    tokenizer = load_tokenizer()
    problems = []

    print("== alignment ==")
    keys = rec[["trace_id", "pos"]].drop_duplicates()
    bad_text, drift = 0, []
    for tid, pos in keys.itertuples(index=False):
        t = traces[tid]
        ids, _ = encode_trace(tokenizer, t["text"])
        if not canonical(t, ids):
            drift.append(tid)
        prefix = tokenizer.decode(ids[:pos])
        stored = meta[(meta.trace_id == tid) & (meta.pos == pos)].prefix
        if not len(stored) or stored.iloc[0] != prefix or not t["text"].startswith(prefix):
            bad_text += 1
    n_keys = len(keys)
    print(f"{n_keys - bad_text}/{n_keys} prefixes: decode(retok[:pos]) == meta prefix and is a "
          "string prefix of the trace")
    print(f"{n_keys - len(drift)}/{n_keys} source traces retokenize to their sampled length "
          f"(n_gen); drift: {drift or 'none'}")
    if bad_text:
        problems.append(f"{bad_text} prefixes misaligned with the trace text")
    if drift:
        problems.append(f"{len(drift)} source traces whose retokenized ids are not the sampled ones")

    sample = traces[rec.trace_id.iloc[0]]
    rendered = build_prompt(tokenizer, sample["problem"])
    used = tokenizer(rendered, add_special_tokens=False)["input_ids"]  # recover.py via encode()
    vllm_like = prompt_ids(tokenizer, sample["problem"])  # generate_vllm.py's string path
    bos = tokenizer.bos_token_id
    lead = lambda ids: next(i for i, x in enumerate(ids) if x != bos)
    print(f"prompt BOS count: recover.py {lead(used)}, tag-B generation (vLLM string path) "
          f"{lead(vllm_like)}")
    if used != vllm_like:
        problems.append(f"prompt tokens differ from generation ({lead(used)} vs {lead(vllm_like)} BOS)")

    print("\n== settings ==")
    cfg_path = f"{d}/recovery_{tag}.config.json"
    if os.path.exists(cfg_path):
        cfg = json.load(open(cfg_path))["config"]
        print(f"config: {cfg}")
        if cfg.get("temp") != args.gen_temp:
            problems.append(f"temp {cfg.get('temp')} != generation {args.gen_temp}")
    else:
        print(f"{cfg_path} missing: temperature and top_p cannot be verified")
        problems.append("sampling temperature unrecorded")
    full = (MAX_NEW - rec.pos).unique()
    print(f"budget per continuation {sorted(rec.budget.unique())}, tag-B remaining budget "
          f"{sorted(int(x) for x in full)}")
    if (rec.budget < MAX_NEW - rec.pos).any():
        problems.append(f"continuations capped at {int(rec.budget.max())}, not {MAX_NEW} - pos")

    print("\n== summary ==")
    per = rec.groupby(["trace_id", "pos"])
    src = [traces[t] for t in rec.trace_id.unique()]
    print(f"{len(rec)} continuations, {per.ngroups} prefixes over {rec.trace_id.nunique()} traces "
          f"({sum(t['is_correct'] for t in src)} correct, {sum(t['trunc'] for t in src)} truncated)")
    print(f"positions {sorted(int(p) for p in rec.pos.unique())}, samples per prefix "
          f"{sorted(int(x) for x in per.size().unique())}")
    print(f"capped (hit the budget) {rec.capped.mean():.1%}, trunc {rec.trunc.mean():.1%}, "
          f"capped yet scored correct {int((rec.capped & rec.is_correct).sum())}")
    print(f"continuation length: median {rec.n_gen.median():.0f}, max {rec.n_gen.max()}")
    summ = f"{d}/recovery_summary_{tag}.csv"
    if os.path.exists(summ):
        s = pd.read_csv(summ)
        over = s[s.recovery_rate_terminated > 1]
        if len(over):
            print(f"recovery_summary: recovery_rate_terminated > 1 for {list(over.trace_id)} "
                  "(numerator counts capped-but-correct samples)")
    if rec.pos.nunique() < 2:
        problems.append("a single checkpoint, so no r(t) curve")
    if per.size().max() < 8:
        problems.append(f"only {per.size().max()} samples per prefix")
    if rec.capped.mean() > 0.05:
        problems.append(f"{rec.capped.mean():.0%} of continuations capped, which biases r down")

    print("\n== verdict ==")
    if problems:
        print("NOT usable as pilot data:")
        for p in problems:
            print(f"  - {p}")
        print("still useful as a rough reference for continuation length at pos 128")
    else:
        print("usable as pilot data")


if __name__ == "__main__":
    main()
