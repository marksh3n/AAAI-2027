import argparse, json, os, time
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cache import (
    StaleCache,
    load_manifest,
    manifest_path,
    part_ok,
    save_manifest,
    sha_file,
    sha_obj,
    short,
    write_dataset_manifest,
)
from generate import MODEL, build_prompt, out_path, recorded_token_inputs, tokenizer_identity
from taxonomy import RESTART

TRACES_PATH = "data/traces.jsonl"
ACTS_PATH = "data/acts.npy"
META_PATH = "data/meta.parquet"
PARTS_DIR = "data/parts"
LAYERS = [-1, 0, 4, 8, 12, 16, 20, 24, 27]  # decoder block index; -1 is the embedding output
# absolute generation offsets, defined as TOKENS OBSERVED (a 1-indexed count).
# pos=128 means "after seeing 128 tokens" -> hidden state at trace_ids[127],
# text prefix trace_ids[:128]. Both then reflect exactly the same 128 tokens.
POSITIONS = [128, 256, 512, 1024, 2048, 4096, 6144, 8192, 12288, 16384, 24576, 32768]
MIN_STEP_TOKENS = 5
CHUNK = 1024
# bump when the extraction convention changes; invalidates every cached part
EXTRACTOR_VERSION = 3
# fp16 and CHUNK shift the last bits; a wrong trace shifts direction
VERIFY_MIN_COS = 0.99
VERIFY_N = 8
# whole-manifest rewrite is slow on Drive FUSE; caps re-extraction after a crash
MANIFEST_EVERY = 10


def paths(tag):
    s = f"_{tag}" if tag else ""
    return out_path(tag), f"data/acts{s}.npy", f"data/meta{s}.parquet", f"data/parts{s}"


def config_fingerprint(tokenizer, device, mode):
    # rendered prompts cover chat template, INSTRUCTION and STRUCTURED
    canary = "What is 1+1?"
    cfg = {
        "extractor_version": EXTRACTOR_VERSION,
        "mode": mode,
        "model": MODEL,
        "layers": list(LAYERS),
        "positions": list(POSITIONS) if mode == "fixed" else None,
        "min_step_tokens": None if mode == "fixed" else MIN_STEP_TOKENS,
        "compute_dtype": "float16" if device == "cuda" else "float32",
        "stored_dtype": "float16",
        "tokenizer": str(getattr(tokenizer, "name_or_path", "")),
        "tokenizer_sha": tokenizer_identity(tokenizer),
        "vocab_size": len(tokenizer),
        "prompt_plain": build_prompt(tokenizer, canary, False),
        "prompt_structured": build_prompt(tokenizer, canary, True),
    }
    return cfg, sha_obj(cfg)


def trace_sha(trace):
    # trace_id survives regeneration; the text under it does not
    return sha_obj(
        {
            "problem": trace["problem"],
            "text": trace["text"],
            "structured": trace.get("structured", False),
        }
    )


def expect_part(trace, trace_ids, token_idx, prompt_ids=None):
    return {
        "trace_id": trace["trace_id"],
        "trace_sha": trace_sha(trace),
        "n_trace_tokens": len(trace_ids),
        "token_idx": list(token_idx),
        # Include actual ordered inputs: unchanged text/count does not prove identical tokens/BOS.
        "prompt_ids_sha": sha_obj(prompt_ids) if prompt_ids is not None else None,
        "trace_ids_sha": sha_obj(trace_ids),
    }


def fixed_positions(trace_ids):
    # pos is tokens observed (1-indexed count): need at least pos tokens to exist, so the
    # cutoff is <=, not <. The hidden state for a pos-token prefix lives at index pos - 1.
    # attention is causal, so state at position p ignores everything after it: feeding the
    # tail of a 32k trace only to read position 4096 is wasted compute.
    positions = [p for p in POSITIONS if p <= len(trace_ids)]
    return positions, [p - 1 for p in positions], trace_ids[: max(POSITIONS) + 1]


def stamp_part(manifest, n, path, entry):
    # the written file's own shape and bytes, so the entry describes what is on disk
    entry = dict(entry)
    entry["shape"] = list(np.load(path, mmap_mode="r").shape)
    entry["sha256"] = sha_file(path)
    manifest["parts"][str(n)] = entry
    return entry


def shared_prefix(seqs):
    # tokens shared by every sample of a problem
    n = 0
    while all(len(s) > n and s[n] == seqs[0][n] for s in seqs):
        n += 1
    return n


def step_spans(text):
    # char spans: one per \n\n paragraph inside <think>, plus one for everything after it.
    # a truncated trace has no </think>, so it is all thinking and gets no answer step.
    think_end = text.find("</think>")
    spans = []
    pos = 0
    for chunk in text[: think_end if think_end >= 0 else len(text)].split("\n\n"):
        if chunk.strip():
            spans.append((pos, pos + len(chunk)))
        pos += len(chunk) + 2
    if think_end >= 0:
        spans.append((think_end, len(text)))
    return spans


def last_token_per_span(spans, offsets):
    starts = np.array([a for a, b in offsets])
    last, counts = [], []
    for span_start, span_end in spans:
        # by start, not end: the token holding a paragraph's last char is usually ".\n\n",
        # which runs past the boundary, and that is the token we want the state at
        inside = np.where((starts >= span_start) & (starts < span_end))[0]
        last.append(int(inside[-1]) if len(inside) else None)
        counts.append(len(inside))
    return last, counts


def merge_short(spans, counts):
    merged = []
    for span, count in zip(spans, counts):
        if merged and count < MIN_STEP_TOKENS:
            merged[-1] = (merged[-1][0], span[1])
        else:
            merged.append(span)
    return merged


def segment(text, offsets):
    if offsets is None:
        raise ValueError("Recorded generation IDs differ from text retokenization; character-based step offsets are unverified. Use fixed checkpoints or resolve token/offset alignment.")
    spans = step_spans(text)
    think, answer = spans[:-1], spans[-1]
    _, counts = last_token_per_span(think, offsets)
    spans = merge_short(think, counts) + [answer]
    last, _ = last_token_per_span(spans, offsets)
    keep = [(s, i) for s, i in zip(spans, last) if i is not None]
    return [s for s, i in keep], [i for s, i in keep]


def input_source(trace):
    if "token_input_provenance" in trace:
        return "recorded_at_generation"
    return "reconstructed_explicit_bos_policy" if "n_bos" in trace else "reconstructed_legacy_default"


def encode(tokenizer, trace, legacy_n_bos=None):
    # Keep the historical one-BOS default for old wrappers, which adjust BOS themselves.
    # New recorded IDs take precedence. An old wrapper's one-BOS assertion fails safely
    # for a recorded two-BOS prompt instead of adding a third BOS silently.
    stored = recorded_token_inputs(tokenizer, trace)
    enc = tokenizer(trace["text"], add_special_tokens=False, return_offsets_mapping=True)
    selected_bos = legacy_n_bos if legacy_n_bos is not None else trace.get("n_bos")
    if selected_bos is not None and (type(selected_bos) is not int or selected_bos not in (1, 2)):
        raise ValueError("Explicit n_bos policy must be integer 1 or 2")
    if stored is not None:
        prompt_ids, trace_ids = stored
        if selected_bos is not None:
            actual = next((i for i, tok in enumerate(prompt_ids) if tok != tokenizer.bos_token_id), len(prompt_ids))
            if actual != selected_bos:
                raise ValueError("Explicit BOS policy conflicts with recorded generation IDs")
        offsets = enc["offset_mapping"] if enc["input_ids"] == trace_ids else None
        return prompt_ids, trace_ids, offsets
    prompt = build_prompt(tokenizer, trace["problem"], trace.get("structured", False))
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    if selected_bos is not None:
        bos = tokenizer.bos_token_id
        if bos is None:
            raise ValueError("Cannot apply BOS policy: tokenizer has no BOS token")
        n = next((i for i, tok in enumerate(prompt_ids) if tok != bos), len(prompt_ids))
        if n == 0:
            raise ValueError("Rendered legacy prompt has no leading BOS; investigate template")
        prompt_ids = [bos] * selected_bos + prompt_ids[n:]
    return prompt_ids, enc["input_ids"], enc["offset_mapping"]


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    model.eval()
    return tokenizer, model, device


def activations(model, device, prompt_ids, trace_ids, token_idx):
    # fed in chunks with a kv cache rather than one forward: a 30k-token sequence makes
    # attention allocate seq x seq, which OOMs a T4. causal attention makes this identical.
    # model.model skips the lm_head; the logits would be a huge tensor we never look at
    if any(type(i) is not int or not 0 <= i < len(trace_ids) for i in token_idx):
        raise ValueError("Activation token index is outside the supplied trace prefix")
    ids = prompt_ids + trace_ids
    picks = [len(prompt_ids) + i for i in token_idx]
    want = set(picks)
    grabbed = {}
    cache = None
    with torch.no_grad():
        for start in range(0, len(ids), CHUNK):
            piece = torch.tensor([ids[start : start + CHUNK]], device=device)
            out = model.model(
                piece, past_key_values=cache, use_cache=True, output_hidden_states=True
            )
            cache = out.past_key_values
            for pos in want:
                if start <= pos < start + piece.shape[1]:
                    grabbed[pos] = torch.stack(
                        [out.hidden_states[layer + 1][0, pos - start] for layer in LAYERS]
                    ).half().cpu()
            del out
    del cache
    return torch.stack([grabbed[p] for p in picks]).numpy()


def check(tokenizer, model, device, trace):
    prompt_ids, trace_ids, offsets = encode(tokenizer, trace)
    spans, token_idx = segment(trace["text"], offsets)
    ids = torch.tensor([prompt_ids + trace_ids], device=device)
    with torch.no_grad():
        logits = model(ids).logits
    pred = logits[0, :-1].argmax(-1)
    target = ids[0, 1:]
    region = slice(len(prompt_ids) - 1, None)
    match = (pred[region] == target[region]).float().mean().item()
    print(f"teacher forcing: argmax matches next token on {match:.1%} of the trace")
    print(f"{len(spans)} steps, {len(trace_ids)} trace tokens\n")
    print("step | text at span end        | text decoded up to chosen token")
    for i, (span, idx) in enumerate(zip(spans, token_idx)):
        # rstrip both: a step's last token is usually ".\n\n", which runs past the span end
        want = trace["text"][span[0] : span[1]].rstrip()[-24:]
        got = tokenizer.decode(trace_ids[: idx + 1]).rstrip()
        flag = "" if got.endswith(want) else "   <-- MISMATCH"
        got = got[-24:]
        print(f"{i:4d} | {want!r:26} | {got!r}{flag}")


def warn_orphans(parts_dir, n_traces):
    # leftover parts from a longer traces.jsonl
    if not os.path.isdir(parts_dir):
        return
    orphans = [
        f for f in os.listdir(parts_dir)
        if f.endswith(".npy") and f[:-4].isdigit() and int(f[:-4]) >= n_traces
    ]
    if orphans:
        print(f"WARNING: {len(orphans)} part files index past the {n_traces} traces in this "
              f"file (e.g. {sorted(orphans)[:3]}); traces.jsonl looks like it shrank")


def cosines(a, b):
    # per (row, layer) cosine, cached vs fresh
    a = a.astype(np.float32).reshape(-1, a.shape[-1])
    b = b.astype(np.float32).reshape(-1, b.shape[-1])
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return (a * b).sum(1) / np.maximum(den, 1e-12)


def verify(tokenizer, model, device, traces, parts_dir, mode, n_verify, adopt, seed=0):
    # check a pre-manifest parts dir: re-extract a sample and compare
    if adopt:
        raise StaleCache("--adopt is disabled for extractor v3: cosine agreement cannot establish exact historical token/BOS provenance. Use --verify for diagnostics and --rebuild to write fresh input-bound activations.")
    cfg, cfg_hash = config_fingerprint(tokenizer, device, mode)
    print(f"verifying {parts_dir} against config {short(cfg_hash)} (mode {mode})\n", flush=True)

    want, plan, missing, broken = {}, {}, [], []
    for n, trace in enumerate(traces):
        prompt_ids, trace_ids, offsets = encode(tokenizer, trace)
        if mode == "fixed":
            _, token_idx, feed = fixed_positions(trace_ids)
        else:
            _, token_idx = segment(trace["text"], offsets)
            feed = trace_ids
        want[n] = expect_part(trace, trace_ids, token_idx, prompt_ids)
        plan[n] = (prompt_ids, feed, token_idx)
        part = f"{parts_dir}/{n}.npy"
        if not os.path.exists(part):
            missing.append(n)
            continue
        shape = np.load(part, mmap_mode="r").shape
        if list(shape[:2]) != [len(token_idx), len(LAYERS)]:
            broken.append((n, f"shape {tuple(shape)}, expected ({len(token_idx)}, {len(LAYERS)}, ...)"))

    skip = {n for n, _ in broken} | set(missing)
    usable = [n for n in range(len(traces)) if n not in skip]
    print(f"{len(usable)}/{len(traces)} parts present with the expected shape, "
          f"{len(missing)} missing, {len(broken)} wrong shape", flush=True)
    for n, why in broken[:10]:
        print(f"  part {n} ({traces[n]['trace_id']}): {why}")
    if len(broken) > 10:
        print(f"  ... and {len(broken) - 10} more")
    warn_orphans(parts_dir, len(traces))
    if not usable:
        print("\nnothing to verify", flush=True)
        return

    rng = np.random.default_rng(seed)
    sample = sorted(rng.choice(usable, min(n_verify, len(usable)), replace=False).tolist())
    print(f"\nre-extracting {len(sample)} sampled parts and comparing against the cache", flush=True)
    print(f"{'part':>6}{'trace_id':>12}{'min cos':>10}{'max |d|':>10}{'exact':>8}  verdict", flush=True)
    worst = 1.0
    for n in sample:
        prompt_ids, feed, token_idx = plan[n]
        fresh = activations(model, device, prompt_ids, feed, token_idx)
        cached = np.load(f"{parts_dir}/{n}.npy")
        cos = cosines(cached, fresh)
        exact = float((cached == fresh).mean())
        mx = float(np.abs(cached.astype(np.float32) - fresh.astype(np.float32)).max())
        worst = min(worst, float(cos.min()))
        print(f"{n:>6}{traces[n]['trace_id']:>12}{cos.min():>10.4f}{mx:>10.3f}{exact:>8.1%}"
              f"  {'ok' if cos.min() >= VERIFY_MIN_COS else 'MISMATCH'}", flush=True)

    print(f"\nworst cosine {worst:.4f} over {len(sample)} parts (threshold {VERIFY_MIN_COS})", flush=True)
    if worst < VERIFY_MIN_COS or broken:
        print("NOT VERIFIED: this cache does not reproduce. Re-extract with --rebuild; any "
              "result already computed from it should be treated as unverified.", flush=True)
        return
    print("the sampled parts reproduce a fresh forward pass", flush=True)
    print("Diagnostic agreement does not establish exact historical inputs. Use --rebuild to create an input-bound cache.")


def fixed(tokenizer, model, device, traces, acts_path, meta_path, parts_dir, rebuild=False):
    cfg, cfg_hash = config_fingerprint(tokenizer, device, "fixed")
    manifest = load_manifest(parts_dir, cfg, cfg_hash, rebuild)
    print(f"config {short(cfg_hash)}, {len(manifest['parts'])} parts already in the manifest",
          flush=True)
    save_manifest(parts_dir, manifest)  # exists from the start, so a crash leaves it readable

    ids, prompts, want, pos_by_n, reused, written = {}, {}, {}, {}, 0, 0
    for n, trace in enumerate(traces):
        prompt_ids, trace_ids, _ = encode(tokenizer, trace)
        ids[n] = trace_ids
        prompts[n] = prompt_ids
        positions, token_idx, feed = fixed_positions(trace_ids)
        pos_by_n[n] = positions
        want[n] = expect_part(trace, trace_ids, token_idx, prompt_ids)
        part = f"{parts_dir}/{n}.npy"
        # reuse needs manifest proof: this trace, these indices, this config
        why = part_ok(manifest["parts"].get(str(n)), part, want[n])
        if why is None:
            reused += 1
            continue
        if os.path.exists(part):
            print(f"re-extracting {n} ({trace['trace_id']}): {why}", flush=True)
        np.save(part, activations(model, device, prompt_ids, feed, token_idx))
        stamp_part(manifest, n, part, want[n])
        written += 1
        if written % MANIFEST_EVERY == 0:
            save_manifest(parts_dir, manifest)
        if n % 25 == 0:
            print(f"{n}/{len(traces)}", flush=True)
    save_manifest(parts_dir, manifest)
    print(f"reused {reused}/{len(traces)} validated parts, extracted {written}", flush=True)
    warn_orphans(parts_dir, len(traces))

    rows, acts = [], []
    for n, trace in enumerate(traces):
        part = f"{parts_dir}/{n}.npy"
        # re-checked: assembly is what pairs a part with a label row
        why = part_ok(manifest["parts"].get(str(n)), part, want[n])
        if why is not None:
            raise StaleCache(f"part {n} ({trace['trace_id']}) failed validation at assembly: {why}")
        h = np.load(part)
        positions = pos_by_n[n]
        # h[j] was extracted at token index positions[j] - 1, and prefix below is
        # ids[n][:positions[j]] -- both must represent the same pos-token prefix.
        assert len(h) == len(positions), (
            f"trace {n}: {len(h)} activations but {len(positions)} positions "
            f"({len(ids[n])} trace tokens) -- activation/text prefixes are out of sync"
        )
        for pos in positions:
            prefix = tokenizer.decode(ids[n][:pos])
            rows.append(
                {
                    "trace_id": trace["trace_id"],
                    "problem_id": trace["problem_id"],
                    "level": trace["level"],
                    "pos": pos,
                    "is_correct": trace["is_correct"],
                    "n_gen": trace["n_gen"],
                    "prefix": prefix,
                    "restarts": len(RESTART.findall(prefix)) / pos * 1000,
                    "row": len(rows),
                    "prefix_sha": sha_obj([prompts[n], ids[n][:pos]]),
                    "n_bos": next((i for i, tok in enumerate(prompts[n]) if tok != tokenizer.bos_token_id), len(prompts[n])),
                    "token_input_source": input_source(trace),
                }
            )
        acts.append(h)

    by_problem = {}
    for n, trace in enumerate(traces):
        by_problem.setdefault(trace["problem_id"], []).append(ids[n])
    sp = sorted(shared_prefix(v) for v in by_problem.values() if len(v) > 1)
    print(f"shared prefix across a problem's samples: median {sp[len(sp) // 2]}, max {max(sp)}")

    acts = np.concatenate(acts)
    np.save(acts_path, acts)
    df = pd.DataFrame(rows)
    df.to_parquet(meta_path)
    write_dataset_manifest(meta_path, acts_path, cfg, cfg_hash, df.trace_id.tolist(), acts.shape)
    print(f"{len(rows)} rows, acts {acts.shape}", flush=True)


def steps(tokenizer, model, device, traces, acts_path, meta_path, parts_dir, rebuild=False):
    cfg, cfg_hash = config_fingerprint(tokenizer, device, "steps")
    manifest = load_manifest(parts_dir, cfg, cfg_hash, rebuild)
    print(f"config {short(cfg_hash)}, {len(manifest['parts'])} parts already in the manifest",
          flush=True)
    save_manifest(parts_dir, manifest)  # exists from the start, so a crash leaves it readable

    # one file per trace so a crash or a colab disconnect costs one trace, not the run
    want, inputs, reused, written = {}, {}, 0, 0
    for n, trace in enumerate(traces):
        prompt_ids, trace_ids, offsets = encode(tokenizer, trace)
        spans, token_idx = segment(trace["text"], offsets)
        want[n] = expect_part(trace, trace_ids, token_idx, prompt_ids)
        inputs[n] = (prompt_ids, trace_ids, token_idx)
        part = f"{parts_dir}/{n}.npy"
        # token_idx in the fingerprint, so resegmentation invalidates the part
        why = part_ok(manifest["parts"].get(str(n)), part, want[n])
        if why is None:
            reused += 1
            continue
        if os.path.exists(part):
            print(f"re-extracting {n} ({trace['trace_id']}): {why}", flush=True)
        np.save(part, activations(model, device, prompt_ids, trace_ids, token_idx))
        stamp_part(manifest, n, part, want[n])
        written += 1
        if written % MANIFEST_EVERY == 0:
            save_manifest(parts_dir, manifest)
        if n % 25 == 0:
            print(f"{n}/{len(traces)}", flush=True)
    save_manifest(parts_dir, manifest)
    print(f"reused {reused}/{len(traces)} validated parts, extracted {written}", flush=True)
    warn_orphans(parts_dir, len(traces))

    rows, acts = [], []
    for n, trace in enumerate(traces):
        part = f"{parts_dir}/{n}.npy"
        why = part_ok(manifest["parts"].get(str(n)), part, want[n])
        if why is not None:
            raise StaleCache(f"part {n} ({trace['trace_id']}) failed validation at assembly: {why}")
        h = np.load(part)
        prompt_ids, trace_ids, token_idx = inputs[n]
        for step in range(len(h)):
            rows.append(
                {
                    "trace_id": trace["trace_id"],
                    "problem_id": trace["problem_id"],
                    "step_idx": step,
                    "n_steps": len(h),
                    "is_correct": trace["is_correct"],
                    "answer": trace["answer"],
                    "row": len(rows),
                    "prefix_sha": sha_obj([prompt_ids, trace_ids[:token_idx[step] + 1]]),
                    "token_input_source": input_source(trace),
                }
            )
        acts.append(h)

    acts = np.concatenate(acts)
    np.save(acts_path, acts)
    df = pd.DataFrame(rows)
    df.to_parquet(meta_path)
    write_dataset_manifest(meta_path, acts_path, cfg, cfg_hash, df.trace_id.tolist(), acts.shape)
    print(f"{len(rows)} steps, acts {acts.shape}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--fixed", action="store_true")
    parser.add_argument("--tag", default="")
    parser.add_argument("--verify", action="store_true",
                        help="re-extract a sample and compare it against the cached parts")
    parser.add_argument("--verify-n", type=int, default=VERIFY_N)
    parser.add_argument("--adopt", action="store_true",
                        help="disabled for v3: approximate verification cannot certify historical token inputs")
    parser.add_argument("--rebuild", action="store_true",
                        help="set a mismatched cache aside and extract again")
    parser.add_argument("--bos-policy", help="Explicit trace-ID-keyed legacy BOS policy JSON bound to the source trace file; never inferred automatically")
    args = parser.parse_args()
    if args.adopt:
        parser.error("--adopt is disabled for v3; use --verify for diagnostics or --rebuild for fresh input-bound activations")

    traces_path, acts_path, meta_path, parts_dir = paths(args.tag)
    traces = [json.loads(line) for line in open(traces_path)]
    if len({t["trace_id"] for t in traces}) != len(traces):
        raise ValueError("Duplicate trace IDs; resolve them before activation extraction")
    if args.bos_policy:
        policy = json.load(open(args.bos_policy))
        if policy.get("traces_sha256") != sha_file(traces_path) or not policy.get("basis"):
            raise ValueError("BOS policy lacks a basis or belongs to a different source file")
        mapping = policy.get("traces", {})
        if set(mapping) != {t["trace_id"] for t in traces} or any(type(v) is not int or v not in (1, 2) for v in mapping.values()):
            raise ValueError("BOS policy must map exactly every source trace ID to integer 1 or 2")
        for t in traces:
            t["n_bos"] = mapping[t["trace_id"]]
    legacy = sum("token_input_provenance" not in t for t in traces)
    if legacy:
        print(f"WARNING: {legacy} text-only traces use reconstructed inputs; historical generation-token identity is unverified. Explicit BOS policy: {bool(args.bos_policy)}", flush=True)
    n_trunc = sum(t["trunc"] for t in traces)
    print(f"truncated {n_trunc}/{len(traces)} ({n_trunc / len(traces):.1%}), counted as incorrect", flush=True)
    for t in traces:
        if t["trunc"]:
            t["is_correct"] = False

    tokenizer, model, device = load_model()
    if args.check:
        check(tokenizer, model, device, traces[0])
        return
    try:
        if args.verify:
            verify(tokenizer, model, device, traces, parts_dir,
                   "fixed" if args.fixed else "steps", args.verify_n, args.adopt)
        elif args.fixed:
            fixed(tokenizer, model, device, traces, acts_path, meta_path, parts_dir, args.rebuild)
        else:
            steps(tokenizer, model, device, traces, acts_path, meta_path, parts_dir, args.rebuild)
    except StaleCache as e:
        raise SystemExit(f"\nstale cache: {e}") from e


if __name__ == "__main__":
    main()
