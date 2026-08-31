import argparse, json, os
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from generate import MODEL, build_prompt, out_path
from taxonomy import RESTART

TRACES_PATH = "data/traces.jsonl"
ACTS_PATH = "data/acts.npy"
META_PATH = "data/meta.parquet"
PARTS_DIR = "data/parts"
LAYERS = [-1, 0, 4, 8, 12, 16, 20, 24, 27]  # decoder block index; -1 is the embedding output
# absolute generation offsets; length-so-far is constant across traces
POSITIONS = [128, 256, 512, 1024, 2048, 4096]
MIN_STEP_TOKENS = 5
CHUNK = 1024


def paths(tag):
    s = f"_{tag}" if tag else ""
    return out_path(tag), f"data/acts{s}.npy", f"data/meta{s}.parquet", f"data/parts{s}"


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
    spans = step_spans(text)
    think, answer = spans[:-1], spans[-1]
    _, counts = last_token_per_span(think, offsets)
    spans = merge_short(think, counts) + [answer]
    last, _ = last_token_per_span(spans, offsets)
    keep = [(s, i) for s, i in zip(spans, last) if i is not None]
    return [s for s, i in keep], [i for s, i in keep]


def encode(tokenizer, trace):
    # must match generation or offsets shift
    prompt = build_prompt(tokenizer, trace["problem"], trace.get("structured", False))
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    enc = tokenizer(trace["text"], add_special_tokens=False, return_offsets_mapping=True)
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


def fixed(tokenizer, model, device, traces, acts_path, meta_path, parts_dir):
    os.makedirs(parts_dir, exist_ok=True)
    ids = {}
    for n, trace in enumerate(traces):
        prompt_ids, trace_ids, _ = encode(tokenizer, trace)
        ids[n] = trace_ids
        part = f"{parts_dir}/{n}.npy"
        if os.path.exists(part):
            continue
        idx = [p for p in POSITIONS if p < len(trace_ids)]
        # attention is causal, so state at position p ignores everything after it: feeding
        # the tail of a 32k trace only to read position 4096 is wasted compute
        np.save(part, activations(model, device, prompt_ids, trace_ids[: max(POSITIONS) + 1], idx))
        if n % 25 == 0:
            print(f"{n}/{len(traces)}", flush=True)

    rows, acts = [], []
    for n, trace in enumerate(traces):
        h = np.load(f"{parts_dir}/{n}.npy")
        for j, pos in enumerate([p for p in POSITIONS if p < len(ids[n])]):
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
    pd.DataFrame(rows).to_parquet(meta_path)
    print(f"{len(rows)} rows, acts {acts.shape}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--fixed", action="store_true")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    traces_path, acts_path, meta_path, parts_dir = paths(args.tag)
    traces = [json.loads(line) for line in open(traces_path)]
    n_trunc = sum(t["trunc"] for t in traces)
    print(f"truncated {n_trunc}/{len(traces)} ({n_trunc / len(traces):.1%}), counted as incorrect")
    kept = traces
    for t in kept:
        if t["trunc"]:
            t["is_correct"] = False

    tokenizer, model, device = load_model()
    if args.check:
        check(tokenizer, model, device, kept[0])
        return
    if args.fixed:
        fixed(tokenizer, model, device, kept, acts_path, meta_path, parts_dir)
        return

    # one file per trace so a crash or a colab disconnect costs one trace, not the run
    os.makedirs(PARTS_DIR, exist_ok=True)
    for n, trace in enumerate(kept):
        part = f"{PARTS_DIR}/{n}.npy"
        if os.path.exists(part):
            continue
        prompt_ids, trace_ids, offsets = encode(tokenizer, trace)
        spans, token_idx = segment(trace["text"], offsets)
        np.save(part, activations(model, device, prompt_ids, trace_ids, token_idx))
        if n % 25 == 0:
            print(f"{n}/{len(kept)}", flush=True)

    rows, acts = [], []
    for n, trace in enumerate(kept):
        h = np.load(f"{PARTS_DIR}/{n}.npy")
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
                }
            )
        acts.append(h)

    acts = np.concatenate(acts)
    np.save(ACTS_PATH, acts)
    pd.DataFrame(rows).to_parquet(META_PATH)
    print(f"{len(rows)} steps, acts {acts.shape}")


if __name__ == "__main__":
    main()
