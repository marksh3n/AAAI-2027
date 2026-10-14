# colab: pip install -q math-verify
import argparse, json, os
from collections import Counter
import numpy as np
import torch
from datasets import load_dataset
from math_verify import parse, verify
from transformers import AutoModelForCausalLM, AutoTokenizer
from cache import sha_obj

MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
OUT_PATH = "data/traces.jsonl"
INSTRUCTION = "\nPlease reason step by step, and put your final answer within \\boxed{}."
# numbered steps and brevity; \boxed{} kept so extraction is unchanged
STRUCTURED = (
    "\nSolve step by step. Number each step as 'Step 1:', 'Step 2:', and so on. "
    "Be brief. End with your final answer within \\boxed{}."
)


def extract_boxed(text):
    start = text.rfind("\\boxed{")
    if start < 0:
        return None
    i = start + 7
    depth = 1
    while i < len(text) and depth:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    return text[start + 7 : i - 1] if depth == 0 else None


def answers_match(pred, gold):
    if pred is None:
        return False
    return bool(verify(parse("$" + gold + "$"), parse("$" + pred + "$")))


def build_prompt(tokenizer, question, structured=False):
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": question + (STRUCTURED if structured else INSTRUCTION)}],
        tokenize=False,
        add_generation_prompt=True,
    )
    # DeepSeek recommends forcing the completion to open with <think>
    return text if text.rstrip().endswith("<think>") else text + "<think>\n"


def load_problems(limit):
    problems = [
        row for row in load_dataset("HuggingFaceH4/MATH-500")["test"] if row["level"] in (3, 4, 5)
    ]
    problems.sort(key=lambda row: row["unique_id"])
    for i, row in enumerate(problems):
        row["problem_id"] = i
    return problems[:limit] if limit else problems


def load_pilot(n_fill=21, seed=0):
    # failed problems resample mixed; fillers level-matched so level cannot proxy the label
    problems = {p["problem_id"]: p for p in load_problems(0)}
    traces = [json.loads(line) for line in open(OUT_PATH)]
    wrong = [t for t in traces if t["trunc"] or not t["is_correct"]]
    picked = {t["problem_id"] for t in wrong}
    rng = np.random.default_rng(seed)
    for level, count in sorted(Counter(t["level"] for t in wrong).items()):
        pool = sorted(
            t["problem_id"] for t in traces if t["level"] == level and t["problem_id"] not in picked
        )
        k = min(round(n_fill * count / len(wrong)), len(pool))
        picked |= set(rng.choice(pool, k, replace=False).tolist())
    return [problems[i] for i in sorted(picked)]


def already_done(path):
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        return {json.loads(line)["trace_id"] for line in f}


def out_path(tag):
    return f"data/traces_{tag}.jsonl" if tag else OUT_PATH


def add_args(parser):
    parser.add_argument("--n", type=int, default=0)
    parser.add_argument("--max-new", type=int, default=32768)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--temp", type=float, default=0.6)
    parser.add_argument("--tag", default="")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--structured", action="store_true")
    return parser


def select(args):
    problems = load_pilot() if args.pilot else load_problems(0)
    if args.n:
        problems = problems[: args.n]
    done = already_done(out_path(args.tag))
    # samples written per problem block, so sample 0 marks it done
    todo = [p for p in problems if f"{p['problem_id']}_0" not in done]
    print(f"{len(todo)}/{len(problems)} problems left, {args.samples} samples each")
    return todo


def tokenizer_identity(tokenizer):
    """Identity of the tokenizer actually used, not only its mutable model name."""
    backend = getattr(tokenizer, "backend_tokenizer", None)
    backend_config = json.loads(backend.to_str()) if backend is not None else None
    if backend_config is not None:
        # Fast-tokenizer calls persist these execution settings. Batch generation
        # may enable padding while extraction tokenizes single unpadded strings.
        # Neither setting changes the recorded nonpadding input IDs themselves.
        backend_config.pop("padding", None)
        backend_config.pop("truncation", None)
    return sha_obj({
        "name": getattr(tokenizer, "name_or_path", ""),
        "vocab": tokenizer.get_vocab(),
        "backend": backend_config,
        # HF generation may alias an existing EOS token as PAD. Vocabulary and
        # backend added-token entries still detect actual token changes.
        "special_tokens": {k: str(v) for k, v in getattr(tokenizer, "special_tokens_map", {}).items()
                           if k != "pad_token"},
        "chat_template": getattr(tokenizer, "chat_template", None),
    })


def token_source_sha(trace):
    return sha_obj({"problem": trace["problem"], "text": trace["text"],
                    "structured": trace.get("structured", False)})


def _token_list(ids, name):
    if not isinstance(ids, list) or any(type(i) is not int or i < 0 for i in ids):
        raise ValueError(f"{name} must be an ordered list of nonnegative integer token IDs")
    return ids


def attach_token_inputs(trace, tokenizer, prompt_ids, completion_ids):
    """Save actual generation IDs. Text-span IDs exclude a terminal EOS only."""
    prompt_ids = _token_list(list(prompt_ids), "prompt_token_ids")
    completion_ids = _token_list(list(completion_ids), "completion_token_ids")
    generated_ids = completion_ids[:]
    if generated_ids and generated_ids[-1] == tokenizer.eos_token_id:
        generated_ids.pop()
    if tokenizer.decode(generated_ids, skip_special_tokens=True) != trace["text"]:
        raise ValueError("Actual generation token IDs do not decode to the saved text; do not retokenize to hide this mismatch")
    trace.update(prompt_token_ids=prompt_ids, generated_token_ids=generated_ids,
                 completion_token_ids=completion_ids)
    trace["token_input_provenance"] = {
        "schema": 1, "capture": "recorded_at_generation", "source_sha": token_source_sha(trace),
        "tokenizer_sha": tokenizer_identity(tokenizer),
        "prefix_sha": sha_obj([prompt_ids, generated_ids]),
        "text_token_convention": "actual completion IDs excluding terminal EOS",
    }
    return trace


def recorded_token_inputs(tokenizer, trace):
    """Return validated recorded IDs, or None for legacy text-only records.

    These hashes detect changed records; they do not authenticate historical data
    if someone reconstructs IDs and fabricates a provenance record after the fact.
    """
    keys = ("prompt_token_ids", "generated_token_ids", "completion_token_ids", "token_input_provenance")
    present = [k in trace for k in keys]
    if not any(present):
        return None
    if not all(present):
        raise ValueError(f"{trace.get('trace_id')}: incomplete recorded-token provenance")
    prompt = _token_list(trace["prompt_token_ids"], "prompt_token_ids")
    generated = _token_list(trace["generated_token_ids"], "generated_token_ids")
    completion = _token_list(trace["completion_token_ids"], "completion_token_ids")
    provenance = trace["token_input_provenance"]
    if not prompt or not isinstance(provenance, dict) or provenance.get("schema") != 1 or provenance.get("capture") != "recorded_at_generation":
        raise ValueError("Unknown or incomplete token provenance")
    expected = completion[:-1] if completion and completion[-1] == tokenizer.eos_token_id else completion
    if generated != expected:
        raise ValueError("Text-span token IDs disagree with the recorded completion")
    if provenance.get("source_sha") != token_source_sha(trace):
        raise ValueError("Recorded tokens belong to different problem/text/formatting")
    if provenance.get("tokenizer_sha") != tokenizer_identity(tokenizer):
        raise ValueError("Recorded tokens were produced with a different tokenizer")
    if provenance.get("prefix_sha") != sha_obj([prompt, generated]):
        raise ValueError("Recorded ordered token IDs changed")
    if tokenizer.decode(generated, skip_special_tokens=True) != trace["text"]:
        raise ValueError("Recorded token IDs do not decode to saved text")
    return prompt[:], generated[:]


def record(p, args, k, text, n_gen, *, tokenizer=None, prompt_ids=None, completion_ids=None):
    answer = extract_boxed(text)
    trace = {
        "trace_id": f"{p['problem_id']}_{k}",
        "problem_id": p["problem_id"],
        "uid": p["unique_id"],
        "level": p["level"],
        "subject": p["subject"],
        "problem": p["problem"],
        "gold": p["answer"],
        "text": text,
        "n_gen": n_gen,
        "trunc": n_gen >= args.max_new or "</think>" not in text,
        "answer": answer,
        "is_correct": answers_match(answer, p["answer"]),
        # extract.py rebuilds this prompt
        "structured": args.structured,
    }
    if tokenizer is not None:
        return attach_token_inputs(trace, tokenizer, prompt_ids, completion_ids)
    return trace


def summarise(path):
    traces = [json.loads(line) for line in open(path)]
    trunc = sum(t["trunc"] for t in traces)
    wrong = sum(t["trunc"] or not t["is_correct"] for t in traces)
    by_problem = {}
    for t in traces:
        by_problem.setdefault(t["problem_id"], []).append(t["trunc"] or not t["is_correct"])
    mixed = sum(0 < sum(v) < len(v) for v in by_problem.values())
    print(f"\n{len(traces)} traces over {len(by_problem)} problems")
    print(f"truncated {trunc} ({trunc / len(traces):.1%}), wrong {wrong} ({wrong / len(traces):.1%})")
    print(f"mixed-outcome problems: {mixed}/{len(by_problem)}")


def main():
    parser = add_args(argparse.ArgumentParser())
    parser.add_argument("--bs", type=int, default=16)
    args = parser.parse_args()

    todo = select(args)
    if not todo:
        return

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    model.eval()
    torch.manual_seed(0)

    for p in todo:
        p["prompt"] = build_prompt(tokenizer, p["problem"], args.structured)
    # sort by length so each batch pads to roughly the same width
    todo.sort(key=lambda p: len(p["prompt"]))

    path = out_path(args.tag)
    out_file = open(path, "a")
    for start in range(0, len(todo), args.bs):
        batch = todo[start : start + args.bs]
        enc = tokenizer(
            [p["prompt"] for p in batch],
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,  # chat template already includes BOS
        ).to(device)
        prompt_len = enc["input_ids"].shape[1]
        with torch.inference_mode():
            out = model.generate(
                **enc,
                do_sample=True,
                temperature=args.temp,
                top_p=0.95,
                max_new_tokens=args.max_new,
                num_return_sequences=args.samples,
                pad_token_id=tokenizer.pad_token_id,
            )
        for i, p in enumerate(batch):
            for k in range(args.samples):
                gen = out[i * args.samples + k, prompt_len:]
                eos = (gen == tokenizer.eos_token_id).nonzero()
                n_gen = eos[0, 0].item() if len(eos) else len(gen)
                text = tokenizer.decode(gen[:n_gen], skip_special_tokens=True)
                prompt_ids = enc["input_ids"][i][enc["attention_mask"][i].bool()].tolist()
                completion_ids = gen[:n_gen + (1 if len(eos) else 0)].tolist()
                out_file.write(json.dumps(record(p, args, k, text, n_gen, tokenizer=tokenizer,
                                                prompt_ids=prompt_ids, completion_ids=completion_ids)) + "\n")
        out_file.flush()
        print(f"{min(start + args.bs, len(todo))}/{len(todo)}", flush=True)
    out_file.close()
    summarise(path)


if __name__ == "__main__":
    main()
