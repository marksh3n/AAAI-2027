# colab: pip install -q math-verify
import argparse, json, os
from collections import Counter
import numpy as np
import torch
from datasets import load_dataset
from math_verify import parse, verify
from transformers import AutoModelForCausalLM, AutoTokenizer

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


def record(p, args, k, text, n_gen):
    answer = extract_boxed(text)
    return {
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
                out_file.write(json.dumps(record(p, args, k, text, n_gen)) + "\n")
        out_file.flush()
        print(f"{min(start + args.bs, len(todo))}/{len(todo)}", flush=True)
    out_file.close()
    summarise(path)



if __name__ == "__main__":
    main()
