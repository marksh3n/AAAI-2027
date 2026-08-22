# colab: pip install -q math-verify
import argparse, json, os
import torch
from datasets import load_dataset
from math_verify import parse, verify
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
OUT_PATH = "data/traces.jsonl"
MAX_NEW = 4096
INSTRUCTION = "\nPlease reason step by step, and put your final answer within \\boxed{}."


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


def build_prompt(tokenizer, question):
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": question + INSTRUCTION}],
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


def already_done():
    if not os.path.exists(OUT_PATH):
        return set()
    with open(OUT_PATH) as f:
        return {json.loads(line)["problem_id"] for line in f}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=0)
    parser.add_argument("--bs", type=int, default=16)
    args = parser.parse_args()

    problems = load_problems(args.n)
    done = already_done()
    todo = [p for p in problems if p["problem_id"] not in done]
    print(f"{len(todo)}/{len(problems)} left")
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
        p["prompt"] = build_prompt(tokenizer, p["problem"])
    # sort by length so each batch pads to roughly the same width
    todo.sort(key=lambda p: len(p["prompt"]))

    out_file = open(OUT_PATH, "a")
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
                temperature=0.6,
                top_p=0.95,
                max_new_tokens=MAX_NEW,
                pad_token_id=tokenizer.pad_token_id,
            )
        for i, p in enumerate(batch):
            gen = out[i, prompt_len:]
            eos = (gen == tokenizer.eos_token_id).nonzero()
            n_gen = eos[0, 0].item() if len(eos) else len(gen)
            text = tokenizer.decode(gen[:n_gen], skip_special_tokens=True)
            answer = extract_boxed(text)
            out_file.write(
                json.dumps(
                    {
                        "trace_id": f"{p['problem_id']}_0",
                        "problem_id": p["problem_id"],
                        "uid": p["unique_id"],
                        "level": p["level"],
                        "subject": p["subject"],
                        "problem": p["problem"],
                        "gold": p["answer"],
                        "text": text,
                        "n_gen": n_gen,
                        "trunc": n_gen >= MAX_NEW or "</think>" not in text,
                        "answer": answer,
                        "is_correct": answers_match(answer, p["answer"]),
                    }
                )
                + "\n"
            )
        out_file.flush()
        print(f"{min(start + args.bs, len(todo))}/{len(todo)}", flush=True)
    out_file.close()


main()
