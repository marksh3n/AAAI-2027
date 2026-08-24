# colab: pip install -q vllm math-verify
import argparse, json
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from generate import (
    MODEL,
    OUT_PATH,
    already_done,
    answers_match,
    build_prompt,
    extract_boxed,
    load_problems,
)

CHUNK = 32


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-new", type=int, default=32768)
    args = parser.parse_args()

    problems = load_problems(0)
    done = already_done()
    todo = [p for p in problems if p["problem_id"] not in done]
    print(f"{len(todo)}/{len(problems)} left")
    if not todo:
        return

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    llm = LLM(
        model=MODEL,
        dtype="float16",
        max_model_len=args.max_new + 1024,
        gpu_memory_utilization=0.90,
    )
    params = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=args.max_new, seed=0)

    out_file = open(OUT_PATH, "a")
    for start in range(0, len(todo), CHUNK):
        batch = todo[start : start + CHUNK]
        outs = llm.generate([build_prompt(tokenizer, p["problem"]) for p in batch], params)
        for p, out in zip(batch, outs):
            text = out.outputs[0].text
            n_gen = len(out.outputs[0].token_ids)
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
                        "trunc": n_gen >= args.max_new or "</think>" not in text,
                        "answer": answer,
                        "is_correct": answers_match(answer, p["answer"]),
                    }
                )
                + "\n"
            )
        out_file.flush()
        print(f"{min(start + CHUNK, len(todo))}/{len(todo)}", flush=True)
    out_file.close()


if __name__ == "__main__":
    main()
