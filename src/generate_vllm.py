# colab: pip install -q vllm math-verify
import argparse, json
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from generate import MODEL, add_args, build_prompt, out_path, record, select, summarise

CHUNK = 8


def main():
    args = add_args(argparse.ArgumentParser()).parse_args()

    todo = select(args)
    if not todo:
        return

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    llm = LLM(
        model=MODEL,
        dtype="float16",
        max_model_len=args.max_new + 1024,
        gpu_memory_utilization=0.90,
    )
    # n>1 shares the prompt kv cache across samples
    params = SamplingParams(
        temperature=args.temp, top_p=0.95, max_tokens=args.max_new, n=args.samples, seed=0
    )

    path = out_path(args.tag)
    out_file = open(path, "a")
    for start in range(0, len(todo), CHUNK):
        batch = todo[start : start + CHUNK]
        prompts = [build_prompt(tokenizer, p["problem"], args.structured) for p in batch]
        for p, out in zip(batch, llm.generate(prompts, params)):
            if out.prompt_token_ids is None:
                raise ValueError("vLLM did not expose the actual prompt IDs; cannot record trace provenance")
            for k, o in enumerate(out.outputs):
                out_file.write(json.dumps(record(p, args, k, o.text, len(o.token_ids), tokenizer=tokenizer,
                                                prompt_ids=list(out.prompt_token_ids),
                                                completion_ids=list(o.token_ids))) + "\n")
        out_file.flush()
        print(f"{min(start + CHUNK, len(todo))}/{len(todo)}", flush=True)
    out_file.close()
    summarise(path)


if __name__ == "__main__":
    main()
