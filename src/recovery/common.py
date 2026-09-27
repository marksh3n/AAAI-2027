# shared helpers for recovery v2. scripts run as `python src/recovery/<x>.py`,
# so src/ goes on the path for generate.py and cache.py.
import hashlib, json, math, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generate import MODEL, answers_match, build_prompt, extract_boxed, out_path  # noqa: E402,F401

TOP_P = 0.95  # hardcoded in generate_vllm.py
MAX_NEW = 32768  # tag-B --max-new, the whole trace budget
OUT_DIR = "data/recovery_v2"
# tag-B lines 0-479 are the first 80-problem run, the rest a later run
FIRST_BATCH_LINES = 480


def load_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MODEL)


def prompt_ids(tokenizer, problem):
    # vLLM tokenizes a string prompt with special tokens on, and the chat template
    # already emits BOS, so generation saw two. run_rollouts.py re-checks against vLLM.
    return tokenizer(build_prompt(tokenizer, problem))["input_ids"]


def encode_trace(tokenizer, text):
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    return enc["input_ids"], enc["offset_mapping"]


def canonical(trace, ids):
    # token ids were never stored. vLLM's count includes EOS on a stop, not on a
    # length cap; any other count means retokenizing the text drifted from the sample.
    return trace["n_gen"] == len(ids) + (0 if trace["n_gen"] >= MAX_NEW else 1)


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def prefix_seed(base, prefix_id, round_):
    # fresh stream per (prefix, round). python hash() is salted, so sha256.
    digest = hashlib.sha256(f"{base}:{prefix_id}:{round_}".encode()).hexdigest()
    return int(digest[:8], 16)


def read_jsonl(path):
    # a file copied while its writer was mid-line ends in a torn line; only that one is skipped
    if not os.path.exists(path):
        return []
    with open(path) as f:
        lines = [line for line in f if line.strip()]
    out = [json.loads(line) for line in lines[:-1]]
    if lines:
        try:
            out.append(json.loads(lines[-1]))
        except json.JSONDecodeError:
            if lines[-1].endswith("\n"):
                raise
    return out
