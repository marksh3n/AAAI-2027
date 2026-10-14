# shared helpers for recovery v2. scripts run as `python src/recovery/<x>.py`,
# so src/ goes on the path for generate.py and cache.py.
import hashlib, json, math, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generate import MODEL, build_prompt  # noqa: E402

TOP_P = 0.95  # hardcoded in generate_vllm.py
MAX_NEW = 32768  # tag-B --max-new, the whole trace budget
OUT_DIR = "data/recovery_v2"
# tag-B lines 0-479 are the first 80-problem run, the rest a later run
FIRST_BATCH_LINES = 480


def load_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MODEL)


def prompt_ids(tokenizer, problem, n_bos=None, structured=False):
    # Legacy mode follows this tokenizer's defaults; it does NOT certify historical
    # generation inputs. An explicit policy avoids version-dependent BOS insertion.
    text = build_prompt(tokenizer, problem, structured)
    if n_bos is None:
        return tokenizer(text)["input_ids"]
    if type(n_bos) is not int or n_bos not in (1, 2):
        raise ValueError("n_bos must be 1 or 2")
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    bos = tokenizer.bos_token_id
    if len(ids) < 2 or ids[0] != bos or ids[1] == bos:
        raise ValueError("Expected exactly one template BOS before applying the explicit policy")
    return [bos] * (n_bos - 1) + ids


def load_bos_policy(path, traces, source_path):
    """Read an evidence-labelled trace-ID policy bound to the exact source bytes."""
    if path is None:
        return None
    from cache import sha_file
    with open(path) as handle:
        policy = json.load(handle)
    ids = [t["trace_id"] for t in traces]
    values = policy.get("traces", {})
    if len(set(ids)) != len(ids) or set(values) != set(ids):
        raise ValueError("BOS policy must contain exactly the unique source trace IDs")
    if policy.get("traces_sha256") != sha_file(source_path) or not policy.get("basis"):
        raise ValueError("BOS policy needs the matching source SHA256 and an evidence basis")
    if any(type(v) is not int or v not in (1, 2) for v in values.values()):
        raise ValueError("BOS policy values must be integer 1 or 2")
    return policy


def trace_inputs(tokenizer, trace, policy=None, need_offsets=False):
    """Prefer validated saved inputs; reconstruct legacy text with explicit provenance."""
    from generate import recorded_token_inputs
    saved = recorded_token_inputs(tokenizer, trace)
    count = policy["traces"][trace["trace_id"]] if policy is not None else None
    offsets = None
    if saved is not None:
        prompt, ids = saved
        if count is not None:
            actual = next((i for i, t in enumerate(prompt) if t != tokenizer.bos_token_id), len(prompt))
            if actual != count:
                raise ValueError(f"{trace['trace_id']}: saved prompt conflicts with BOS policy")
        provenance = "recorded_at_generation"
        if need_offsets:
            encoded, offsets = encode_trace(tokenizer, trace["text"])
            if list(encoded) != list(ids):
                raise ValueError(f"{trace['trace_id']}: stored token IDs differ from offset tokenization; cannot snap checkpoints")
    else:
        prompt = prompt_ids(tokenizer, trace["problem"], count, trace.get("structured", False))
        ids, offsets = encode_trace(tokenizer, trace["text"])
        provenance = "reconstructed_explicit_bos_policy" if policy is not None else "reconstructed_legacy_unverified"
    return list(prompt), list(ids), offsets, provenance


def encode_trace(tokenizer, text):
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    return enc["input_ids"], enc["offset_mapping"]


def canonical(trace, ids):
    provenance = trace.get("token_input_provenance")
    if isinstance(provenance, dict) and provenance.get("capture") == "recorded_at_generation":
        # trace_inputs validates provenance before this diagnostic. HF's n_gen
        # excludes terminal EOS; vLLM's includes it. Exact saved IDs remove the
        # need to guess their equality from that historical count convention.
        generated, completion = trace.get("generated_token_ids"), trace.get("completion_token_ids")
        return (isinstance(generated, list) and isinstance(completion, list)
                and list(ids) == generated and trace["n_gen"] in (len(generated), len(completion)))
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
