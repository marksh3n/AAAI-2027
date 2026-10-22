#!/usr/bin/env python3
"""Small, reference Hugging Face generation / exact-token extraction pipeline.

No probe training, answer grading, or continuation-rollout experiment is included.
Importing this module does not import torch, download models, or execute a run.
Generation and extraction require an explicit frozen plan and local candidates.
This single-trace HF reference implementation can be slow; no throughput claim.

Candidates JSONL: {"problem_id":"...", "problem":"...", "gold":"..."}.
Plan schema (all fields required):
  schema_version: 1; phase: "pilot" or "confirmation";
  model_id: "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B";
  model_revision / tokenizer_revision: immutable 40-character commit hashes;
  selected_problem_ids: nonempty, distinct string IDs;
  generation: {samples_per_problem, temperature, top_p, max_new_tokens, base_seed};
  extraction: {checkpoints: [512, 2048, 4096, 6144, 8192]} (a subset is allowed).

The common prompt is rendered with the pinned chat template, then tokenized with
add_special_tokens=False. Exactly one leading BOS is required; none is inserted
or removed. Actual generation IDs, rather than retokenized text, feed extraction.
Recorded hashes detect subsequent changes, not intentional provenance forgery.
"""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import tempfile

MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
CHECKPOINTS = (512, 2048, 4096, 6144, 8192)
INSTRUCTION = "\nPlease reason step by step, and put your final answer within \\boxed{}."
BOS_RULE = "chat_template_then_tokenize_add_special_tokens_false_exactly_one_leading_BOS"
VERSION = 1


def sha_obj(value):
    # Deliberately matches the project's cache.sha_obj for prefix comparisons.
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def token_ids(value, name):
    require(isinstance(value, list) and all(type(x) is int and x >= 0 for x in value),
            f"{name}: expected ordered nonnegative integer token IDs")
    return value


def validate_plan(plan):
    required = {"schema_version", "phase", "model_id", "model_revision", "tokenizer_revision",
                "selected_problem_ids", "generation", "extraction"}
    require(isinstance(plan, dict) and set(plan) == required, "Plan fields differ from documented schema")
    require(plan["schema_version"] == VERSION, "Unsupported plan schema")
    require(plan["phase"] in ("pilot", "confirmation"), "Declare pilot or confirmation phase")
    require(plan["model_id"] == MODEL, "This reference implementation supports only the declared 1.5B model")
    for key in ("model_revision", "tokenizer_revision"):
        require(isinstance(plan[key], str) and re.fullmatch(r"[0-9a-f]{40}", plan[key]),
                f"{key}: an immutable lowercase 40-hex commit is required, not main")
    ids = plan["selected_problem_ids"]
    require(isinstance(ids, list) and ids and all(isinstance(x, str) and x for x in ids)
            and len(ids) == len(set(ids)), "Select distinct nonempty string problem IDs explicitly")
    g = plan["generation"]
    require(isinstance(g, dict) and set(g) == {"samples_per_problem", "temperature", "top_p", "max_new_tokens", "base_seed"},
            "Generation fields differ from documented schema")
    for key in ("samples_per_problem", "max_new_tokens"):
        require(type(g[key]) is int and g[key] > 0, f"{key} must be a positive integer")
    require(type(g["base_seed"]) is int and 0 <= g["base_seed"] < 2**32, "Invalid base_seed")
    for key in ("temperature", "top_p"):
        require(type(g[key]) in (int, float) and math.isfinite(g[key]) and g[key] > 0,
                f"Invalid {key}")
    require(g["top_p"] <= 1, "top_p must be <= 1")
    e = plan["extraction"]
    require(isinstance(e, dict) and set(e) == {"checkpoints"}, "Unknown extraction setting")
    checkpoints = e["checkpoints"]
    require(isinstance(checkpoints, list) and checkpoints
            and all(type(x) is int and x in CHECKPOINTS for x in checkpoints)
            and checkpoints == sorted(set(checkpoints)), "Use sorted distinct supported checkpoints")
    return plan


def load_inputs(plan_path, problems_path):
    plan = validate_plan(json.loads(Path(plan_path).read_text()))
    candidates = {}
    with open(problems_path) as handle:
        for line_no, line in enumerate(handle, 1):
            require(bool(line.strip()), f"Blank candidate line {line_no}")
            row = json.loads(line)
            require(isinstance(row, dict) and {"problem_id", "problem", "gold"} <= set(row),
                    f"Candidate line {line_no}: missing problem_id/problem/gold")
            require(all(isinstance(row[k], str) and row[k].strip() for k in ("problem_id", "problem", "gold")),
                    f"Candidate line {line_no}: IDs, questions and gold answers must be nonempty strings")
            require(row["problem_id"] not in candidates, "Duplicate candidate problem_id")
            candidates[row["problem_id"]] = row
    require(set(plan["selected_problem_ids"]) <= set(candidates), "A selected problem is missing")
    selected = [candidates[x] for x in plan["selected_problem_ids"]]
    manifest = {"schema_version": VERSION, "plan": plan, "plan_sha256": sha_obj(plan),
                "candidates_file_sha256": sha_file(problems_path),
                "selected_problems_sha256": sha_obj(selected), "bos_rule": BOS_RULE,
                "implementation_sha256": sha_file(__file__)}
    manifest["settings_hash"] = sha_obj(manifest)
    return plan, selected, manifest


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".pending-", suffix=".json", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def bind_output(output_dir, manifest, create=False):
    directory = Path(output_dir)
    path = directory / "run_manifest.json"
    if path.exists():
        require(json.loads(path.read_text()) == manifest,
                "Resume refused: plan, source/gold, implementation, or settings changed. Use a new output directory.")
    else:
        require(create, "No run manifest: extraction accepts this helper's recorded outputs only")
        require(not directory.exists() or not list(directory.iterdir()), "Nonempty output directory has no run manifest")
        atomic_json(path, manifest)
    return directory


def tokenizer_identity(tokenizer):
    backend = getattr(tokenizer, "backend_tokenizer", None)
    config = json.loads(backend.to_str()) if backend is not None else None
    if config is not None:
        config.pop("padding", None)
        config.pop("truncation", None)
    return sha_obj({"vocab": tokenizer.get_vocab(), "backend": config,
                    "special_tokens": {k: str(v) for k, v in tokenizer.special_tokens_map.items() if k != "pad_token"},
                    "chat_template": tokenizer.chat_template})


def exact_prompt(tokenizer, question):
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": question + INSTRUCTION}], tokenize=False, add_generation_prompt=True)
    if not rendered.rstrip().endswith("<think>"):
        rendered += "<think>\n"
    ids = token_ids(tokenizer(rendered, add_special_tokens=False)["input_ids"], "prompt")
    bos = tokenizer.bos_token_id
    require(type(bos) is int and len(ids) >= 2 and ids[0] == bos and ids[1] != bos,
            "Prompt must contain exactly one leading BOS from its chat template; do not silently insert/remove BOS")
    return rendered, ids


def sample_key(problem_id, sample_index):
    return hashlib.sha256(json.dumps([problem_id, sample_index]).encode()).hexdigest()


def sample_seed(plan, problem_id, sample_index):
    return int(sha_obj([plan["generation"]["base_seed"], problem_id, sample_index])[:8], 16)


def runtime_identity(plan, tokenizer, model):
    require(getattr(model.config, "_commit_hash", None) == plan["model_revision"],
            "Loaded model config does not report the requested immutable model revision")
    return {"model_id": plan["model_id"], "model_revision": plan["model_revision"],
            "tokenizer_revision": plan["tokenizer_revision"], "tokenizer_sha256": tokenizer_identity(tokenizer),
            "torch_version": importlib.metadata.version("torch"),
            "transformers_version": importlib.metadata.version("transformers"),
            "compute_dtype": "float16", "attention_implementation": "sdpa",
            "bos_token_id": tokenizer.bos_token_id, "eos_token_id": tokenizer.eos_token_id,
            "num_hidden_layers": model.config.num_hidden_layers, "hidden_size": model.config.hidden_size,
            "max_position_embeddings": model.config.max_position_embeddings}


def load_model(plan):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    require(torch.cuda.is_available(), "A CUDA runtime is required for this reference run; no CPU generation fallback")
    tokenizer = AutoTokenizer.from_pretrained(plan["model_id"], revision=plan["tokenizer_revision"],
                                              trust_remote_code=False, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(plan["model_id"], revision=plan["model_revision"],
                                                trust_remote_code=False, torch_dtype=torch.float16,
                                                attn_implementation="sdpa").to("cuda").eval()
    runtime = runtime_identity(plan, tokenizer, model)
    require(type(tokenizer.eos_token_id) is int, "A single EOS token ID is required")
    require(runtime["num_hidden_layers"] == 28 and runtime["hidden_size"] == 1536,
            "Unexpected model architecture; do not reuse this extraction layout")
    return tokenizer, model, runtime


def bind_runtime(directory, runtime, create=False):
    path = Path(directory) / "runtime.json"
    if path.exists():
        require(json.loads(path.read_text()) == runtime, "Resume refused: tokenizer/model/runtime settings changed")
    else:
        require(create, "Recorded run has no runtime identity")
        atomic_json(path, runtime)


def make_record(problem, sample_index, plan, manifest, runtime, prompt_text, prompt_ids, completion_ids, tokenizer):
    prompt_ids = token_ids(list(prompt_ids), "prompt_token_ids")
    completion_ids = token_ids(list(completion_ids), "completion_token_ids")
    eos = runtime["eos_token_id"]
    eos_at = [i for i, value in enumerate(completion_ids) if value == eos]
    require(not eos_at or eos_at == [len(completion_ids) - 1], "Unexpected interior EOS/padding in single-trace generation")
    generated_ids = completion_ids[:-1] if eos_at else completion_ids[:]
    row = {"schema_version": VERSION, "capture": "recorded_at_generation", "problem": problem,
           "sample_index": sample_index, "trace_id": sample_key(problem["problem_id"], sample_index),
           "seed": sample_seed(plan, problem["problem_id"], sample_index),
           "settings_hash": manifest["settings_hash"], "runtime": runtime, "bos_rule": BOS_RULE,
           "prompt_text": prompt_text, "prompt_token_ids": prompt_ids,
           "completion_token_ids": completion_ids, "generated_token_ids": generated_ids,
           "text": tokenizer.decode(generated_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False),
           "eos_completion_index": eos_at[0] if eos_at else None,
           "finish_reason": "eos" if eos_at else "max_new_tokens",
           "generation": plan["generation"], "prefix_sha": sha_obj([prompt_ids, generated_ids])}
    row["record_sha256"] = sha_obj(row)
    validate_record(row, problem, sample_index, plan, manifest, runtime, tokenizer)
    return row


def validate_record(row, problem, sample_index, plan, manifest, runtime, tokenizer):
    require(isinstance(row, dict), "Invalid trace record")
    unsigned = {k: v for k, v in row.items() if k != "record_sha256"}
    require(row.get("record_sha256") == sha_obj(unsigned), "Trace contents/hash changed")
    require(row.get("schema_version") == VERSION and row.get("capture") == "recorded_at_generation",
            "Legacy or reconstructed traces are not accepted")
    require(row.get("problem") == problem and row.get("sample_index") == sample_index
            and row.get("trace_id") == sample_key(problem["problem_id"], sample_index), "Trace/problem/gold identity mismatch")
    require(row.get("settings_hash") == manifest["settings_hash"] and row.get("runtime") == runtime,
            "Trace configuration/revision/runtime mismatch")
    require(row.get("generation") == plan["generation"] and row.get("bos_rule") == BOS_RULE
            and row.get("seed") == sample_seed(plan, problem["problem_id"], sample_index), "Trace generation settings mismatch")
    prompt = token_ids(row.get("prompt_token_ids"), "prompt_token_ids")
    completion = token_ids(row.get("completion_token_ids"), "completion_token_ids")
    generated = token_ids(row.get("generated_token_ids"), "generated_token_ids")
    expected_text, expected_prompt = exact_prompt(tokenizer, problem["problem"])
    require(prompt == expected_prompt and row.get("prompt_text") == expected_text, "Recorded prompt differs from the frozen template/question")
    eos = runtime["eos_token_id"]
    eos_at = [i for i, token in enumerate(completion) if token == eos]
    require(not eos_at or eos_at == [len(completion) - 1], "Interior EOS is invalid")
    require(generated == (completion[:-1] if eos_at else completion), "Completion/text-span IDs disagree")
    require(0 < len(completion) <= plan["generation"]["max_new_tokens"], "Unexpected completion length")
    require(bool(eos_at) or len(completion) == plan["generation"]["max_new_tokens"], "Non-EOS completion ended before the configured cap")
    require(row.get("eos_completion_index") == (eos_at[0] if eos_at else None)
            and row.get("finish_reason") == ("eos" if eos_at else "max_new_tokens"), "Incorrect EOS boundary/finish reason")
    require(row.get("prefix_sha") == sha_obj([prompt, generated]), "Ordered token hash mismatch")
    require(row.get("text") == tokenizer.decode(generated, skip_special_tokens=False, clean_up_tokenization_spaces=False),
            "Recorded IDs do not decode to the saved text")
    return row


def expected_records(problems, plan):
    return [(p, i, sample_key(p["problem_id"], i)) for p in problems
            for i in range(plan["generation"]["samples_per_problem"])]


def check_trace_directory(directory, expected):
    folder = Path(directory) / "traces"
    if folder.exists():
        unexpected = {x.stem for x in folder.glob("*.json")} - {key for _, _, key in expected}
        require(not unexpected, "Output contains traces outside this plan")
    return folder


def generation_config(plan, tokenizer):
    from transformers import GenerationConfig

    g = plan["generation"]
    # Build from scratch: do not inherit the model repo's mutable sampling defaults.
    return GenerationConfig(do_sample=True, temperature=g["temperature"], top_p=g["top_p"], top_k=0,
                            max_new_tokens=g["max_new_tokens"], num_beams=1, num_return_sequences=1,
                            repetition_penalty=1.0, no_repeat_ngram_size=0, typical_p=1.0,
                            min_p=None, epsilon_cutoff=0.0, eta_cutoff=0.0,
                            bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id,
                            pad_token_id=tokenizer.eos_token_id, use_cache=True,
                            return_dict_in_generate=True, output_scores=False)


def run_generation(plan, problems, manifest, directory, confirm_confirmation_run=False):
    require(plan["phase"] != "confirmation" or confirm_confirmation_run,
            "Confirmation generation requires --confirm-confirmation-run after freezing the plan")
    import torch
    from transformers import set_seed

    directory = bind_output(directory, manifest, create=True)
    expected = expected_records(problems, plan)
    folder = check_trace_directory(directory, expected)
    tokenizer, model, runtime = load_model(plan)
    bind_runtime(directory, runtime, create=True)
    config = generation_config(plan, tokenizer)
    config_path = directory / "generation_config.json"
    effective = config.to_dict()
    if config_path.exists():
        require(json.loads(config_path.read_text()) == effective, "Effective generation configuration changed")
    else:
        atomic_json(config_path, effective)
    # Preflight all existing records and context budgets before generating anything.
    pending = []
    for problem, index, key in expected:
        prompt_text, prompt = exact_prompt(tokenizer, problem["problem"])
        require(len(prompt) + plan["generation"]["max_new_tokens"] <= runtime["max_position_embeddings"],
                f"{problem['problem_id']}: prompt + max_new_tokens exceeds the model's declared context limit")
        path = folder / f"{key}.json"
        if path.exists():
            validate_record(json.loads(path.read_text()), problem, index, plan, manifest, runtime, tokenizer)
        else:
            pending.append((problem, index, path, prompt_text, prompt))
    for problem, index, path, prompt_text, prompt in pending:
        seed = sample_seed(plan, problem["problem_id"], index)
        set_seed(seed)
        inputs = torch.tensor([prompt], dtype=torch.long, device="cuda")
        with torch.inference_mode():
            output = model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs), generation_config=config)
        sequence = output.sequences[0].tolist()
        require(sequence[:len(prompt)] == prompt, "Generation returned a different prompt prefix")
        record = make_record(problem, index, plan, manifest, runtime, prompt_text, prompt,
                             sequence[len(prompt):], tokenizer)
        atomic_json(path, record)
        print(json.dumps({"saved": str(path), "problem_id": problem["problem_id"], "sample_index": index,
                          "generated_tokens": len(record["generated_token_ids"]), "finish_reason": record["finish_reason"]}), flush=True)
        del inputs, output


def answer_boundary(tokenizer, record):
    """Conservative marker annotation, not proof that a prefix contains no answer.

    Text offsets may only label actual IDs when round-trip tokenization agrees.
    Missing markers or changed token segmentation are explicitly unknown.
    """
    text = record["text"]
    markers = [(text.find(s), s) for s in ("</think>", "\\boxed{") if s in text]
    if not markers:
        return {"status": "unknown", "reason": "no_reliable_answer_marker"}
    try:
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    except (NotImplementedError, TypeError, ValueError):
        return {"status": "unknown", "reason": "token_offsets_unavailable"}
    if encoded.get("input_ids") != record["generated_token_ids"]:
        return {"status": "unknown", "reason": "text_offsets_do_not_match_recorded_ids"}
    offsets = encoded.get("offset_mapping")
    if offsets is None or len(offsets) != len(record["generated_token_ids"]):
        return {"status": "unknown", "reason": "token_offsets_unavailable"}
    if any(not isinstance(pair, (tuple, list)) or len(pair) != 2
           or not all(type(x) is int for x in pair)
           or not 0 <= pair[0] < pair[1] <= len(text) for pair in offsets):
        return {"status": "unknown", "reason": "invalid_or_zero_width_token_offsets"}
    if any(offsets[i][0] < offsets[i - 1][0] or offsets[i][1] < offsets[i - 1][1]
           for i in range(1, len(offsets))):
        return {"status": "unknown", "reason": "nonmonotone_token_offsets"}
    start, marker = min(markers)
    # A token crossing into a marker is already not a pre-marker checkpoint.
    first = next((i for i, (a, b) in enumerate(offsets) if b > start), None)
    if first is None:
        return {"status": "unknown", "reason": "marker_not_mapped_to_tokens"}
    return {"status": "marker_located", "marker": marker, "marker_char_start": start,
            "first_marker_token_index": first,
            "limitation": "marker-based annotation; unmarked answers can occur earlier"}


def checkpoint_metadata(record, checkpoints, boundary):
    prompt, generated = record["prompt_token_ids"], record["generated_token_ids"]
    rows = []
    for position in checkpoints:
        if position > len(generated):
            continue
        stage = "unknown"
        if boundary["status"] == "marker_located":
            stage = "before_answer_marker" if position <= boundary["first_marker_token_index"] else "at_or_after_answer_marker"
        prefix = generated[:position]
        rows.append({"checkpoint": position, "generation_token_index": position - 1,
                     "model_input_token_index": len(prompt) + position - 1,
                     "prompt_token_ids": prompt, "prefix_gen_token_ids": prefix,
                     "prefix_sha": sha_obj([prompt, prefix]), "answer_stage": stage})
    return rows


def extraction_metadata(record, checkpoints, runtime, boundary):
    return {"schema_version": VERSION, "trace_id": record["trace_id"], "problem_id": record["problem"]["problem_id"],
            "source_record_sha256": record["record_sha256"], "settings_hash": record["settings_hash"],
            "runtime": runtime, "layer_order": list(range(-1, 28)), "layer_convention": "hidden_states[layer+1]",
            "stored_dtype": "float16", "forward": "model.model; causal; use_cache=False; no future completion tokens",
            "answer_boundary": boundary, "checkpoints": checkpoint_metadata(record, checkpoints, boundary),
            "missing_checkpoints": [p for p in checkpoints if p > len(record["generated_token_ids"])],
            "grading": "not_performed"}


def validate_extraction(npz_path, metadata_path, expected):
    import numpy as np

    require(Path(npz_path).exists() and Path(metadata_path).exists(),
            "Incomplete extraction pair. Move the partial output aside before retrying; it is not reusable.")
    meta = json.loads(Path(metadata_path).read_text())
    require(meta.get("metadata") == expected, "Extraction source/config/token inputs changed")
    require(meta.get("npz_sha256") == sha_file(npz_path), "Extracted vectors file changed")
    with np.load(npz_path, allow_pickle=False) as data:
        require(set(data.files) == {"vectors", "metadata_json"}, "Unexpected NPZ layout")
        vectors = data["vectors"]
        require(vectors.shape == (len(expected["checkpoints"]), 29, 1536) and vectors.dtype == np.float16,
                "Unexpected vector shape/dtype")
        require(bool(np.isfinite(vectors).all()), "Nonfinite hidden vectors")
        require(json.loads(str(data["metadata_json"].item())) == expected, "Embedded metadata differs")


def write_extraction(npz_path, metadata_path, vectors, metadata):
    import numpy as np

    npz_path = Path(npz_path)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".pending-", suffix=".npz", dir=npz_path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            np.savez(handle, vectors=vectors, metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, npz_path)
        atomic_json(metadata_path, {"metadata": metadata, "npz_sha256": sha_file(npz_path)})
    finally:
        if os.path.exists(name):
            os.unlink(name)
    validate_extraction(npz_path, metadata_path, metadata)


def run_extraction(plan, problems, manifest, directory):
    import numpy as np
    import torch

    directory = bind_output(directory, manifest)
    expected = expected_records(problems, plan)
    folder = check_trace_directory(directory, expected)
    require(all((folder / f"{key}.json").exists() for _, _, key in expected),
            "Generation is incomplete: finish the planned traces before extraction")
    tokenizer, model, runtime = load_model(plan)
    bind_runtime(directory, runtime)
    records = []
    for problem, index, key in expected:
        row = validate_record(json.loads((folder / f"{key}.json").read_text()), problem, index,
                              plan, manifest, runtime, tokenizer)
        records.append(row)
    for record in records:
        boundary = answer_boundary(tokenizer, record)
        metadata = extraction_metadata(record, plan["extraction"]["checkpoints"], runtime, boundary)
        stem = directory / "extractions" / record["trace_id"]
        npz_path, meta_path = stem.with_suffix(".npz"), stem.with_suffix(".json")
        if npz_path.exists() or meta_path.exists():
            validate_extraction(npz_path, meta_path, metadata)
            continue
        vectors = []
        # Each forward sees ONLY prompt + this recorded prefix. No logits and no
        # persistent KV cache. Maximum generated prefix is 8192, plus the prompt.
        for checkpoint in metadata["checkpoints"]:
            ids = checkpoint["prompt_token_ids"] + checkpoint["prefix_gen_token_ids"]
            require(len(ids) <= runtime["max_position_embeddings"], "Extraction input exceeds model context")
            inputs = torch.tensor([ids], dtype=torch.long, device="cuda")
            with torch.inference_mode():
                output = model.model(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                                     use_cache=False, output_hidden_states=True, return_dict=True)
            require(len(output.hidden_states) == 29, "Unexpected hidden-state layer layout")
            vector = torch.stack([layer[0, -1, :] for layer in output.hidden_states]).to(torch.float16).cpu().numpy()
            require(vector.shape == (29, 1536) and bool(np.isfinite(vector).all()), "Bad activation vector")
            vectors.append(vector)
            del inputs, output
        array = np.stack(vectors) if vectors else np.empty((0, 29, 1536), dtype=np.float16)
        write_extraction(npz_path, meta_path, array, metadata)
        print(json.dumps({"saved": str(npz_path), "checkpoints": len(vectors),
                          "answer_boundary": boundary["status"]}), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "generate", "extract"):
        child = sub.add_parser(name)
        child.add_argument("--plan", required=True)
        child.add_argument("--problems", required=True)
        if name != "validate":
            child.add_argument("--output-dir", required=True)
        if name == "generate":
            child.add_argument("--confirm-confirmation-run", action="store_true")
    args = parser.parse_args(argv)
    plan, problems, manifest = load_inputs(args.plan, args.problems)
    if args.command == "validate":
        print(json.dumps({"valid": True, "phase": plan["phase"], "problems": len(problems),
                          "planned_traces": len(expected_records(problems, plan)), "settings_hash": manifest["settings_hash"],
                          "warning": "CPU schema check only; no model/revision download or experiment executed"}, indent=2))
    elif args.command == "generate":
        run_generation(plan, problems, manifest, args.output_dir, args.confirm_confirmation_run)
    else:
        run_extraction(plan, problems, manifest, args.output_dir)


if __name__ == "__main__":
    main()
