"""CPU-only audit of saved rollout input hashes against one/two-BOS candidates.

Pass a clean JSONL rollout file. Every record is validated; malformed or duplicate
records fail the audit instead of being skipped. Matching a saved prefix hash
establishes its token candidate under the repository's hash definition, not the
historical generation inputs or the inputs that produced an activation array.
"""
import argparse
import ast
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re

from cache import sha_obj, sha_file


def strict_loads(raw):
    def object_hook(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                raise ValueError(f"duplicate JSON key {key}")
            out[key] = value
        return out

    def reject_constant(value):
        raise ValueError(f"non-finite JSON constant {value}")

    return json.loads(raw, object_pairs_hook=object_hook, parse_constant=reject_constant)


def input_hash(prompt_ids, generated_prefix_ids):
    return sha_obj([list(prompt_ids), list(generated_prefix_ids)])


def register_key(seen, key):
    if key in seen:
        raise ValueError(f"duplicate prefix/round {key}")
    seen.add(key)


def check_rollout_manifest(path, manifest_path):
    if manifest_path is None:
        return
    expected = strict_loads(Path(manifest_path).read_text())["cleaned_rollouts_sha256"]
    actual = sha_file(path)
    if actual != expected:
        raise ValueError(f"cleaned rollout file hash does not match cleaning manifest: {actual} != {expected}")


def bos_candidates(prompt_ids, bos_id):
    """Candidates are explicit, not an automatic token repair."""
    ids = list(prompt_ids)
    n = 0
    while n < len(ids) and ids[n] == bos_id:
        n += 1
    if n != 1:
        raise ValueError(f"chat template must emit exactly one leading BOS; got {n}")
    return {1: ids, 2: [bos_id] + ids}


def matching_bos(prompt_ids, generated_prefix_ids, bos_id, stored_hash):
    candidates = bos_candidates(prompt_ids, bos_id)
    matches = [n for n, p in candidates.items() if input_hash(p, generated_prefix_ids) == stored_hash]
    return matches[0] if len(matches) == 1 else None


def checkpoint_cut(row):
    checkpoint = row["checkpoint"]
    if checkpoint == "t0":
        expected = 0
        if row["trace_id"] is not None:
            raise ValueError("t0 must not reference a source trace")
    else:
        match = re.fullmatch(r"p([1-9][0-9]*)", str(checkpoint))
        if not match:
            raise ValueError(f"unsupported checkpoint {checkpoint!r}")
        expected = int(match.group(1))
    if type(row["cut_token_idx"]) is not int or row["cut_token_idx"] != expected:
        raise ValueError(f"checkpoint {checkpoint} disagrees with cut_token_idx={row['cut_token_idx']}")
    return expected


def load_build_prompt(path):
    """Load only literal prompt constants and build_prompt, not model imports."""
    tree = ast.parse(Path(path).read_text())
    definitions = []
    names = {"INSTRUCTION", "STRUCTURED"}
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in node.targets):
            value = ast.literal_eval(node.value)
            if not isinstance(value, str):
                raise ValueError("prompt constants must be literal strings")
            definitions.append(node)
        if isinstance(node, ast.FunctionDef) and node.name == "build_prompt":
            definitions.append(node)
    scope = {}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), "exec"), scope)
    if not names.issubset(scope) or "build_prompt" not in scope:
        raise ValueError("generate.py lacks expected prompt function/constants")
    return scope["build_prompt"]


class LocalTokenizer:
    """Pinned tokenizers backend plus the supplied Jinja chat template."""
    def __init__(self, directory):
        from jinja2 import Environment, StrictUndefined
        from tokenizers import Tokenizer
        directory = Path(directory)
        self.config = strict_loads((directory / "tokenizer_config.json").read_text())
        self.backend = Tokenizer.from_file(str(directory / "tokenizer.json"))
        self.bos_token = self.config["bos_token"]["content"]
        self.eos_token = self.config["eos_token"]["content"]
        self.bos_token_id = self.backend.token_to_id(self.bos_token)
        self.template = Environment(undefined=StrictUndefined).from_string(self.config["chat_template"])

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        if tokenize:
            raise ValueError("audit only renders the template; tokenization is explicit")
        return self.template.render(messages=messages, bos_token=self.bos_token,
                                    eos_token=self.eos_token, add_generation_prompt=add_generation_prompt)

    def encode(self, text):
        return self.backend.encode(text, add_special_tokens=False).ids


def write_csv(path, rows, fields):
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(args):
    args.out.mkdir(parents=True, exist_ok=True)
    check_rollout_manifest(args.rollouts, args.rollout_manifest)
    manifest = strict_loads((args.tokenizer_dir / "tokenizer_manifest.json").read_text())
    for file in manifest["files"]:
        if sha_file(args.tokenizer_dir / file["name"]) != file["sha256"]:
            raise ValueError(f"tokenizer file hash mismatch: {file['name']}")
    tokenizer = LocalTokenizer(args.tokenizer_dir)
    build_prompt = load_build_prompt(args.generation_code)
    sources, problems, issues = {}, {}, []
    for line, text in enumerate(args.traces.open(), 1):
        t = strict_loads(text)
        if t["trace_id"] in sources:
            raise ValueError(f"duplicate source trace_id: {t['trace_id']}")
        if t["problem_id"] in problems and problems[t["problem_id"]]["problem"] != t["problem"]:
            raise ValueError(f"conflicting problem text: {t['problem_id']}")
        sources[t["trace_id"]] = (line, t)
        problems[t["problem_id"]] = t
    prompts, generated, seen, results = {}, {}, set(), []
    for physical_line, text in enumerate(args.rollouts.open(), 1):
        try:
            r = strict_loads(text)
            key = (r["prefix_id"], r["round"])
            register_key(seen, key)
            cut = checkpoint_cut(r)
            pid, tid = r["problem_id"], r["trace_id"]
            if pid not in problems:
                raise ValueError(f"unknown problem_id {pid}")
            if tid is not None:
                if tid not in sources or sources[tid][1]["problem_id"] != pid:
                    raise ValueError(f"trace/problem mismatch: {tid}/{pid}")
                source_line, trace = sources[tid]
                if tid not in generated:
                    generated[tid] = tokenizer.encode(trace["text"])
                if cut > len(generated[tid]):
                    raise ValueError(f"cut exceeds retokenized source: {tid}/{cut}")
                gen = generated[tid][:cut]
                structured = bool(trace.get("structured", False))
                expected_id = f"ceil/{tid}/{r['checkpoint']}"
                source_batch = "first_480_lines" if source_line <= args.first_batch_lines else "later_lines"
            else:
                source_line, source_batch, gen, structured = None, "problem_start", [], False
                expected_id = f"ceil/{pid}/t0"
            if r["prefix_id"] != expected_id:
                raise ValueError(f"prefix_id mismatch; expected {expected_id}")
            prompt_key = (pid, structured)
            if prompt_key not in prompts:
                prompt = build_prompt(tokenizer, problems[pid]["problem"], structured)
                prompts[prompt_key] = tokenizer.encode(prompt)
            prompt_ids = prompts[prompt_key]
            match = matching_bos(prompt_ids, gen, tokenizer.bos_token_id, r["prefix_sha"])
            if match is None:
                issues.append(dict(line=physical_line, error="neither one- nor two-BOS candidate matches stored prefix_sha"))
            results.append(dict(physical_line=physical_line, prefix_id=r["prefix_id"], round=r["round"],
                                trace_id=tid, problem_id=pid, checkpoint=r["checkpoint"], cut_token_idx=cut,
                                source_line=source_line, source_batch=source_batch,
                                matched_bos_count=match, saved_prefix_sha=r["prefix_sha"],
                                one_bos_sha=input_hash(prompt_ids, gen),
                                two_bos_sha=input_hash([tokenizer.bos_token_id] + prompt_ids, gen),
                                encoded_prefix_tokens=len(gen), prompt_tokens_one_bos=len(prompt_ids)))
        except (KeyError, TypeError, ValueError) as exc:
            issues.append(dict(line=physical_line, error=str(exc)))
    metadata_checks = {"provided": bool(args.meta)}
    if args.meta:
        import pandas as pd
        metadata = pd.read_parquet(args.meta)
        duplicate = metadata.duplicated(["trace_id", "pos"]).sum()
        keys = set(zip(metadata.trace_id, metadata.pos))
        absent = [r["prefix_id"] for r in results if r["trace_id"] is not None
                  and (r["trace_id"], r["cut_token_idx"]) not in keys]
        metadata_checks.update(rows=len(metadata), duplicate_trace_positions=int(duplicate), missing_rollout_checkpoints=absent)
        if duplicate or absent:
            issues.append(dict(line="metadata", error=f"{duplicate} duplicate metadata keys; {len(absent)} missing rollout checkpoints"))
    groups = Counter((r["source_batch"], r["checkpoint"], r["matched_bos_count"]) for r in results)
    cohort_ids = sorted({r["trace_id"] for r in results if r["trace_id"] is not None})
    source_batch_counts = Counter("first_480_lines" if sources[t][0] <= args.first_batch_lines else "later_lines" for t in cohort_ids)
    report = dict(status="hash_candidates_verified" if not issues and results else "issues_found",
                  original_generation_token_identity="not_verified", activation_input_token_identity="not_verified",
                  inputs={"traces_sha256": sha_file(args.traces), "rollouts_sha256": sha_file(args.rollouts),
                          "generate_code_sha256": sha_file(args.generation_code), "tokenizer": manifest},
                  valid_audited_records=len(results), hash_matches_by_bos=dict(Counter(str(r["matched_bos_count"]) for r in results)),
                  distinct_source_traces=len(cohort_ids), source_batch_trace_counts=dict(source_batch_counts),
                  grouped_counts=[dict(source_batch=k[0], checkpoint=k[1], matched_bos_count=k[2], records=v) for k, v in sorted(groups.items(), key=lambda item: str(item[0]))],
                  metadata=metadata_checks, issues=issues,
                  interpretation="Matching candidates cryptographically reproduce saved rollout input hashes using this repository's hash definition. They do not independently establish the original generation or historical activation inputs.",
                  source_batch_note=f"Line grouping uses configured first_batch_lines={args.first_batch_lines}; this grouping does not itself prove historical BOS counts.")
    (args.out / "alignment_hash_report.json").write_text(json.dumps(report, indent=2) + "\n")
    if results:
        write_csv(args.out / "alignment_hash_rows.csv", results, list(results[0]))
    write_csv(args.out / "alignment_hash_issues.csv", issues, ["line", "error"])
    print(json.dumps({k: report[k] for k in ["status", "valid_audited_records", "hash_matches_by_bos", "distinct_source_traces", "source_batch_trace_counts", "metadata", "issues"]}, indent=2))
    return 0 if report["status"] == "hash_candidates_verified" else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--rollout-manifest", type=Path, help="optional cleaning manifest; rejects incomplete/changed derived files")
    parser.add_argument("--tokenizer-dir", type=Path, required=True, help="local tokenizer files plus tokenizer_manifest.json")
    parser.add_argument("--generation-code", type=Path, default=Path(__file__).with_name("generate.py"))
    parser.add_argument("--meta", type=Path)
    parser.add_argument("--first-batch-lines", type=int, default=480)
    parser.add_argument("--out", type=Path, required=True)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
