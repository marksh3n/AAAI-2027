#!/usr/bin/env python3
"""Build an unfrozen MATH L3–5 candidate pool; never generate model answers.

Local preparation uses only the Python standard library. The optional download
command fetches pinned public source files and lazily imports pyarrow for the
full-MATH parquet. No dataset rows are treated as an official train/test split
unless that distinction is present in their source: the selected full-MATH
mirror exposes its combined collection as a single split named ``train``.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import re
import unicodedata
import urllib.parse
import urllib.request

MATH_REPO = "qwedsacf/competition_math"
MATH500_REPO = "HuggingFaceH4/MATH-500"
NORMALIZATION = "Unicode NFKC, then collapse all whitespace; preserve case and punctuation"
CANONICAL_MATH_PATH = re.compile(
    r"^(?:train|test)/(?:algebra|counting_and_probability|geometry|intermediate_algebra|number_theory|prealgebra|precalculus)/[^/]+\.json$"
)


def normalize_problem(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError("Problem text must be a string")
    return " ".join(unicodedata.normalize("NFKC", text).split())


def text_hash(text: str) -> str:
    return hashlib.sha256(normalize_problem(text).encode("utf-8")).hexdigest()


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _no_duplicate_keys(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"Duplicate JSON key: {key}")
        out[key] = value
    return out


def read_jsonl(path: Path):
    with Path(path).open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line, object_pairs_hook=_no_duplicate_keys,
                                 parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)))
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected a JSON object")
            yield line_no, row


def _problem(row):
    # Do not fall back to `text`: trace exports use it for the model's response.
    value = row.get("problem", row.get("question"))
    return value if isinstance(value, str) and normalize_problem(value) else None


def source_keys(row: dict) -> set[tuple[str, str]]:
    """Return only coherent, explicitly namespaced IDs or canonical MATH paths.

    Bare integer/string row IDs and problem IDs from unrelated exports are not
    comparable. Dataset-relative numeric IDs require an explicit namespace.
    Canonical MATH train/test paths carry their own original dataset namespace.
    """
    keys = set()
    namespace, identifier = row.get("source_dataset"), row.get("source_id")
    if isinstance(namespace, str) and namespace.strip() and identifier is not None:
        if isinstance(identifier, (str, int)) and not isinstance(identifier, bool):
            keys.add((namespace.strip(), str(identifier)))
    for field in ("source", "_source"):
        source = row.get(field)
        if isinstance(source, dict):
            namespace = source.get("dataset", source.get("dataset_id"))
            identifier = source.get("source_id", source.get("id"))
            if isinstance(namespace, str) and namespace.strip() and isinstance(identifier, (str, int)) and not isinstance(identifier, bool):
                keys.add((namespace.strip(), str(identifier)))
    for field in ("unique_id", "source_id", "path", "id"):
        identifier = row.get(field)
        if isinstance(identifier, str):
            identifier = identifier.removeprefix("MATH/")
            if CANONICAL_MATH_PATH.fullmatch(identifier):
                keys.add(("hendrycks/MATH", identifier))
    return keys


def _escaped(text: str, idx: int) -> bool:
    backslashes = 0
    idx -= 1
    while idx >= 0 and text[idx] == "\\":
        backslashes += 1
        idx -= 1
    return backslashes % 2 == 1


def extract_boxed_answer(solution: str) -> str | None:
    """Extract the last boxed/fbox expression, respecting nested/escaped braces.

    An unbalanced or unbraced final box is rejected, not replaced by an earlier
    intermediate answer. No mathematical normalization or answer guessing occurs.
    """
    if not isinstance(solution, str):
        return None
    markers = list(re.finditer(r"\\(?:boxed|fbox)\b", solution))
    if not markers:
        return None
    start = markers[-1].end()
    while start < len(solution) and solution[start].isspace():
        start += 1
    if start >= len(solution) or solution[start] != "{":
        return None
    depth = 1
    for i in range(start + 1, len(solution)):
        if solution[i] == "{" and not _escaped(solution, i):
            depth += 1
        elif solution[i] == "}" and not _escaped(solution, i):
            depth -= 1
            if depth == 0:
                return solution[start + 1:i].strip() or None
    return None


def _level(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        match = re.fullmatch(r"(?:Level\s+)?([1-5])", value.strip(), re.I)
        if match:
            return int(match[1])
    return None


def _gold(row):
    for key in ("gold", "answer"):
        value = row.get(key)
        if isinstance(value, (str, int, float)) and not isinstance(value, bool) and str(value).strip():
            return str(value).strip(), key
    answer = extract_boxed_answer(row.get("solution"))
    return answer, "last_boxed_solution" if answer is not None else None


def _exclusion_index(path: Path):
    hashes, ids, count = set(), set(), 0
    for line_no, row in read_jsonl(path):
        count += 1
        problem, keys = _problem(row), source_keys(row)
        if problem is None and not keys:
            raise ValueError(f"{path}:{line_no}: exclusion row has neither problem text nor coherent source ID")
        if problem is not None:
            hashes.add(text_hash(problem))
        ids.update(keys)
    if count == 0:
        raise ValueError(f"Exclusion file is empty: {path}")
    return hashes, ids, count


def prepare_pool(math_path, math500_path, inspected_paths, out_dir, seed):
    math_path, math500_path, out_dir = Path(math_path), Path(math500_path), Path(out_dir)
    inspected_paths = [Path(p) for p in inspected_paths]
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an explicit integer")
    input_paths = [math_path, math500_path, *inspected_paths]
    # Read and validate all exclusion files before writing any outputs.
    math_hashes, math_ids, math500_count = _exclusion_index(math500_path)
    inspected_hashes, inspected_ids, inspected_counts = set(), set(), {}
    for path in inspected_paths:
        hashes, ids, count = _exclusion_index(path)
        inspected_hashes.update(hashes)
        inspected_ids.update(ids)
        inspected_counts[str(path)] = count
    for filename in ("candidate_pool.jsonl", "preparation_report.json", "exclusions.csv"):
        if (out_dir / filename).resolve() in {p.resolve() for p in input_paths}:
            raise ValueError("Output would overwrite a source input")
    exclusions, groups = [], defaultdict(list)
    source_count, eligible_level_count = 0, 0
    all_levels = Counter()
    input_sha = file_hash(math_path)

    def exclude(line, row, reason, digest=""):
        exclusions.append({"source_line": line, "problem_sha256": digest,
                           "source_ids": json.dumps(sorted(source_keys(row))), "reason": reason})

    for line_no, row in read_jsonl(math_path):
        source_count += 1
        level = _level(row.get("level"))
        all_levels[str(level) if level is not None else "unrecognized"] += 1
        if level not in (3, 4, 5):
            exclude(line_no, row, "outside_levels_3_5" if level is not None else "invalid_level")
            continue
        eligible_level_count += 1
        problem = _problem(row)
        if problem is None:
            exclude(line_no, row, "missing_problem")
            continue
        digest, ids = text_hash(problem), source_keys(row)
        if digest in math_hashes:
            exclude(line_no, row, "math500_text", digest)
        elif ids & math_ids:
            exclude(line_no, row, "math500_source_id", digest)
        elif digest in inspected_hashes:
            exclude(line_no, row, "previously_inspected_text", digest)
        elif ids & inspected_ids:
            exclude(line_no, row, "previously_inspected_source_id", digest)
        else:
            gold, gold_method = _gold(row)
            if gold is None:
                exclude(line_no, row, "missing_or_malformed_gold", digest)
                continue
            groups[digest].append((line_no, row, problem, level, gold, gold_method))
    pool, duplicate_gold_conflicts = [], 0
    for digest, members in sorted(groups.items()):
        # Reject ambiguous labels instead of silently selecting a conflicting gold.
        if len({normalize_problem(x[4]) for x in members}) > 1:
            duplicate_gold_conflicts += 1
            for line_no, row, *_ in members:
                exclude(line_no, row, "duplicate_gold_conflict", digest)
            continue
        if len({x[3] for x in members}) > 1:
            for line_no, row, *_ in members:
                exclude(line_no, row, "duplicate_level_conflict", digest)
            continue
        # Canonical representative independent of input row ordering.
        members.sort(key=lambda x: json.dumps(x[1], sort_keys=True, ensure_ascii=False))
        line_no, row, problem, level, gold, gold_method = members[0]
        source = dict(row.get("_source", {})) if isinstance(row.get("_source"), dict) else {}
        source.update({"input_file": math_path.name, "input_sha256": input_sha,
                       "input_line": line_no, "coherent_ids": [list(k) for k in sorted(source_keys(row))]})
        item = {"problem_id": "mathsha256:" + digest, "problem": problem, "gold": gold,
                "solution": row.get("solution"), "level": level,
                "subject": row.get("subject", row.get("type")),
                "normalized_text_sha256": digest, "gold_extraction": gold_method,
                "source": source}
        pool.append(item)
        for duplicate_line, duplicate, *_ in members[1:]:
            exclude(duplicate_line, duplicate, "duplicate_problem_text", digest)
    random.Random(seed).shuffle(pool)
    for rank, row in enumerate(pool, 1):
        row["candidate_order"] = rank
    out_dir.mkdir(parents=True, exist_ok=True)
    pool_path = out_dir / "candidate_pool.jsonl"
    with pool_path.open("w", encoding="utf-8") as f:
        for row in pool:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
    exclusions.sort(key=lambda r: r["source_line"])
    with (out_dir / "exclusions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["source_line", "problem_sha256", "source_ids", "reason"])
        writer.writeheader()
        writer.writerows(exclusions)
    report = {"artifact_status": "candidate_pool_not_frozen", "final_confirmation_set_selected": False,
              "normalization": NORMALIZATION, "seed": seed, "ordering": "sort by normalized-text SHA256, then seeded shuffle",
              "source_rows": source_count, "source_levels": dict(sorted(all_levels.items())),
              "eligible_level_rows_before_exclusions": eligible_level_count,
              "math500_exclusion_records": math500_count,
              "additional_inspected_exports": inspected_counts,
              "candidate_count": len(pool), "candidate_levels": dict(sorted(Counter(str(r['level']) for r in pool).items())),
              "excluded_by_reason": dict(sorted(Counter(r['reason'] for r in exclusions).items())),
              "duplicate_gold_conflict_groups": duplicate_gold_conflicts,
              "input_files": [{"path": str(p), "bytes": p.stat().st_size, "sha256": file_hash(p)} for p in input_paths],
              "candidate_file": {"path": str(pool_path), "bytes": pool_path.stat().st_size, "sha256": file_hash(pool_path)},
              "limitations": ["Freshness is relative only to the MATH-500 and inspected exports supplied here.",
                              "NFKC/whitespace matching does not rule out paraphrases, semantic duplicates or model pretraining exposure.",
                              "A mirror split named train does not certify original MATH train/test membership.",
                              "Freeze the protocol and final sample size before selecting or generating confirmation outcomes."]}
    if source_count != len(pool) + len(exclusions):
        raise AssertionError("Every source row must be retained once or have an exclusion reason")
    (out_dir / "preparation_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def _fetch(url, destination):
    request = urllib.request.Request(url, headers={"User-Agent": "MJJS-candidate-preparation/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response, Path(destination).open("wb") as out:
        while block := response.read(1024 * 1024):
            out.write(block)


def download_sources(out_dir, math_revision=None, math500_revision=None):
    """Download pinned source files; pyarrow is needed only for parquet conversion.

    Resolve the requested revision to a commit, then fetch that immutable URL.
    Source JSONL exports carry per-row dataset/split/revision provenance.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifests, outputs = [], {}
    for role, repo, revision in [("math_full", MATH_REPO, math_revision), ("math500", MATH500_REPO, math500_revision)]:
        api = f"https://huggingface.co/api/datasets/{repo}"
        if revision:
            api += "/revision/" + urllib.parse.quote(revision, safe="")
        with urllib.request.urlopen(api, timeout=60) as response:
            info = json.load(response)
        commit = info["sha"]
        if not re.fullmatch(r"[0-9a-f]{40}", commit):
            raise ValueError("HF did not return a pinned commit SHA")
        files = [r["rfilename"] for r in info["siblings"]]
        if role == "math_full":
            selected = [f for f in files if f.endswith(".parquet")]
            if len(selected) != 1 or not selected[0].startswith("data/train-"):
                raise ValueError(f"Unexpected full-MATH file layout: {selected}; inspect before proceeding")
            source_file, split = selected[0], "train"
        else:
            if "test.jsonl" not in files:
                raise ValueError("Expected MATH-500 test.jsonl was not found")
            source_file, split = "test.jsonl", "test"
        url = f"https://huggingface.co/datasets/{repo}/resolve/{commit}/{source_file}"
        raw = out_dir / (role + ".source" + Path(source_file).suffix)
        _fetch(url, raw)
        if raw.suffix == ".parquet":
            try:
                import pyarrow.parquet as pq
            except ImportError as exc:
                raise RuntimeError("Download conversion needs pyarrow; run: pip install pyarrow") from exc
            rows = pq.read_table(raw).to_pylist()
        else:
            rows = [r for _, r in read_jsonl(raw)]
        target = out_dir / f"{role}.jsonl"
        with target.open("w", encoding="utf-8") as out:
            for idx, original in enumerate(rows):
                record = dict(original)
                if "_source" in record:
                    raise ValueError("Source unexpectedly already has _source; inspect before replacing")
                record["_source"] = {"dataset": repo, "revision": commit, "config": "default",
                                     "split": split, "row_index": idx, "file": source_file}
                out.write(json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
        manifests.append({"role": role, "dataset": repo, "revision": commit, "split": split,
                          "source_file": source_file, "url": url, "row_count": len(rows),
                          "raw_file": str(raw), "raw_sha256": file_hash(raw),
                          "jsonl_file": str(target), "jsonl_sha256": file_hash(target)})
        outputs[role] = str(target)
    manifest = {"downloaded_at_utc": datetime.now(timezone.utc).isoformat(), "sources": manifests,
                "note": "The full-MATH mirror combines 12.5k problems in one split; no original split membership is inferred."}
    (out_dir / "source_download_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    download = sub.add_parser("download", help="Download public pinned sources; optional pyarrow dependency")
    download.add_argument("--out", type=Path, required=True)
    download.add_argument("--math-revision")
    download.add_argument("--math500-revision")
    prepare = sub.add_parser("prepare", help="Prepare candidate JSONL using stdlib only")
    prepare.add_argument("--math-full", type=Path, required=True)
    prepare.add_argument("--math500", type=Path, required=True)
    prepare.add_argument("--inspected", type=Path, nargs="*", default=[])
    prepare.add_argument("--out", type=Path, required=True)
    prepare.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    if args.command == "download":
        result = download_sources(args.out, args.math_revision, args.math500_revision)
    else:
        result = prepare_pool(args.math_full, args.math500, args.inspected, args.out, args.seed)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
