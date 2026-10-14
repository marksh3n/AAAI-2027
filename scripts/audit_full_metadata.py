#!/usr/bin/env python3
"""Read-only full-cache metadata audit; never validates an unavailable array."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def csv_write(path, rows, fields):
    with Path(path).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def run(args):
    sys.path.insert(0, str(args.repo_src))
    from audit_alignment import LocalTokenizer, input_hash, load_build_prompt, strict_loads

    args.out.mkdir(parents=True, exist_ok=True)
    cfg = strict_loads((args.full_dir / "full_config.json").read_text())
    m = pd.read_parquet(args.full_dir / "meta_full.parquet")
    tm = strict_loads((args.tokenizer_dir / "tokenizer_manifest.json").read_text())
    for f in tm["files"]:
        if sha(args.tokenizer_dir / f["name"]) != f["sha256"]:
            raise ValueError(f"Tokenizer artifact changed: {f['name']}")
    tok = LocalTokenizer(args.tokenizer_dir)
    build_prompt = load_build_prompt(args.generation_code)
    traces = [strict_loads(line) for line in args.traces.open()]
    if len({t["trace_id"] for t in traces}) != len(traces):
        raise ValueError("Duplicate source trace IDs")
    by_source = {t["trace_id"]: (i, t) for i, t in enumerate(traces)}
    by_problem = {t["problem_id"]: t for t in traces}
    issues, overlaps, population = [], [], []

    def flag(check, location, detail):
        issues.append(dict(check=check, location=str(location), detail=str(detail)))

    required = {"trace_id", "problem_id", "line", "n_bos", "ckpt", "pos", "is_correct", "trunc", "n_gen", "prefix", "restarts", "row"}
    if not required <= set(m.columns):
        raise ValueError(f"Missing metadata columns {required - set(m.columns)}")
    if cfg.get("rows") != len(m) or cfg.get("traces") != len(traces):
        flag("config_counts", "all", f"metadata={len(m)}, source={len(traces)}, config={cfg}")
    if cfg.get("layers") != list(range(-1, 28)):
        flag("config_layers", "all", cfg.get("layers"))
    if cfg.get("layer_convention") != "hidden_states[l+1]":
        flag("layer_convention", "all", cfg.get("layer_convention"))
    if cfg.get("bos") != f"1 for lines < {args.pilot_lines}, else 2":
        flag("config_bos_policy", "all", cfg.get("bos"))
    if cfg.get("token_index") != "pos-1; end = last trace token":
        flag("config_token_index", "all", cfg.get("token_index"))
    if m.row.tolist() != list(range(len(m))):
        flag("row_order", "all", "row is not the contiguous zero-based parquet row order")
    for row in m[m.duplicated(["trace_id", "ckpt"], keep=False)].itertuples():
        flag("duplicate_trace_checkpoint", row.row, f"{row.trace_id}/{row.ckpt}")
    for tid in sorted(set(m.trace_id) - set(by_source)):
        flag("unknown_trace", tid, "Metadata source trace not found")

    rollout_by_trace, t0 = defaultdict(list), []
    rollout_seen = set()
    for line, raw in enumerate(args.rollouts.open(), 1):
        r = strict_loads(raw)
        key = (r["prefix_id"], r["round"])
        if key in rollout_seen:
            raise ValueError(f"Duplicate rollout prefix/round: {key}")
        rollout_seen.add(key)
        small = {k: v for k, v in r.items() if k != "samples"}
        if r["trace_id"] is None:
            t0.append(small)
        else:
            rollout_by_trace[r["trace_id"]].append(small)
    grouped = {tid: rows for tid, rows in m.groupby("trace_id", sort=False)}
    prompt_cache, expected_sequence = {}, []
    exact_fixed = exact_end = total_fixed = total_end = 0
    restart_pattern = re.compile(r"\b(wait|hold on|let me re|actually)\b", re.I)
    for line, trace in enumerate(traces):
        tid = trace["trace_id"]
        ids = tok.encode(trace["text"])
        rows = grouped.get(tid, m.iloc[:0])
        expected_positions = [p for p in cfg["positions"] if p <= len(ids)]
        has_end = not trace["trunc"] and "</think>" in trace["text"]
        expected_ckpts = [str(p) for p in expected_positions] + (["end"] if has_end else [])
        expected_sequence.extend((tid, c) for c in expected_ckpts)
        if list(rows.ckpt) != expected_ckpts:
            flag("checkpoint_set_or_order", tid, f"expected={expected_ckpts}, actual={list(rows.ckpt)}")
        for row in rows.itertuples(index=False):
            for key in ("problem_id", "n_gen", "trunc"):
                if getattr(row, key) != trace[key]:
                    flag("source_field", row.row, f"{key}: {getattr(row,key)!r} != {trace[key]!r}")
            if row.line != line:
                flag("source_line", row.row, f"{row.line} != {line}")
            expected_bos = 1 if line < args.pilot_lines else 2
            if row.n_bos != expected_bos:
                flag("metadata_bos_policy", row.row, f"{row.n_bos} != {expected_bos}")
            if bool(row.is_correct) != (bool(trace["is_correct"]) and not trace["trunc"]):
                flag("correctness_label", row.row, "Mismatch with source correctness-and-not-truncated convention")
            expected_pos = len(ids) if row.ckpt == "end" else int(row.ckpt)
            if row.pos != expected_pos:
                flag("checkpoint_position", row.row, f"{row.pos} != {expected_pos}")
            expected_prefix = trace["text"] if row.ckpt == "end" else tok.backend.decode(ids[:expected_pos], skip_special_tokens=False)
            matches = row.prefix == expected_prefix
            if row.ckpt == "end":
                total_end += 1
                exact_end += matches
            else:
                total_fixed += 1
                exact_fixed += matches
            if not matches:
                flag("prefix_text", row.row, f"{tid}/{row.ckpt}: expected_sha={hashlib.sha256(expected_prefix.encode()).hexdigest()}")
            restart = len(restart_pattern.findall(expected_prefix)) / max(expected_pos, 1) * 1000
            if not np.isclose(row.restarts, restart, rtol=1e-12, atol=1e-12):
                flag("restart_density", row.row, f"{row.restarts} != {restart}")
        for rollout in rollout_by_trace.get(tid, []):
            ckpt = rollout["checkpoint"]
            cut = rollout["cut_token_idx"]
            selection = rows[rows.ckpt == str(cut)]
            if len(selection) != 1:
                flag("rollout_meta_join", rollout["prefix_id"], f"Metadata matches={len(selection)}")
                continue
            mr = selection.iloc[0]
            if ckpt != f"p{cut}" or rollout["problem_id"] != trace["problem_id"]:
                flag("rollout_source_cut", rollout["prefix_id"], "Trace/problem/checkpoint mismatch")
            prompt_key = (trace["problem_id"], trace.get("structured", False))
            if prompt_key not in prompt_cache:
                rendered = build_prompt(tok, trace["problem"], trace.get("structured", False))
                p = tok.encode(rendered)
                if p[0] != tok.bos_token_id or p[1] == tok.bos_token_id:
                    raise ValueError("Expected one-BOS chat template")
                prompt_cache[prompt_key] = p
            one = prompt_cache[prompt_key]
            candidate = [tok.bos_token_id] * (int(mr.n_bos) - 1) + one
            h = input_hash(candidate, ids[:cut])
            saved = rollout["prefix_sha"]
            status = "declared_input_hash_matches" if h == saved else "declared_input_hash_mismatch"
            overlaps.append(dict(prefix_id=rollout["prefix_id"], trace_id=tid, problem_id=trace["problem_id"],
                                 checkpoint=ckpt, cut_token_idx=cut, meta_row=int(mr.row), source_line=line,
                                 source_batch="pilot" if line < args.pilot_lines else "main", metadata_n_bos=int(mr.n_bos),
                                 one_bos_hash_matches=input_hash(one, ids[:cut]) == saved,
                                 two_bos_hash_matches=input_hash([tok.bos_token_id] + one, ids[:cut]) == saved,
                                 metadata_declared_input_sha=h, saved_rollout_prefix_sha=saved, status=status,
                                 numerical_activation_provenance_verified=False))
    if list(zip(m.trace_id, m.ckpt)) != expected_sequence:
        flag("global_row_sequence", "all", "Metadata order differs from source order × planned checkpoints")
    for tid in set(rollout_by_trace) - set(by_source):
        flag("unknown_rollout_source", tid, "Trace does not exist")
    t0_two = 0
    for r in t0:
        t = by_problem[r["problem_id"]]
        ids = tok.encode(build_prompt(tok, t["problem"], False))
        if r["cut_token_idx"] != 0:
            flag("t0_cut", r["prefix_id"], r["cut_token_idx"])
        t0_two += input_hash([tok.bos_token_id] + ids, []) == r["prefix_sha"]
    overlap_groups = Counter((r["source_batch"], r["checkpoint"], r["status"]) for r in overlaps)
    for (ckpt, n_bos), group in m.groupby(["ckpt", "n_bos"], sort=False):
        population.append(dict(checkpoint=ckpt, metadata_n_bos=int(n_bos), rows=len(group),
                               traces=group.trace_id.nunique(), correct=int(group.is_correct.sum()),
                               wrong=int((~group.is_correct).sum()), truncated=int(group.trunc.sum())))
    analysis_counts = []
    for ckpt, group in m.groupby("ckpt", sort=False):
        group = group[~group.trunc]
        distinct_labels = group.groupby("problem_id").is_correct.nunique()
        group = group[group.problem_id.isin(distinct_labels[distinct_labels == 2].index)]
        analysis_counts.append(dict(ckpt=str(ckpt), traces=len(group), problems=int(group.problem_id.nunique()),
                                    wrong=int((~group.is_correct).sum())))
    table_count_checks = []
    count_path = args.full_dir / "e0_full_table_counts.csv"
    if count_path.exists():
        table_counts = pd.read_csv(count_path, dtype={"ckpt": str})
        reconstructed = pd.DataFrame(analysis_counts)
        joined = table_counts.merge(reconstructed, on="ckpt", suffixes=("_table", "_reconstructed"), how="outer")
        for record in joined.to_dict("records"):
            record["matches"] = all(record[f"{col}_table"] == record[f"{col}_reconstructed"]
                                    for col in ["traces", "problems", "wrong"])
            table_count_checks.append(record)
            if not record["matches"]:
                flag("table_population_counts", record["ckpt"], record)
    payload_bytes = len(m) * len(cfg["layers"]) * 1536 * 2
    report = dict(status="metadata_checks_pass" if not issues else "metadata_issues_found",
                  source_traces=len(traces), metadata_rows=len(m), layers=cfg["layers"],
                  expected_array_payload_bytes=payload_bytes,
                  array_downloaded=False, numerical_array_validated=False,
                  array_note="acts_full.npy was not available locally; metadata and expected byte count do not validate its shape, values, row binding or extraction provenance.",
                  historical_generation_ids_verified=False,
                  exact_fixed_prefix_matches=int(exact_fixed), fixed_prefix_rows=total_fixed,
                  exact_end_prefix_matches=int(exact_end), end_rows=total_end,
                  input_hash_overlap=dict(rows=len(overlaps), distinct_source_traces=len({r['trace_id'] for r in overlaps}),
                                          matching_metadata_declared_inputs=sum(r['status']=='declared_input_hash_matches' for r in overlaps),
                                          mismatching_metadata_declared_inputs=sum(r['status']=='declared_input_hash_mismatch' for r in overlaps),
                                          t0_two_bos_matches=int(t0_two), t0_records=len(t0),
                                          grouped=[dict(source_batch=k[0], checkpoint=k[1], status=k[2], rows=v) for k,v in sorted(overlap_groups.items())]),
                  interpretation="The overlap check reconstructs the inputs declared by metadata.n_bos and compares their hashes to saved rollout hashes. It does not certify the actual numerical activation inputs. Pilot/main BOS assignments follow full_config and the source line boundary, not original saved generation IDs.",
                  issues=issues, population=population, analysis_counts=analysis_counts,
                  table_population_counts_checked=bool(table_count_checks),
                  table_population_counts_match=all(r["matches"] for r in table_count_checks) if table_count_checks else None,
                  hashes={"traces":sha(args.traces), "meta_full":sha(args.full_dir/'meta_full.parquet'),
                          "full_config":sha(args.full_dir/'full_config.json'), "rollouts":sha(args.rollouts),
                          "generation_code":sha(args.generation_code)}, tokenizer=tm)
    (args.out/'full_metadata_report.json').write_text(json.dumps(report,indent=2)+'\n')
    csv_write(args.out/'full_metadata_issues.csv',issues,['check','location','detail'])
    csv_write(args.out/'full_metadata_rollout_overlap.csv',overlaps,list(overlaps[0]))
    csv_write(args.out/'full_metadata_population.csv',population,list(population[0]))
    if table_count_checks:
        csv_write(args.out/'full_metadata_table_population_checks.csv',table_count_checks,list(table_count_checks[0]))
    print(json.dumps({k:report[k] for k in ['status','source_traces','metadata_rows','exact_fixed_prefix_matches','fixed_prefix_rows','exact_end_prefix_matches','end_rows','input_hash_overlap','issues']},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--traces',type=Path,required=True)
    p.add_argument('--rollouts',type=Path,required=True)
    p.add_argument('--full-dir',type=Path,required=True)
    p.add_argument('--tokenizer-dir',type=Path,required=True)
    p.add_argument('--generation-code',type=Path,required=True)
    p.add_argument('--repo-src',type=Path,required=True)
    p.add_argument('--pilot-lines',type=int,default=480)
    p.add_argument('--out',type=Path,required=True)
    run(p.parse_args())
