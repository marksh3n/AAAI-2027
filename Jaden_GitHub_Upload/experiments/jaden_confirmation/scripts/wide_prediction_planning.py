#!/usr/bin/env python3
"""Convert supplied wide P(wrong) exports and run illustrative planning scenarios.

No models are fitted and no confirmation protocol is approved here. The CSVs do
not establish their out-of-fold, checkpoint, or method-selection provenance.
The two populations are evaluated separately, never pooled.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from plan_sample_size import plan_sample_size, validate_predictions

INPUT_FILES = {
    "pop342": "e0_6144_preds_pop342.csv",
    "cohort227": "e0_6144_preds_cohort227.csv",
}
PROBE_COLUMN = "L20_hgb_topk50"
BASELINE_COLUMNS = ("base_lr_tfidf", "base_lr_len")
EFFECTS = (.02, .03, .05, .10)
SCORE_NAME = re.compile(r"(?:base_lr_(?:len|tfidf)|L\d+_(?:lr|hgb|mlp)_(?:raw|svd50|topk50|svd50\+len))\Z")
BASELINE_NOTES = {
    "base_lr_len": (
        "Observable-prefix baseline: the supplied repository's len_features uses "
        "log1p(observed token count, observed paragraph count), not final trace length. "
        "The actual export's provenance is unverified."
    ),
    "base_lr_tfidf": "Prefix TF-IDF baseline column; actual export provenance is unverified.",
}


def _ids(series: pd.Series, name: str) -> pd.Series:
    if series.isna().any():
        raise ValueError(f"Missing {name} values are not allowed.")
    if any(isinstance(x, (bool, np.bool_)) or not isinstance(x, (str, int, np.integer)) for x in series):
        raise ValueError(f"{name} must contain string or integer IDs, not booleans/floating-point values.")
    values = series.astype(str)
    if values.str.strip().eq("").any() or not values.eq(values.str.strip()).all():
        raise ValueError(f"{name} must be nonempty and have no surrounding whitespace.")
    return values


def convert_wide(frame: pd.DataFrame, probe_column: str, baseline_column: str,
                 checkpoint: str = "p6144") -> tuple[pd.DataFrame, dict]:
    """Return correctness-oriented predictions and an explicit provenance note.

    y_correct = 1-y_wrong and BOTH scores = 1-P(wrong). A label-only or
    score-only reversal would change the ranking target and is not performed.
    Only known prediction-column names from this export schema are accepted.
    """
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("Input must be a nonempty DataFrame.")
    if frame.columns.duplicated().any():
        raise ValueError("Duplicate column names are not allowed.")
    if not isinstance(checkpoint, str) or not re.fullmatch(r"p[1-9]\d*", checkpoint):
        raise ValueError("Declare one checkpoint such as 'p6144'.")
    if probe_column == baseline_column:
        raise ValueError("Probe and baseline must be distinct prediction columns.")
    for column in (probe_column, baseline_column):
        if not isinstance(column, str) or SCORE_NAME.fullmatch(column) is None:
            raise ValueError(f"{column!r} is not a recognized prediction column; metadata cannot be used as a score.")
    required = ["problem_id", "trace_id", "y_wrong", probe_column, baseline_column]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    if frame[required].isna().any().any():
        raise ValueError("Missing IDs, labels, or selected probability values are not allowed.")
    problems = _ids(frame["problem_id"], "problem_id")
    traces = _ids(frame["trace_id"], "trace_id")
    if traces.duplicated().any():
        raise ValueError("Duplicate trace_id: each trace must appear exactly once in this single-checkpoint export.")
    try:
        labels = pd.to_numeric(frame["y_wrong"], errors="raise")
    except (ValueError, TypeError) as exc:
        raise ValueError("y_wrong must contain only 0/1 labels.") from exc
    if not labels.isin([0, 1]).all():
        raise ValueError("y_wrong must contain only 0/1 labels.")
    scores = {}
    for column in (probe_column, baseline_column):
        try:
            values = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=float)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{column}: probabilities must be numeric.") from exc
        if not np.isfinite(values).all():
            raise ValueError(f"{column}: nonfinite probabilities are not allowed.")
        if ((values < 0) | (values > 1)).any():
            raise ValueError(f"{column}: probabilities must be in [0, 1].")
        scores[column] = 1.0 - values
    if "checkpoint" in frame.columns:
        if frame["checkpoint"].isna().any() or not frame["checkpoint"].astype(str).eq(checkpoint).all():
            raise ValueError("Embedded checkpoint values conflict with the declared checkpoint.")
    converted = pd.DataFrame({
        "problem_id": problems.to_numpy(), "trace_id": traces.to_numpy(),
        "checkpoint": checkpoint, "y_correct": 1 - labels.to_numpy(dtype=int),
        "probe_score": scores[probe_column], "baseline_score": scores[baseline_column],
    })
    converted = validate_predictions(converted)
    note = {
        "source_label_column": "y_wrong", "source_score_orientation": "assumed P(wrong) from the supplied export description",
        "conversion": "y_correct=1-y_wrong; probe_score=1-probe_column; baseline_score=1-baseline_column",
        "probe_column": probe_column, "baseline_column": baseline_column,
        "checkpoint": checkpoint,
        "checkpoint_basis": ("declared argument and matching embedded column; not independently verified"
                             if "checkpoint" in frame.columns else
                             "declared argument; bundled p6144 is inferred from filenames/user description, not independently verified"),
        "oof_provenance": "unverified: these CSVs do not establish grouped out-of-fold fitting or feature-selection provenance",
        "protocol_status": "illustrative development planning only; not an approved confirmation protocol",
        "probe_note": "hgb denotes histogram gradient boosting, not a single decision tree; exact export fitting provenance is unverified",
        "baseline_note": BASELINE_NOTES.get(baseline_column, "Use the exact column name; fitting provenance is unverified."),
    }
    return converted, note


def _read_wide(path: Path) -> pd.DataFrame:
    # Pandas otherwise silently mangles duplicate CSV headers into name/name.1.
    with path.open(newline="") as handle:
        header = next(csv.reader(handle), [])
    if not header or len(header) != len(set(header)):
        raise ValueError(f"{path.name}: empty or duplicate CSV column names.")
    return pd.read_csv(path, dtype={"problem_id": str, "trace_id": str})


def _overlap(frames: dict[str, pd.DataFrame]) -> dict:
    left = frames["pop342"].set_index("trace_id")
    right = frames["cohort227"].set_index("trace_id")
    common = sorted(set(left.index) & set(right.index))
    score_differences = {}
    for column in (PROBE_COLUMN, *BASELINE_COLUMNS):
        a = left.loc[common, column].to_numpy(dtype=float)
        b = right.loc[common, column].to_numpy(dtype=float)
        score_differences[column] = {
            "shared_traces_with_different_scores": int(np.count_nonzero(a != b)),
            "max_absolute_score_difference": float(np.max(np.abs(a - b))) if common else None,
        }
    inconsistent = sum(
        str(left.at[tid, "problem_id"]) != str(right.at[tid, "problem_id"])
        or float(left.at[tid, "y_wrong"]) != float(right.at[tid, "y_wrong"])
        for tid in common
    )
    return {
        "shared_trace_ids": len(common),
        "shared_problem_ids": len(set(left.problem_id) & set(right.problem_id)),
        "cohort227_trace_ids_are_subset_of_pop342": set(right.index) <= set(left.index),
        "shared_traces_with_conflicting_problem_or_label": int(inconsistent),
        "selected_score_differences": score_differences,
        "interpretation": "Overlapping populations and differing scores are separate analyses, not independent replications. No rows or estimates are pooled. The reason scores differ is not established by these exports.",
    }


def run_bundled_scenarios(work: str | Path) -> dict:
    """Evaluate four fixed illustrative comparisons and save separate outputs.

    Returns reports (dict), summary/sensitivity (lists), input_hashes (dict),
    overlap diagnostics, notes, and output paths. Does not select a winner,
    choose an approved effect, or estimate total recruitment from observed yield.
    """
    work = Path(work)
    paths = {name: work / "data" / "development" / filename for name, filename in INPUT_FILES.items()}
    frames = {name: _read_wide(path) for name, path in paths.items()}
    input_hashes = {name: {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                    for name, path in paths.items()}
    reports, summary, sensitivity, converted_frames = {}, [], [], {}
    for population, frame in frames.items():
        for baseline in BASELINE_COLUMNS:
            scenario = f"{population}__{PROBE_COLUMN}__vs__{baseline}"
            converted, provenance = convert_wide(frame, PROBE_COLUMN, baseline, checkpoint="p6144")
            report = plan_sample_size(
                converted, checkpoint="p6144", useful_effects=list(EFFECTS), powers=[.8, .9],
                alpha=.05, alternative="two-sided", variance_multipliers=[1.0, 1.5],
                eligible_fraction=None,
            )
            context = {"scenario": scenario, "population": population, "source_file": paths[population].name,
                       "checkpoint": "p6144", "probe_column": PROBE_COLUMN, "baseline_column": baseline}
            report["wide_export_provenance"] = provenance
            report["scenario"] = context
            report["input_sha256"] = input_hashes[population]["sha256"]
            reports[scenario] = report
            summary.append({**context, "input_traces": len(converted),
                            "input_problems": int(converted.problem_id.nunique()), **report["pilot"],
                            "status": "illustrative_development_only_protocol_not_approved"})
            sensitivity.extend({**context, **row} for row in report["sensitivity"])
            converted_frames[scenario] = converted
    overlap = _overlap(frames)
    out = work / "outputs" / "development_planning"
    result = {
        "purpose": "four_illustrative_development_sample_size_scenarios",
        "protocol_approved": False, "reports": reports, "summary": summary,
        "sensitivity": sensitivity, "input_hashes": input_hashes, "overlap": overlap,
        "notes": [
            "Checkpoint p6144 and P(wrong) orientation come from supplied filenames/description, not independently verified execution records.",
            "Grouped OOF, fold-local neuron selection, and prior method selection are unverified. These computations do not establish leakage-free predictions.",
            "The four comparisons and effects .02/.03/.05/.10 are illustrative sensitivity scenarios, not a selected or approved primary protocol.",
            "Each scenario uses one two-sided difference test at alpha=.05; four simultaneous confirmatory claims would require a separately frozen multiplicity plan.",
            "No total sampled-problem target is computed: eligible_fraction is unknown and is not inferred from these selected development populations.",
            "The input populations overlap and their shared traces have differing fitted scores. They are not pooled or treated as independent replications.",
        ],
        "output_paths": {
            "report": str(out / "development_planning_report.json"),
            "summary": str(out / "development_planning_summary.csv"),
            "sensitivity": str(out / "development_planning_sensitivity.csv"),
            "converted": {name: str(out / f"{name}__correct_oriented.csv") for name in converted_frames},
        },
    }
    # No outputs are written until all four inputs/conversions/plans validate.
    out.mkdir(parents=True, exist_ok=True)
    for scenario, frame in converted_frames.items():
        frame.to_csv(result["output_paths"]["converted"][scenario], index=False)
    Path(result["output_paths"]["report"]).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    pd.DataFrame(summary).to_csv(result["output_paths"]["summary"], index=False)
    pd.DataFrame(sensitivity).to_csv(result["output_paths"]["sensitivity"], index=False)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", required=True, type=Path)
    args = parser.parse_args()
    result = run_bundled_scenarios(args.work)
    print(pd.DataFrame(result["summary"])[["population", "probe_column", "baseline_column", "eligible_mixed_problems", "probe_auroc", "baseline_auroc"]].to_string(index=False))
    print("Illustrative planning only; no confirmation protocol or total recruitment target has been approved.")
