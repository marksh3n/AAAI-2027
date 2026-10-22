#!/usr/bin/env python3
"""Plan detection of a useful positive AUROC difference from development OOF data.

The analysis unit is an independently sampled problem. Within each eligible
problem, every correct x incorrect trace pair contributes to the AUROC. The
paired difference is the ratio sum(probe wins - baseline wins) / sum(pairs).
This script estimates the variance of that ratio's problem-cluster influence
values, not a row-independent or pair-independent variance.

This is a normal-approximation planning calculation, not a power guarantee.
It does not train models, select an effect from the observed difference, or
plan an equivalence/noninferiority test. A nonsignificant superiority result
would not establish that the methods are equivalent or that no signal exists.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

REQUIRED = (
    "problem_id", "trace_id", "checkpoint", "y_correct", "probe_score",
    "baseline_score",
)


def validate_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Validate a development OOF table; scores must increase with correctness."""
    missing = [c for c in REQUIRED if c not in predictions.columns]
    if missing:
        raise ValueError(f"Missing prediction columns: {', '.join(missing)}")
    df = predictions[list(REQUIRED)].copy()
    if df.empty or df.isna().any().any():
        raise ValueError("Prediction input must be nonempty with no missing required values.")
    for c in ("problem_id", "trace_id", "checkpoint"):
        df[c] = df[c].astype(str)
        if df[c].str.strip().eq("").any():
            raise ValueError(f"Empty {c} values are not allowed.")
    if df.duplicated(["problem_id", "trace_id", "checkpoint"]).any():
        raise ValueError("Duplicate problem/trace/checkpoint rows are not independent observations.")
    if (df.groupby("trace_id")["problem_id"].nunique() > 1).any():
        raise ValueError("A trace_id maps to more than one problem_id.")
    labels = df["y_correct"].astype(str).str.lower().map(
        {"0": 0, "1": 1, "false": 0, "true": 1, "0.0": 0, "1.0": 1}
    )
    if labels.isna().any():
        raise ValueError("y_correct must contain only 0/1 or boolean labels.")
    df["y_correct"] = labels.astype(int)
    if (df.groupby("trace_id")["y_correct"].nunique() > 1).any():
        raise ValueError("A trace has inconsistent correctness labels across checkpoints.")
    for c in ("probe_score", "baseline_score"):
        df[c] = pd.to_numeric(df[c], errors="raise")
        if not np.isfinite(df[c].to_numpy(dtype=float)).all():
            raise ValueError(f"{c} contains nonfinite values.")
    return df


def problem_pair_statistics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Return one paired contribution per mixed problem at ONE checkpoint.

    Ties count as half a win. Problems with only one outcome are excluded.
    Rows are sorted so output does not depend on input order.
    """
    df = validate_predictions(predictions)
    if df["checkpoint"].nunique() != 1:
        raise ValueError("Select exactly one prespecified checkpoint before planning.")
    rows: list[dict[str, Any]] = []
    for pid, group in df.groupby("problem_id", sort=True):
        good = group[group.y_correct == 1]
        bad = group[group.y_correct == 0]
        if good.empty or bad.empty:
            continue
        wins: dict[str, float] = {}
        for name in ("probe_score", "baseline_score"):
            negatives = bad[name].to_numpy(dtype=float)
            # One trace at a time avoids constructing a large all-pairs matrix.
            wins[name] = sum(
                float(np.count_nonzero(value > negatives))
                + 0.5 * float(np.count_nonzero(value == negatives))
                for value in good[name].to_numpy(dtype=float)
            )
        rows.append({
            "problem_id": pid,
            "n_traces": int(len(group)),
            "n_correct": int(len(good)),
            "n_wrong": int(len(bad)),
            "pairs": int(len(good) * len(bad)),
            "probe_wins": wins["probe_score"],
            "baseline_wins": wins["baseline_score"],
            "difference_wins": wins["probe_score"] - wins["baseline_score"],
        })
    return pd.DataFrame(rows, columns=[
        "problem_id", "n_traces", "n_correct", "n_wrong", "pairs",
        "probe_wins", "baseline_wins", "difference_wins",
    ])


def clustered_variance(clusters: pd.DataFrame) -> dict[str, float | int]:
    """Estimate Var(sqrt(G) * paired AUROC difference) by cluster influence.

    For problem g, a_g is its paired win difference and b_g its pair count.
    theta_hat = sum(a_g)/sum(b_g), psi_g = (a_g - theta_hat*b_g)/mean(b_g).
    sample_variance(psi_g) / G estimates the variance of theta_hat.
    """
    g = len(clusters)
    if g < 2:
        raise ValueError("At least two eligible mixed problems are needed to estimate cluster variance.")
    pairs = clusters["pairs"].to_numpy(dtype=float)
    differences = clusters["difference_wins"].to_numpy(dtype=float)
    theta = float(differences.sum() / pairs.sum())
    influence = (differences - theta * pairs) / pairs.mean()
    variance = float(np.var(influence, ddof=1))
    return {
        "eligible_mixed_problems": g,
        "eligible_traces": int(clusters["n_traces"].sum()),
        "correct_wrong_pairs": int(pairs.sum()),
        "probe_auroc": float(clusters["probe_wins"].sum() / pairs.sum()),
        "baseline_auroc": float(clusters["baseline_wins"].sum() / pairs.sum()),
        "observed_paired_delta_diagnostic_only": theta,
        "cluster_influence_variance": variance,
        "cluster_standard_error": math.sqrt(variance / g),
        "pair_weight_effective_problem_count_diagnostic": float(pairs.sum() ** 2 / np.square(pairs).sum()),
        "largest_problem_pair_share": float(pairs.max() / pairs.sum()),
    }


def plan_sample_size(
    predictions: pd.DataFrame | str | Path,
    checkpoint: str = "6144",
    useful_effects: list[float] | tuple[float, ...] | None = None,
    powers: tuple[float, ...] | list[float] = (0.80, 0.90),
    alpha: float = 0.05,
    alternative: str = "two-sided",
    eligible_fraction: float | None = None,
    variance_multipliers: tuple[float, ...] | list[float] = (1.0, 1.5),
    min_development_problems: int = 30,
    min_confirmation_problems: int = 30,
    max_eligible_problems: int = 1_000_000,
) -> dict[str, Any]:
    """Return a JSON-ready sample-size sensitivity report, without writing files.

    ``useful_effects`` must be positive, scientifically useful absolute AUROC
    improvements chosen independently of these observed development estimates.
    The default checkpoint 6144 is a DRAFT, not a claim of mentor approval.
    The default two-sided test uses z_(1-alpha/2); alternative='greater' uses
    z_(1-alpha). The power approximation targets the positive rejection tail.
    ``eligible_fraction`` is an explicitly supplied expected fraction of newly
    sampled problems that reach this checkpoint with both outcomes and satisfy
    all exclusions. It must account for the planned generation/sample policy.
    """
    if useful_effects is None or len(useful_effects) == 0:
        raise ValueError("Set useful_effects explicitly after agreeing on a meaningful AUROC improvement; none is approved by this script.")
    if not powers or not variance_multipliers:
        raise ValueError("At least one target power and one variance multiplier are required.")
    if not math.isfinite(alpha) or not 0 < alpha < 0.5:
        raise ValueError("alpha must be between 0 and 0.5.")
    if alternative not in ("two-sided", "greater"):
        raise ValueError("alternative must be 'two-sided' or 'greater'.")
    if any(not math.isfinite(x) or not 0 < x < 1 for x in useful_effects):
        raise ValueError("Useful effects must be finite, positive absolute AUROC differences below 1.")
    if any(not math.isfinite(x) or not 0.5 < x < 1 for x in powers):
        raise ValueError("Target powers must be between 0.5 and 1.")
    if any(not math.isfinite(x) or x < 1 for x in variance_multipliers):
        raise ValueError("Variance sensitivity multipliers must be finite and at least 1.")
    if eligible_fraction is not None and (not math.isfinite(eligible_fraction) or not 0 < eligible_fraction <= 1):
        raise ValueError("eligible_fraction must be in (0, 1], or None to leave total recruitment unknown.")
    if min_development_problems < 2 or min_confirmation_problems < 2:
        raise ValueError("Minimum problem counts must be at least 2.")
    if max_eligible_problems < min_confirmation_problems:
        raise ValueError("The planning bound must exceed the confirmation minimum.")
    if isinstance(predictions, (str, Path)):
        raw = pd.read_csv(predictions, dtype={"problem_id": str, "trace_id": str, "checkpoint": str})
    else:
        raw = predictions
    all_rows = validate_predictions(raw)
    selected = all_rows[all_rows.checkpoint == str(checkpoint)]
    if selected.empty:
        raise ValueError(f"No development OOF rows at checkpoint {checkpoint!r}. No sample-size calculation was made.")
    if np.array_equal(selected.probe_score.to_numpy(), selected.baseline_score.to_numpy()):
        raise ValueError("Probe and baseline scores are identical on every selected row; check that distinct prediction columns were supplied.")
    clusters = problem_pair_statistics(selected)
    if len(clusters) < min_development_problems:
        raise ValueError(
            f"Only {len(clusters)} eligible mixed development problems; minimum is {min_development_problems}. "
            "Cluster-variance planning is too unstable under the selected minimum. Obtain more development predictions; "
            "do not treat rows or correct/wrong pairs as independent problems."
        )
    summary = clustered_variance(clusters)
    variance = float(summary["cluster_influence_variance"])
    if variance <= 1e-15 or not math.isfinite(variance):
        raise ValueError("The pilot has zero or unusable cluster variance; it cannot justify a finite confirmation sample size.")
    warnings = [
        "Checkpoint 6144 is a draft default; freeze the primary checkpoint, layer, model, baseline and exclusions before confirmation.",
        "The input must contain development-only out-of-fold predictions grouped by problem. This script cannot verify how predictions were produced.",
        "The influence calculation conditions on the supplied fitted predictions; overlapping CV training sets can induce cross-problem dependence, and retraining/method-selection variability is not fully modeled. Prefer a frozen model on independent held-out confirmation problems, or validate planning with prospective pilot replications.",
        "The observed development delta is diagnostic only; the sample-size calculation uses your explicitly supplied useful effects.",
        "The normal approximation assumes independent future problems from a comparable eligible population and similar traces-per-problem and pair-weight distribution.",
        "Pilot variance and eligible yield may change across models, generation settings, truncation rules, checkpoints or populations; multipliers are sensitivity assumptions, not confidence bounds.",
        "The default is one two-sided difference test at alpha=0.05 (matching a 95% interval); the optional 'greater' alternative is a one-sided superiority test. The test direction must be agreed before confirmation. Neither design establishes equivalence/noninferiority from failure to reject.",
        "No multiplicity correction is included. Choosing a checkpoint or method after examining confirmation results invalidates the single-test plan.",
    ]
    if len(clusters) < 30:
        warnings.append("Fewer than 30 development clusters were explicitly allowed; estimates are particularly unstable and should remain exploratory.")
    if summary["largest_problem_pair_share"] > 0.10:
        warnings.append("One problem contributes over 10% of all pairs. Pair-weight concentration may make normal-approximation planning unreliable.")
    if eligible_fraction is None:
        warnings.append("Total sampled problems are not computed because eligible_fraction was not supplied. The observed development fraction is not substituted automatically.")
    else:
        warnings.append("Total sampled problems use ceil(required eligible / eligible_fraction): an expected-yield planning conversion, not a probability guarantee of reaching the eligible target.")
    sensitivity = []
    critical_quantile = float(norm.ppf(1 - alpha / 2 if alternative == "two-sided" else 1 - alpha))
    for effect in useful_effects:
        for power in powers:
            for multiplier in variance_multipliers:
                planned_variance = variance * multiplier
                raw_n = math.ceil(planned_variance * (critical_quantile + norm.ppf(power)) ** 2 / effect ** 2)
                needed = max(min_confirmation_problems, raw_n)
                bounded = needed <= max_eligible_problems
                g = needed if bounded else None
                total = math.ceil(g / eligible_fraction) if g is not None and eligible_fraction is not None else None
                sensitivity.append({
                    "useful_auroc_improvement": float(effect),
                    "target_power": float(power),
                    "alpha": float(alpha),
                    "alternative": alternative,
                    "variance_multiplier": float(multiplier),
                    "normal_approximation_eligible_problems": int(raw_n),
                    "minimum_confirmation_problem_floor": int(min_confirmation_problems),
                    "required_eligible_mixed_problems": g,
                    "total_sampled_problems_expected_yield_only": total,
                    "eligible_fraction_assumption": eligible_fraction,
                    "approximate_positive_rejection_power_at_planned_n": float(norm.cdf(effect * math.sqrt(g / planned_variance) - critical_quantile)) if g is not None else None,
                    "status": "planning_estimate_only" if bounded else "exceeds_planning_bound",
                })
    return {
        "purpose": "development_only_sample_size_sensitivity",
        "design": "single_two_sided_difference_test" if alternative == "two-sided" else "single_one_sided_superiority_test",
        "alternative": alternative,
        "score_orientation": "higher predicts correct",
        "primary_checkpoint": str(checkpoint),
        "checkpoint_and_effects_require_prespecification": True,
        "selected_input_problems": int(selected.problem_id.nunique()),
        "excluded_nonmixed_problems": int(selected.problem_id.nunique() - len(clusters)),
        "pilot": summary,
        "variance_formula": "sample_var((difference_wins_g - delta_hat * pairs_g) / mean_pairs); SE(delta_hat)=sqrt(variance/G)",
        "planning_formula": "G=ceil(variance * variance_multiplier * (z_critical+z_power)^2 / useful_effect^2), subject to the declared minimum; z_critical=z_(1-alpha/2) for two-sided, z_(1-alpha) for greater",
        "power_note": "Power targets the positive rejection tail. For a two-sided difference test this omits the opposite tail and is a conservative normal approximation to total rejection power.",
        "sensitivity": sensitivity,
        "warnings": warnings,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", type=Path, required=True, help="Development OOF CSV; never use the confirmation holdout to plan its own size.")
    ap.add_argument("--checkpoint", default="6144", help="One primary checkpoint. Default 6144 is DRAFT, not approved.")
    ap.add_argument("--useful-effects", type=float, nargs="+", required=True, help="Explicit scientifically useful AUROC improvements; do not substitute the observed pilot effect.")
    ap.add_argument("--powers", type=float, nargs="+", default=[0.80, 0.90])
    ap.add_argument("--alpha", type=float, default=0.05, help="Alpha for one prespecified difference test; two-sided by default.")
    ap.add_argument("--alternative", choices=("two-sided", "greater"), default="two-sided")
    ap.add_argument("--eligible-fraction", type=float, default=None, help="Expected eligible mixed fraction of newly sampled problems; omit to leave total recruitment uncomputed.")
    ap.add_argument("--variance-multipliers", type=float, nargs="+", default=[1.0, 1.5], help="Exploratory variance sensitivity assumptions, not confidence bounds.")
    ap.add_argument("--min-development-problems", type=int, default=30)
    ap.add_argument("--min-confirmation-problems", type=int, default=30)
    ap.add_argument("--out-dir", type=Path, required=True, help="A separate output directory, not an input cache.")
    args = ap.parse_args()
    try:
        report = plan_sample_size(
            args.predictions, checkpoint=args.checkpoint, useful_effects=args.useful_effects,
            powers=args.powers, alpha=args.alpha, alternative=args.alternative, eligible_fraction=args.eligible_fraction,
            variance_multipliers=args.variance_multipliers,
            min_development_problems=args.min_development_problems,
            min_confirmation_problems=args.min_confirmation_problems,
        )
    except (ValueError, TypeError, KeyError) as exc:
        ap.error(str(exc))
    outputs = [args.out_dir / "sample_size_plan.json", args.out_dir / "sample_size_sensitivity.csv"]
    if any(p.resolve() == args.predictions.resolve() for p in outputs):
        ap.error("An output path would overwrite the input predictions.")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    outputs[0].write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    pd.DataFrame(report["sensitivity"]).to_csv(outputs[1], index=False)
    print(f"Read {report['pilot']['eligible_mixed_problems']} eligible development problems at checkpoint {args.checkpoint}.")
    print("Planning sensitivity only; effects, primary analysis and recruitment require agreement before confirmation.")
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
