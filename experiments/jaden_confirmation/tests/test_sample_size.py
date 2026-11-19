"""Synthetic unit fixtures only: these numbers are not MJJS experiment results."""
import importlib.util
import math
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
from scipy.stats import norm

SPEC = importlib.util.spec_from_file_location(
    "plan_sample_size", Path(__file__).parents[1] / "scripts" / "plan_sample_size.py"
)
planner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(planner)


def fixture():
    # 3 independently sampled problems with 2, 1 and 4 pairs respectively.
    data = [
        ("a", "a1", 1, .8, .2), ("a", "a2", 1, .6, .8), ("a", "a3", 0, .5, .5),
        ("b", "b1", 1, .2, .9), ("b", "b2", 0, .7, .1),
        ("c", "c1", 1, .5, .5), ("c", "c2", 1, .5, .5),
        ("c", "c3", 0, .5, .5), ("c", "c4", 0, .5, .5),
    ]
    frame = pd.DataFrame(data, columns=["problem_id", "trace_id", "y_correct", "probe_score", "baseline_score"])
    frame["checkpoint"] = "6144"
    return frame


class SampleSizeTests(unittest.TestCase):
    def test_known_pair_counts_ties_and_cluster_variance(self):
        clusters = planner.problem_pair_statistics(fixture())
        self.assertEqual(clusters.pairs.tolist(), [2, 1, 4])
        self.assertEqual(clusters.probe_wins.tolist(), [2.0, 0.0, 2.0])
        self.assertEqual(clusters.baseline_wins.tolist(), [1.0, 1.0, 2.0])
        result = planner.clustered_variance(clusters)
        self.assertAlmostEqual(result["probe_auroc"], 4 / 7)
        self.assertAlmostEqual(result["baseline_auroc"], 4 / 7)
        self.assertEqual(result["eligible_mixed_problems"], 3)
        # Three cluster influences: +3/7, -3/7, 0; not seven independent pairs.
        self.assertAlmostEqual(result["cluster_influence_variance"], 9 / 49)
        self.assertAlmostEqual(result["cluster_standard_error"], math.sqrt(3 / 49))

    def test_row_order_invariance_and_problem_grouping(self):
        first = planner.problem_pair_statistics(fixture())
        second = planner.problem_pair_statistics(fixture().sample(frac=1, random_state=17))
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(len(first), 3)
        self.assertEqual(int(first.n_traces.sum()), 9)

    def test_nonmixed_problems_are_excluded(self):
        data = fixture()
        one = data.iloc[[0]].copy()
        one["problem_id"], one["trace_id"] = "single", "single1"
        data = pd.concat([data, one], ignore_index=True)
        report = planner.plan_sample_size(data, useful_effects=[.05], min_development_problems=2)
        self.assertEqual(report["excluded_nonmixed_problems"], 1)
        self.assertEqual(report["pilot"]["eligible_mixed_problems"], 3)

    def test_prespecified_effect_is_required(self):
        with self.assertRaisesRegex(ValueError, "Set useful_effects"):
            planner.plan_sample_size(fixture())

    def test_eligible_fraction_is_not_inferred_from_pilot(self):
        report = planner.plan_sample_size(fixture(), useful_effects=[.05], min_development_problems=2)
        self.assertTrue(all(r["total_sampled_problems_expected_yield_only"] is None for r in report["sensitivity"]))

    def test_formula_and_total_conversion(self):
        result = planner.plan_sample_size(
            fixture(), useful_effects=[.05], powers=[.8], variance_multipliers=[1.0],
            eligible_fraction=.25, min_development_problems=2, alternative="greater",
        )["sensitivity"][0]
        expected = math.ceil((9 / 49) * (norm.ppf(.95) + norm.ppf(.8)) ** 2 / .05 ** 2)
        self.assertEqual(result["required_eligible_mixed_problems"], expected)
        self.assertEqual(result["total_sampled_problems_expected_yield_only"], 4 * expected)
        self.assertGreaterEqual(result["approximate_positive_rejection_power_at_planned_n"], .8)

    def test_two_sided_default_requires_more_than_greater(self):
        kwargs = dict(useful_effects=[.05], powers=[.8], variance_multipliers=[1.], min_development_problems=2)
        two = planner.plan_sample_size(fixture(), **kwargs)
        one = planner.plan_sample_size(fixture(), alternative="greater", **kwargs)
        self.assertEqual(two["alternative"], "two-sided")
        self.assertGreater(two["sensitivity"][0]["required_eligible_mixed_problems"], one["sensitivity"][0]["required_eligible_mixed_problems"])

    def test_lower_effect_and_variance_inflation_increase_sample_size(self):
        rows = planner.plan_sample_size(
            fixture(), useful_effects=[.025, .05], powers=[.8], variance_multipliers=[1., 1.5],
            min_development_problems=2,
        )["sensitivity"]
        self.assertGreater(rows[0]["required_eligible_mixed_problems"], rows[2]["required_eligible_mixed_problems"])
        self.assertGreater(rows[1]["required_eligible_mixed_problems"], rows[0]["required_eligible_mixed_problems"])

    def test_duplicate_rows_and_score_copy_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            planner.problem_pair_statistics(pd.concat([fixture(), fixture().iloc[[0]]]))
        same = fixture()
        same["baseline_score"] = same.probe_score
        with self.assertRaisesRegex(ValueError, "identical"):
            planner.plan_sample_size(same, useful_effects=[.05], min_development_problems=2)

    def test_few_problems_are_not_treated_as_many_pairs(self):
        with self.assertRaisesRegex(ValueError, "Only 3 eligible mixed"):
            planner.plan_sample_size(fixture(), useful_effects=[.05])

    def test_multiple_checkpoints_cannot_be_pooled(self):
        extra = fixture().copy()
        extra["checkpoint"] = "4096"
        with self.assertRaisesRegex(ValueError, "exactly one"):
            planner.problem_pair_statistics(pd.concat([fixture(), extra]))

    def test_zero_cluster_variance_does_not_imply_zero_needed_sample(self):
        df = fixture()
        df["probe_score"] = df.y_correct.astype(float)
        df["baseline_score"] = 1. - df.y_correct
        with self.assertRaisesRegex(ValueError, "zero or unusable"):
            planner.plan_sample_size(df, useful_effects=[.05], min_development_problems=2)

    def test_nonfinite_scores_and_wrong_label_orientation_data_rejected(self):
        bad = fixture()
        bad.loc[0, "probe_score"] = np.inf
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            planner.validate_predictions(bad)
        bad = fixture()
        bad.loc[0, "y_correct"] = 2
        with self.assertRaisesRegex(ValueError, "only 0/1"):
            planner.validate_predictions(bad)


if __name__ == "__main__":
    unittest.main()
