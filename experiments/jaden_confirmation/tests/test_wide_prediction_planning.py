"""Synthetic checks only; fixture values are not MJJS experimental results."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import wide_prediction_planning as wide
from plan_sample_size import problem_pair_statistics, clustered_variance


def fixture():
    # Each problem has one correct (y_wrong=0) and one wrong trace.
    # Probe wins: 1, 0, .5. Baseline wins: .5, 1, 0.
    return pd.DataFrame({
        "problem_id": ["a", "a", "b", "b", "c", "c"],
        "trace_id": ["a0", "a1", "b0", "b1", "c0", "c1"],
        "y_wrong": [0, 1, 0, 1, 0, 1],
        "L20_hgb_topk50": [.2, .8, .7, .3, .4, .4],
        "base_lr_tfidf": [.5, .5, .1, .9, .8, .2],
        "base_lr_len": [.5, .5, .1, .9, .8, .2],
    })


def many_problems(repeats=12):
    parts = []
    for index in range(repeats):
        frame = fixture()
        for column in ("problem_id", "trace_id"):
            frame[column] = str(index) + "_" + frame[column]
        parts.append(frame)
    return pd.concat(parts, ignore_index=True)


class WideConversionTests(unittest.TestCase):
    def test_joint_orientation_and_ties_have_known_pair_scores(self):
        frame = fixture()
        original = frame.copy(deep=True)
        converted, note = wide.convert_wide(frame, "L20_hgb_topk50", "base_lr_tfidf")
        self.assertEqual(converted.y_correct.tolist(), [1, 0, 1, 0, 1, 0])
        np.testing.assert_allclose(converted.probe_score, [.8, .2, .3, .7, .6, .6])
        clusters = problem_pair_statistics(converted)
        self.assertEqual(clusters.probe_wins.tolist(), [1, 0, .5])
        self.assertEqual(clusters.baseline_wins.tolist(), [.5, 1, 0])
        result = clustered_variance(clusters)
        self.assertEqual(result["probe_auroc"], .5)
        self.assertEqual(result["baseline_auroc"], .5)
        self.assertEqual(result["cluster_influence_variance"], .75)
        pd.testing.assert_frame_equal(frame, original)
        self.assertIn("unverified", note["oof_provenance"])
        self.assertIn("not independently verified", note["checkpoint_basis"])

    def test_reversing_labels_and_both_scores_preserves_wrong_oriented_auroc(self):
        source = fixture()
        converted, _ = wide.convert_wide(source, "L20_hgb_topk50", "base_lr_tfidf")
        # Direct original-orientation computation: a positive means WRONG here.
        for column, target in [("L20_hgb_topk50", "probe_wins"), ("base_lr_tfidf", "baseline_wins")]:
            direct = []
            for _, group in source.groupby("problem_id"):
                wrong = group.loc[group.y_wrong == 1, column].iloc[0]
                correct = group.loc[group.y_wrong == 0, column].iloc[0]
                direct.append(float(wrong > correct) + .5 * float(wrong == correct))
            self.assertEqual(problem_pair_statistics(converted)[target].tolist(), direct)

    def test_invalid_labels_rejected(self):
        for value in (2, -1, .5, "not-a-label", np.inf):
            with self.subTest(value=value):
                bad = fixture().astype({"y_wrong": object})
                bad.loc[0, "y_wrong"] = value
                with self.assertRaisesRegex(ValueError, "only 0/1"):
                    wide.convert_wide(bad, "L20_hgb_topk50", "base_lr_tfidf")

    def test_missing_required_data_rejected(self):
        for column in ("problem_id", "trace_id", "y_wrong", "L20_hgb_topk50", "base_lr_tfidf"):
            with self.subTest(column=column):
                missing = fixture().drop(columns=[column])
                with self.assertRaisesRegex(ValueError, "Missing required columns"):
                    wide.convert_wide(missing, "L20_hgb_topk50", "base_lr_tfidf")
                missing = fixture().astype({column: object})
                missing.loc[0, column] = None
                with self.assertRaisesRegex(ValueError, "Missing"):
                    wide.convert_wide(missing, "L20_hgb_topk50", "base_lr_tfidf")

    def test_score_values_must_be_finite_probabilities(self):
        for column in ("L20_hgb_topk50", "base_lr_tfidf"):
            for value in (-.01, 1.01, np.inf, -np.inf, "bad"):
                with self.subTest(column=column, value=value):
                    bad = fixture().astype({column: object})
                    bad.loc[0, column] = value
                    with self.assertRaisesRegex(ValueError, "probabilities"):
                        wide.convert_wide(bad, "L20_hgb_topk50", "base_lr_tfidf")

    def test_duplicate_rows_ids_and_columns_are_rejected(self):
        for across_problems in (False, True):
            bad = fixture()
            bad.loc[1 if not across_problems else 2, "trace_id"] = bad.loc[0, "trace_id"]
            with self.assertRaisesRegex(ValueError, "Duplicate trace_id"):
                wide.convert_wide(bad, "L20_hgb_topk50", "base_lr_tfidf")
        bad = pd.concat([fixture(), fixture()[["base_lr_tfidf"]]], axis=1)
        with self.assertRaisesRegex(ValueError, "Duplicate column"):
            wide.convert_wide(bad, "L20_hgb_topk50", "base_lr_tfidf")

    def test_metadata_cannot_be_used_as_scores(self):
        for column in ("y_wrong", "problem_id", "trace_id", "checkpoint", "fold", "seed"):
            with self.subTest(column=column):
                with self.assertRaisesRegex(ValueError, "metadata cannot"):
                    wide.convert_wide(fixture(), column, "base_lr_tfidf")
        with self.assertRaisesRegex(ValueError, "distinct"):
            wide.convert_wide(fixture(), "base_lr_tfidf", "base_lr_tfidf")

    def test_invalid_ids_and_checkpoint_conflicts_rejected(self):
        for value in ("", " padded ", .5, True):
            bad = fixture()
            bad.loc[0, "problem_id"] = value
            with self.assertRaisesRegex(ValueError, "problem_id"):
                wide.convert_wide(bad, "L20_hgb_topk50", "base_lr_tfidf")
        bad = fixture().assign(checkpoint="p4096")
        with self.assertRaisesRegex(ValueError, "checkpoint values conflict"):
            wide.convert_wide(bad, "L20_hgb_topk50", "base_lr_tfidf")

    def test_exact_baseline_name_and_feature_caveat_preserved(self):
        _, note = wide.convert_wide(fixture(), "L20_hgb_topk50", "base_lr_len")
        self.assertEqual(note["baseline_column"], "base_lr_len")
        self.assertIn("paragraph", note["baseline_note"])
        self.assertIn("not final trace length", note["baseline_note"])
        self.assertIn("not a single decision tree", note["probe_note"])


class ScenarioOutputTests(unittest.TestCase):
    def install_inputs(self, work):
        folder = work / "data" / "development"
        folder.mkdir(parents=True)
        population = many_problems()
        cohort = many_problems(10)
        # Same trace IDs but different fitted predictions: never concatenate them.
        cohort["L20_hgb_topk50"] += .01
        population.to_csv(folder / wide.INPUT_FILES["pop342"], index=False)
        cohort.to_csv(folder / wide.INPUT_FILES["cohort227"], index=False)
        return folder

    def test_four_separate_scenarios_and_no_automatic_total_recruitment(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            folder = self.install_inputs(work)
            before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir()}
            result = wide.run_bundled_scenarios(work)
            self.assertEqual(len(result["reports"]), 4)
            self.assertEqual(len(result["summary"]), 4)
            self.assertEqual(len(result["sensitivity"]), 64)
            self.assertFalse(result["protocol_approved"])
            self.assertEqual(result["overlap"]["shared_trace_ids"], 60)
            self.assertEqual(result["overlap"]["selected_score_differences"]["L20_hgb_topk50"]["shared_traces_with_different_scores"], 60)
            self.assertEqual({row["eligible_mixed_problems"] for row in result["summary"]}, {30, 36})
            self.assertTrue(all(row["total_sampled_problems_expected_yield_only"] is None for row in result["sensitivity"]))
            self.assertEqual({row["useful_auroc_improvement"] for row in result["sensitivity"]}, {.02, .03, .05, .10})
            self.assertEqual(json.loads(Path(result["output_paths"]["report"]).read_text())["input_hashes"], result["input_hashes"])
            for path in result["output_paths"]["converted"].values():
                data = pd.read_csv(path)
                self.assertEqual(set(data.checkpoint), {"p6144"})
            after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir()}
            self.assertEqual(before, after)

    def test_bad_second_population_fails_before_any_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            folder = self.install_inputs(work)
            path = folder / wide.INPUT_FILES["cohort227"]
            data = pd.read_csv(path)
            data.loc[0, "base_lr_len"] = 5.
            data.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "probabilities"):
                wide.run_bundled_scenarios(work)
            self.assertFalse((work / "outputs").exists())

    def test_csv_duplicate_headers_are_not_silently_mangled(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            folder = self.install_inputs(work)
            path = folder / wide.INPUT_FILES["pop342"]
            path.write_text(path.read_text().replace("base_lr_len", "base_lr_tfidf", 1))
            with self.assertRaisesRegex(ValueError, "duplicate CSV column"):
                wide.run_bundled_scenarios(work)
            self.assertFalse((work / "outputs").exists())


if __name__ == "__main__":
    unittest.main()
