"""CPU-only tests for the untouched-MATH confirmation pool preparation.

Run from the kit root: python -m unittest discover -s tests -v
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from prepare_problems import (  # noqa: E402
    extract_boxed_answer,
    normalize_problem,
    prepare_pool,
    source_keys,
)


class NormalizationTests(unittest.TestCase):
    def test_nfkc_and_whitespace_only(self):
        self.assertEqual(normalize_problem("  Ａ\t+\n x  =  ３ \u00a0"), "A + x = 3")
        self.assertEqual(normalize_problem("Find\u00a0\u00a0x.\n"), "Find x.")

    def test_mathematical_case_is_preserved(self):
        self.assertNotEqual(normalize_problem("Find X."), normalize_problem("Find x."))
        self.assertNotEqual(normalize_problem(r"Find \Gamma."), normalize_problem(r"Find \gamma."))

    def test_no_punctuation_or_mathematical_rewriting(self):
        self.assertNotEqual(normalize_problem("Find x+y."), normalize_problem("Find x + y."))
        self.assertNotEqual(normalize_problem(r"Find $x$."), normalize_problem("Find x."))


class BoxedAnswerTests(unittest.TestCase):
    def test_plain_and_nested_braces(self):
        self.assertEqual(extract_boxed_answer(r"Therefore \boxed{42}."), "42")
        self.assertEqual(
            extract_boxed_answer(r"Thus \boxed{\frac{a_{1}}{b^{2}}}."),
            r"\frac{a_{1}}{b^{2}}",
        )

    def test_escaped_braces_are_literal_answer_content(self):
        self.assertEqual(extract_boxed_answer(r"Answer: \boxed{\{1,2\}}."), r"\{1,2\}")
        self.assertEqual(extract_boxed_answer(r"Answer: \boxed{\}}."), r"\}")

    def test_absent_or_malformed_box_does_not_invent_gold(self):
        for text in ("No answer given", r"\boxed{42", r"\boxed{\frac{1}{2}", r"\boxed 42"):
            with self.subTest(text=text):
                self.assertIsNone(extract_boxed_answer(text))


class SourceKeyTests(unittest.TestCase):
    def test_unqualified_numbers_do_not_identify_cross_dataset_problems(self):
        for row in ({"id": 7}, {"problem_id": 7}, {"id": "7", "problem_id": "7"}):
            with self.subTest(row=row):
                self.assertEqual(source_keys(row), set())

    def test_explicit_qualified_id(self):
        self.assertIn(("example/MATH", "007"), source_keys({"source_dataset": "example/MATH", "source_id": "007"}))
        self.assertEqual(source_keys({"source_id": "007"}), set())
        self.assertEqual(source_keys({"source_dataset": "example/MATH"}), set())

    def test_canonical_math_unique_ids_keep_the_split(self):
        self.assertIn(("hendrycks/MATH", "train/algebra/123.json"), source_keys({"unique_id": "train/algebra/123.json"}))
        self.assertIn(("hendrycks/MATH", "test/algebra/123.json"), source_keys({"unique_id": "test/algebra/123.json"}))
        self.assertTrue(source_keys({"unique_id": "train/algebra/123.json"}).isdisjoint(source_keys({"unique_id": "test/algebra/123.json"})))


class PreparePoolTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def write_jsonl(self, name, rows):
        path = self.root / name
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
        return path

    def run_pool(self, rows, math500=None, inspected=(), name="out", seed=17):
        math_path = self.write_jsonl(name + "-math.jsonl", rows)
        if math500 is None:
            math500 = [{"problem": "Unrelated valid MATH500 exclusion fixture"}]
        math500_path = self.write_jsonl(name + "-math500.jsonl", math500)
        inspected_paths = []
        if inspected:
            inspected_paths.append(self.write_jsonl(name + "-inspected.jsonl", inspected))
        out = self.root / name
        report = prepare_pool(math_path, math500_path, inspected_paths, out, seed)
        candidates = [json.loads(line) for line in (out / "candidate_pool.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        return report, candidates, out

    def test_filters_overlap_duplicates_levels_and_missing_gold(self):
        rows = [
            {"problem": "Keep A", "level": 3, "answer": "1", "problem_id": 1},
            {"problem": "Keep B", "level": "Level 5", "gold": "2", "problem_id": 2},
            {"problem": "Excluded by MATH500", "level": 4, "answer": "3", "problem_id": 3},
            {"problem": "Excluded by inspected traces", "level": 3, "answer": "4", "problem_id": 4},
            {"problem": "Duplicate problem", "level": 4, "answer": "5", "problem_id": 5},
            {"problem": "  Duplicate\tproblem ", "level": 4, "answer": "5", "problem_id": 6},
            {"problem": "Too easy", "level": 2, "answer": "7", "problem_id": 7},
            {"problem": "Missing gold", "level": 4, "problem_id": 8},
            {"problem": "Malformed gold", "level": 3, "solution": r"\boxed{9", "problem_id": 9},
            {"problem": "Boxed solution gold", "level": "Level 4", "solution": r"Hence \boxed{\frac{1}{2}}.", "problem_id": 10},
        ]
        report, candidates, out = self.run_pool(
            rows,
            math500=[{"problem": "  Ｅxcluded by\nMATH500  ", "problem_id": 999}],
            inspected=[{"problem": "Excluded  by inspected\ttraces", "problem_id": 998}],
        )
        self.assertEqual({normalize_problem(row["problem"]) for row in candidates}, {"Keep A", "Keep B", "Duplicate problem", "Boxed solution gold"})
        self.assertEqual(len(candidates), 4)
        self.assertIsInstance(report, dict)
        self.assertEqual(json.loads((out / "preparation_report.json").read_text()), report)
        self.assertTrue((out / "exclusions.csv").is_file())
        self.assertGreater((out / "exclusions.csv").stat().st_size, 0)

    def test_bare_numeric_id_collision_does_not_exclude_other_problem(self):
        _, candidates, _ = self.run_pool(
            [{"problem": "Untouched different question", "level": 3, "gold": "7", "problem_id": 7}],
            math500=[{"problem": "A MATH500 question", "problem_id": 7}],
            inspected=[{"problem": "An inspected question", "id": 7}],
        )
        self.assertEqual(len(candidates), 1)

    def test_qualified_source_id_excludes_without_text_match(self):
        rows = [{"problem": "Original wording", "level": 3, "answer": "1", "source_dataset": "example/MATH", "source_id": "abc"}]
        _, candidates, _ = self.run_pool(rows, inspected=[{"source_dataset": "example/MATH", "source_id": "abc"}])
        self.assertEqual(candidates, [])

    def test_canonical_unique_id_excludes_but_preserves_other_split(self):
        rows = [
            {"problem": "Train item", "level": 3, "answer": "1", "unique_id": "train/algebra/123.json"},
            {"problem": "Test item", "level": 3, "answer": "2", "unique_id": "test/algebra/123.json"},
        ]
        _, candidates, _ = self.run_pool(rows, math500=[{"unique_id": "test/algebra/123.json"}])
        self.assertEqual([row["problem"] for row in candidates], ["Train item"])

    def test_seeded_order_is_repeatable_and_input_order_independent(self):
        rows = [{"problem": f"Untouched problem {i}", "level": 3 + i % 3, "answer": str(i), "unique_id": f"train/algebra/{i}.json"} for i in range(24)]
        _, a, _ = self.run_pool(rows, name="a", seed=37)
        _, b, _ = self.run_pool(rows, name="b", seed=37)
        _, c, _ = self.run_pool(list(reversed(rows)), name="c", seed=37)
        self.assertEqual([r["problem"] for r in a], [r["problem"] for r in b])
        self.assertEqual([r["problem"] for r in a], [r["problem"] for r in c])
        self.assertEqual(len(a), 24)

    def test_missing_inspected_file_is_a_hard_failure(self):
        math_path = self.write_jsonl("math.jsonl", [{"problem": "Candidate", "level": 3, "answer": "1"}])
        math500_path = self.write_jsonl("math500.jsonl", [{"problem": "Unrelated MATH500 question"}])
        with self.assertRaises(FileNotFoundError):
            prepare_pool(math_path, math500_path, [self.root / "missing-inspected.jsonl"], self.root / "out", 1)

    def test_empty_provided_exclusion_file_is_a_hard_failure(self):
        math_path = self.write_jsonl("math.jsonl", [{"problem": "Candidate", "level": 3, "answer": "1"}])
        valid_exclusion = self.write_jsonl("valid-exclusion.jsonl", [{"problem": "Unrelated exclusion"}])
        empty_exclusion = self.write_jsonl("empty-exclusion.jsonl", [])
        for math500_path, inspected_paths in ((empty_exclusion, []), (valid_exclusion, [empty_exclusion])):
            with self.subTest(math500_path=math500_path, inspected_paths=inspected_paths):
                with self.assertRaisesRegex(ValueError, "empty"):
                    prepare_pool(math_path, math500_path, inspected_paths, self.root / "empty-exclusion-out", 1)
        self.assertFalse((self.root / "empty-exclusion-out" / "candidate_pool.jsonl").exists())

    def test_conflicting_duplicate_gold_and_level_are_excluded(self):
        rows = [
            {"problem": "Gold conflict", "level": 3, "answer": "1"},
            {"problem": " Gold\tconflict ", "level": 3, "answer": "2"},
            {"problem": "Level conflict", "level": 3, "answer": "3"},
            {"problem": "Level  conflict", "level": 4, "answer": "3"},
            {"problem": "Unambiguous candidate", "level": 5, "answer": "4"},
        ]
        _, candidates, out = self.run_pool(rows)
        self.assertEqual([r["problem"] for r in candidates], ["Unambiguous candidate"])
        with (out / "exclusions.csv").open(newline="", encoding="utf-8") as handle:
            reasons = [row["reason"] for row in csv.DictReader(handle)]
        self.assertEqual(reasons.count("duplicate_gold_conflict"), 2)
        self.assertEqual(reasons.count("duplicate_level_conflict"), 2)

    def test_unusable_exclusion_record_is_a_hard_failure(self):
        candidate = [{"problem": "Candidate", "level": 3, "answer": "1"}]
        for exclusion in ({}, {"problem": "  "}, {"problem": "", "problem_id": 7}, {"id": "7"}):
            with self.subTest(exclusion=exclusion):
                with self.assertRaises(ValueError):
                    self.run_pool(candidate, inspected=[exclusion], name="bad-exclusion")


if __name__ == "__main__":
    unittest.main()
