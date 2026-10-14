"""CPU-only regression tests for rollout integrity and input alignment gates."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "probe_cmp"))
import ceiling as C


class RolloutIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.row = dict(prefix_id="ceil/1_0/p512", round=0, problem_id=1,
                        trace_id="1_0", checkpoint="p512", cut_token_idx=512,
                        n=1, prefix_sha="input-A", config_hash="config-A",
                        samples=[dict(sample_idx=0, correct=True, truncated=False, answer="7")])

    def tearDown(self):
        self.temp.cleanup()

    def load(self, rows):
        path = self.directory / "rollouts.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        return C.load_rollouts(path)[0]

    def test_identical_resume_sample_counted_once(self):
        frame = self.load([self.row, self.row])
        self.assertEqual(frame.iloc[0]["n"], 1)

    def test_conflicting_resume_sample_rejected(self):
        changed = copy.deepcopy(self.row)
        changed["samples"][0]["correct"] = False
        with self.assertRaisesRegex(ValueError, "conflicting repeated"):
            self.load([self.row, changed])

    def test_changed_input_hash_rejected(self):
        changed = copy.deepcopy(self.row)
        changed["prefix_sha"] = "input-B"
        with self.assertRaisesRegex(ValueError, "conflicting metadata/input"):
            self.load([self.row, changed])

    def test_changed_budget_or_gold_across_rounds_rejected(self):
        original = {**self.row, "max_tokens": 100, "gold_answer": "7"}
        for change in ({"max_tokens": 50}, {"gold_answer": "8"}):
            changed = {**copy.deepcopy(original), "round": 1, **change}
            with self.assertRaisesRegex(ValueError, "conflicting metadata/input"):
                self.load([original, changed])

    def test_malformed_interior_record_rejected(self):
        path = self.directory / "rollouts.jsonl"
        path.write_text('{"broken":\n' + json.dumps(self.row) + "\n")
        with self.assertRaisesRegex(ValueError, "quarantine explicitly"):
            C.load_rollouts(path)

    def test_wrong_cut_rejected(self):
        self.row["cut_token_idx"] = 511
        with self.assertRaisesRegex(ValueError, "token cut disagree"):
            self.load([self.row])

    def test_conflicting_sample_count_rejected(self):
        self.row["n"] = 8
        with self.assertRaisesRegex(ValueError, "sample count"):
            self.load([self.row])

    def test_empty_file_rejected(self):
        with self.assertRaisesRegex(ValueError, "No complete"):
            self.load([])

    def test_duplicate_cut_under_different_prefix_id_rejected(self):
        duplicate = copy.deepcopy(self.row)
        duplicate["prefix_id"] += "/alias"
        with self.assertRaisesRegex(ValueError, "same source trace/checkpoint"):
            self.load([self.row, duplicate])

    def test_out_of_range_sample_index_rejected(self):
        self.row["samples"][0]["sample_idx"] = 8
        with self.assertRaisesRegex(ValueError, "0 through n-1"):
            self.load([self.row])

    def test_duplicate_json_keys_and_nonfinite_values_rejected(self):
        for raw in ('{"x":1,"x":2}', '{"x":NaN}'):
            with self.assertRaises(ValueError):
                C.strict_json(raw)

    def source(self):
        return pd.DataFrame([dict(trace_id="1_0", problem_id=1, y=0,
                                  is_correct=True, answer="7")]).set_index("trace_id", drop=False)

    def test_known_extraction_rollout_hash_mismatch_rejected(self):
        frame = self.load([self.row])
        meta = pd.DataFrame([dict(trace_id="1_0", pos=512, prefix_sha="input-B",
                                  prefix="thinking", restarts=0)]).set_index(["trace_id", "pos"])
        with self.assertRaisesRegex(ValueError, "input hashes disagree"):
            C.run(frame, self.source(), "unused", str(self.directory / "out"), meta=meta)

    def test_unknown_source_rejected_before_predictions(self):
        frame = self.load([self.row])
        frame["trace_id"] = "missing"
        with self.assertRaisesRegex(ValueError, "unknown source"):
            C.run(frame, self.source(), "unused", str(self.directory / "out"))

    def test_wrong_problem_rejected_before_predictions(self):
        frame = self.load([self.row])
        frame["problem_id"] = 2
        with self.assertRaisesRegex(ValueError, "problem IDs disagree"):
            C.run(frame, self.source(), "unused", str(self.directory / "out"))


if __name__ == "__main__":
    unittest.main()
