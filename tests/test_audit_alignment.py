import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from audit_alignment import bos_candidates, check_rollout_manifest, checkpoint_cut, input_hash, load_build_prompt, matching_bos, register_key, strict_loads
from cache import sha_obj


class AlignmentAuditTests(unittest.TestCase):
    def test_duplicate_prefix_round_rejected(self):
        seen = set()
        register_key(seen, ("p", 0))
        register_key(seen, ("p", 1))
        with self.assertRaises(ValueError):
            register_key(seen, ("p", 0))

    def test_changed_cleaned_file_rejected_by_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            rollouts = Path(directory) / "clean.jsonl"
            manifest = Path(directory) / "manifest.json"
            rollouts.write_text('{}\n')
            manifest.write_text(json.dumps({"cleaned_rollouts_sha256": "incorrect"}))
            with self.assertRaises(ValueError):
                check_rollout_manifest(rollouts, manifest)

    def test_hash_matches_repository_format(self):
        self.assertEqual(input_hash([1, 2], [3, 4]), sha_obj([[1, 2], [3, 4]]))

    def test_one_vs_two_bos_and_changed_prefix(self):
        stored = input_hash([1, 1, 2], [3, 4])
        self.assertEqual(matching_bos([1, 2], [3, 4], 1, stored), 2)
        self.assertIsNone(matching_bos([1, 2], [3, 5], 1, stored))
        self.assertEqual(matching_bos([1, 2], [3, 4], 1, input_hash([1, 2], [3, 4])), 1)

    def test_reject_template_without_exact_one_bos(self):
        for ids in ([2, 3], [1, 1, 2], []):
            with self.assertRaises(ValueError):
                bos_candidates(ids, 1)

    def test_checkpoint_mismatch_and_t0(self):
        with self.assertRaises(ValueError):
            checkpoint_cut(dict(checkpoint="p512", cut_token_idx=511, trace_id="1_0"))
        with self.assertRaises(ValueError):
            checkpoint_cut(dict(checkpoint="t0", cut_token_idx=0, trace_id="1_0"))
        self.assertEqual(checkpoint_cut(dict(checkpoint="t0", cut_token_idx=0, trace_id=None)), 0)

    def test_strict_json(self):
        for raw in ['{"x":1,"x":2}', '{"x":NaN}', 'broken']:
            with self.assertRaises(ValueError):
                strict_loads(raw)

    def test_loads_prompt_without_model_imports(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "generate.py"
            source.write_text('import package_that_does_not_exist\nINSTRUCTION="i"\nSTRUCTURED="s"\ndef build_prompt(tokenizer, question, structured=False):\n    return question + (STRUCTURED if structured else INSTRUCTION)\n')
            build = load_build_prompt(source)
            self.assertEqual(build(None, "q", True), "qs")


if __name__ == "__main__":
    unittest.main()
