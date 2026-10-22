"""CPU-only guard tests; all tokens/vectors are synthetic, no model download."""
import copy
from contextlib import nullcontext
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("exact_token_run", Path(__file__).parents[1] / "scripts" / "exact_token_run.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class Tokenizer:
    bos_token_id = 1
    eos_token_id = 2
    special_tokens_map = {"bos_token": "BOS", "eos_token": "EOS"}
    chat_template = "synthetic_one_bos"
    backend_tokenizer = None

    def get_vocab(self):
        return {"BOS": 1, "EOS": 2, "a": 107}

    def apply_chat_template(self, messages, **kwargs):
        return "^" + messages[0]["content"] + "<think>\n"

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        assert add_special_tokens is False
        ids = [1 if c == "^" else ord(c) + 10 for c in text]
        row = {"input_ids": ids}
        if return_offsets_mapping:
            row["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return row

    def decode(self, ids, **kwargs):
        return "".join("^" if i == 1 else "[EOS]" if i == 2 else chr(i - 10) for i in ids)


def plan():
    return {"schema_version": 1, "phase": "pilot", "model_id": runner.MODEL,
            "model_revision": "a" * 40, "tokenizer_revision": "b" * 40,
            "selected_problem_ids": ["p1"],
            "generation": {"samples_per_problem": 2, "temperature": 0.7, "top_p": 0.95,
                           "max_new_tokens": 8192, "base_seed": 0},
            "extraction": {"checkpoints": [512, 2048, 4096, 6144, 8192]}}


class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.plan = plan()
        self.problem = {"problem_id": "p1", "problem": "What is 1+1?", "gold": "2"}
        self.plan_path = self.root / "plan.json"
        self.problems_path = self.root / "problems.jsonl"
        self.plan_path.write_text(json.dumps(self.plan))
        self.problems_path.write_text(json.dumps(self.problem) + "\n")
        _, _, self.manifest = runner.load_inputs(self.plan_path, self.problems_path)
        self.tokenizer = Tokenizer()
        self.runtime = {"eos_token_id": 2, "model_revision": "a" * 40,
                        "tokenizer_revision": "b" * 40, "tokenizer_sha256": runner.tokenizer_identity(self.tokenizer)}

    def record(self, text="a" * 510 + "</think>answer\\boxed{2}"):
        prompt_text, prompt = runner.exact_prompt(self.tokenizer, self.problem["problem"])
        completion = self.tokenizer(text)["input_ids"] + [2]
        return runner.make_record(self.problem, 0, self.plan, self.manifest, self.runtime,
                                  prompt_text, prompt, completion, self.tokenizer)

    def test_strict_plan_and_pinned_revisions(self):
        runner.validate_plan(self.plan)
        for field, value in [("model_revision", "main"), ("tokenizer_revision", "a" * 39),
                             ("selected_problem_ids", ["p1", "p1"]), ("phase", "automatic")]:
            bad = copy.deepcopy(self.plan)
            bad[field] = value
            with self.assertRaises(ValueError):
                runner.validate_plan(bad)
        for key, value in [("top_p", 1.1), ("samples_per_problem", True), ("temperature", float("nan"))]:
            bad = copy.deepcopy(self.plan)
            bad["generation"][key] = value
            with self.assertRaises(ValueError):
                runner.validate_plan(bad)

    def test_candidate_gold_and_plan_changes_block_resume(self):
        output = self.root / "run"
        runner.bind_output(output, self.manifest, create=True)
        runner.bind_output(output, self.manifest)
        changed = dict(self.problem, gold="3")
        self.problems_path.write_text(json.dumps(changed) + "\n")
        _, _, manifest = runner.load_inputs(self.plan_path, self.problems_path)
        with self.assertRaisesRegex(ValueError, "Resume refused"):
            runner.bind_output(output, manifest, create=True)
        self.problems_path.write_text((json.dumps(self.problem) + "\n") * 2)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            runner.load_inputs(self.plan_path, self.problems_path)

    def test_exact_one_bos(self):
        _, ids = runner.exact_prompt(self.tokenizer, "Q")
        self.assertEqual(ids[0], 1)
        self.assertNotEqual(ids[1], 1)
        with patch.object(self.tokenizer, "apply_chat_template", return_value="^^hello"):
            with self.assertRaisesRegex(ValueError, "one leading BOS"):
                runner.exact_prompt(self.tokenizer, "Q")
        with patch.object(self.tokenizer, "apply_chat_template", return_value="hello"):
            with self.assertRaisesRegex(ValueError, "one leading BOS"):
                runner.exact_prompt(self.tokenizer, "Q")

    def test_actual_completion_and_eos_saved_without_double_count(self):
        row = self.record()
        self.assertEqual(len(row["completion_token_ids"]), len(row["generated_token_ids"]) + 1)
        self.assertEqual(row["eos_completion_index"], len(row["generated_token_ids"]))
        self.assertEqual(row["finish_reason"], "eos")
        self.assertEqual(row["runtime"]["model_revision"], "a" * 40)
        runner.validate_record(row, self.problem, 0, self.plan, self.manifest, self.runtime, self.tokenizer)
        row["generated_token_ids"][0] += 1
        with self.assertRaisesRegex(ValueError, "hash changed"):
            runner.validate_record(row, self.problem, 0, self.plan, self.manifest, self.runtime, self.tokenizer)

    def test_legacy_and_wrong_runtime_rejected(self):
        with self.assertRaises(ValueError):
            runner.validate_record({"text": "answer"}, self.problem, 0, self.plan,
                                   self.manifest, self.runtime, self.tokenizer)
        row = self.record()
        with self.assertRaisesRegex(ValueError, "runtime mismatch"):
            runner.validate_record(row, self.problem, 0, self.plan, self.manifest,
                                   dict(self.runtime, model_revision="c" * 40), self.tokenizer)

    def test_short_capped_or_interior_eos_rejected(self):
        text, prompt = runner.exact_prompt(self.tokenizer, self.problem["problem"])
        for completion in ([107, 2, 107], [107, 107]):
            with self.assertRaises(ValueError):
                runner.make_record(self.problem, 0, self.plan, self.manifest, self.runtime,
                                   text, prompt, completion, self.tokenizer)

    def test_marker_boundaries_and_prefix_ids(self):
        record = self.record("a" * 512 + "</think>" + "a" * 2000)
        boundary = runner.answer_boundary(self.tokenizer, record)
        self.assertEqual(boundary["first_marker_token_index"], 512)
        rows = runner.checkpoint_metadata(record, [512, 2048, 4096], boundary)
        self.assertEqual([r["checkpoint"] for r in rows], [512, 2048])
        self.assertEqual(rows[0]["answer_stage"], "before_answer_marker")
        self.assertEqual(rows[1]["answer_stage"], "at_or_after_answer_marker")
        self.assertEqual(rows[0]["generation_token_index"], 511)
        self.assertEqual(rows[0]["model_input_token_index"], len(record["prompt_token_ids"]) + 511)
        self.assertEqual(rows[0]["prefix_sha"], runner.sha_obj([record["prompt_token_ids"], record["generated_token_ids"][:512]]))
        unknown = runner.answer_boundary(self.tokenizer, self.record("a" * 520))
        self.assertEqual(unknown["status"], "unknown")
        self.assertEqual(runner.checkpoint_metadata(record, [512], unknown)[0]["answer_stage"], "unknown")

    def test_roundtrip_drift_does_not_reconstruct_prefixes(self):
        record = self.record()
        # Same bytes with a changed hypothetical token segmentation must not be
        # relabeled using offsets from a different token sequence.
        record["generated_token_ids"] = [777] + record["generated_token_ids"][1:]
        boundary = runner.answer_boundary(self.tokenizer, record)
        self.assertEqual(boundary, {"status": "unknown", "reason": "text_offsets_do_not_match_recorded_ids"})
        rows = runner.checkpoint_metadata(record, [512], boundary)
        self.assertEqual(rows[0]["prefix_gen_token_ids"][0], 777)

    def test_seed_and_trace_identity_independent_of_iteration_order(self):
        key = runner.sample_key("p1", 0)
        seed = runner.sample_seed(self.plan, "p1", 0)
        reordered = copy.deepcopy(self.plan)
        reordered["selected_problem_ids"] = ["p2", "p1"]
        self.assertEqual(seed, runner.sample_seed(reordered, "p1", 0))
        self.assertNotEqual(key, runner.sample_key("p1", 1))
        self.assertNotEqual(seed, runner.sample_seed(self.plan, "p1", 1))

    def test_confirmation_requires_explicit_flag_before_gpu_import(self):
        confirmation = copy.deepcopy(self.plan)
        confirmation["phase"] = "confirmation"
        with self.assertRaisesRegex(ValueError, "confirm-confirmation-run"):
            runner.run_generation(confirmation, [self.problem], self.manifest, self.root / "unused")
        self.assertFalse((self.root / "unused").exists())

    def test_generation_configuration_explicit_topk_and_no_inherited_defaults(self):
        fake = types.SimpleNamespace(GenerationConfig=lambda **kwargs: kwargs)
        with patch.dict(sys.modules, {"transformers": fake}):
            config = runner.generation_config(self.plan, self.tokenizer)
        self.assertEqual(config["top_k"], 0)
        self.assertEqual(config["temperature"], 0.7)
        self.assertEqual(config["top_p"], 0.95)
        self.assertEqual(config["num_return_sequences"], 1)
        self.assertEqual(config["repetition_penalty"], 1.0)

    def test_atomic_extraction_shapes_hashes_and_incomplete_pair(self):
        import numpy as np

        record = self.record("a" * 520 + "</think>")
        meta = runner.extraction_metadata(record, [512, 2048], self.runtime,
                                          runner.answer_boundary(self.tokenizer, record))
        npz_path, json_path = self.root / "vectors.npz", self.root / "vectors.json"
        vectors = np.ones((1, 29, 1536), dtype=np.float16)
        runner.write_extraction(npz_path, json_path, vectors, meta)
        runner.validate_extraction(npz_path, json_path, meta)
        self.assertEqual(meta["missing_checkpoints"], [2048])
        with open(npz_path, "ab") as handle:
            handle.write(b"changed")
        with self.assertRaisesRegex(ValueError, "vectors file changed"):
            runner.validate_extraction(npz_path, json_path, meta)
        json_path.unlink()
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            runner.validate_extraction(npz_path, json_path, meta)

    def test_extraction_feeds_recorded_prefix_only_and_reads_last_token(self):
        import numpy as np

        self.runtime["max_position_embeddings"] = 16384
        directory = runner.bind_output(self.root / "run", self.manifest, create=True)
        runner.bind_runtime(directory, self.runtime, create=True)
        record = self.record("a" * 2050 + "</think>answer")
        runner.atomic_json(directory / "traces" / (record["trace_id"] + ".json"), record)
        other = copy.deepcopy(record)
        other["sample_index"] = 1
        other["trace_id"] = runner.sample_key("p1", 1)
        other["seed"] = runner.sample_seed(self.plan, "p1", 1)
        del other["record_sha256"]
        other["record_sha256"] = runner.sha_obj(other)
        runner.atomic_json(directory / "traces" / (other["trace_id"] + ".json"), other)
        fed = []

        class Array:
            def __init__(self, values):
                self.values = np.asarray(values)
            def to(self, *args):
                self.values = self.values.astype(np.float16)
                return self
            def cpu(self):
                return self
            def numpy(self):
                return self.values

        class Hidden:
            def __init__(self, index):
                self.index = index
            def __getitem__(self, item):
                self_test.assertEqual(item, (0, -1, slice(None)))
                return Array(np.full(1536, self.index))

        def model_forward(input_ids, attention_mask, use_cache, output_hidden_states, return_dict):
            self.assertFalse(use_cache)
            self.assertTrue(output_hidden_states)
            self.assertTrue(return_dict)
            fed.append(input_ids.values[0].tolist())
            return types.SimpleNamespace(hidden_states=[Hidden(i) for i in range(29)])

        self_test = self
        torch = types.SimpleNamespace(
            tensor=lambda ids, **kwargs: Array(ids), long="long", float16="float16",
            ones_like=lambda value: Array(np.ones_like(value.values)), inference_mode=nullcontext,
            stack=lambda values: Array(np.stack([x.values for x in values])))
        model = types.SimpleNamespace(model=model_forward)
        with patch.dict(sys.modules, {"torch": torch}), patch.object(runner, "load_model", return_value=(self.tokenizer, model, self.runtime)), patch("builtins.print"):
            runner.run_extraction(self.plan, [self.problem], self.manifest, directory)
        expected = [record["prompt_token_ids"] + record["generated_token_ids"][:p] for p in (512, 2048)] * 2
        self.assertEqual(fed, expected)
        with np.load(directory / "extractions" / (record["trace_id"] + ".npz"), allow_pickle=False) as data:
            self.assertEqual(data["vectors"].shape, (2, 29, 1536))
            self.assertEqual(data["vectors"][0, 28, 0], 28)

    def test_zero_width_marker_offsets_are_unknown(self):
        record = self.record()

        class ZeroOffsets(Tokenizer):
            def __call__(self, *args, **kwargs):
                data = super().__call__(*args, **kwargs)
                if "offset_mapping" in data:
                    data["offset_mapping"][510] = (0, 0)
                return data

        self.assertEqual(runner.answer_boundary(ZeroOffsets(), record)["status"], "unknown")


if __name__ == "__main__":
    unittest.main()
