"""CPU-only regression checks for token provenance, BOS policy and rollout resume."""
import copy
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src" / "recovery"), str(ROOT / "src")]
for name, attrs in {
    "torch": {}, "datasets": {"load_dataset": lambda *a, **k: {}},
    "transformers": {"AutoTokenizer": object, "AutoModelForCausalLM": object},
    "math_verify": {"parse": lambda x: x, "verify": lambda a, b: a == b},
}.items():
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    sys.modules[name] = module

import common
import build_prefixes as bp
import run_rollouts as rr
from cache import StaleCache, sha_file
from generate import attach_token_inputs


class Tokenizer:
    bos_token_id, eos_token_id = 1, 2
    name_or_path, chat_template, special_tokens_map = "test", "template", {}

    def get_vocab(self):
        return {"a": 10, "b": 11}

    def apply_chat_template(self, *args, **kwargs):
        return "<think>"

    def __call__(self, text, **kwargs):
        if text == "ab":
            return {"input_ids": [10, 11], "offset_mapping": [(0, 1), (1, 2)]}
        return {"input_ids": [1, 3]}

    def decode(self, ids, **kwargs):
        return "".join({10: "a", 11: "b"}.get(i, "") for i in ids)


def trace(tid="1_0"):
    return {"trace_id": tid, "problem_id": 1, "problem": "question", "text": "ab"}


def prefix():
    return {"prefix_id": "ceil/1_0/p2", "problem_id": 1, "trace_id": "1_0",
            "checkpoint": "p2", "cut_token_idx": 2, "prompt_token_ids": [1, 3],
            "prefix_gen_token_ids": [10, 11], "max_tokens": 10, "gold_answer": "7"}


def completed(rec):
    return {**{k: rec[k] for k in ("prefix_id", "problem_id", "trace_id", "checkpoint", "cut_token_idx")},
            "round": 0, "n": 2, "prefix_sha": rr.prefix_sha(rec), "config_hash": "cfg",
            "prefix_definition_sha": rr.prefix_definition_sha(rec),
            "gold_answer": rec["gold_answer"], "max_tokens": rec["max_tokens"],
            "seed": common.prefix_seed(0, rec["prefix_id"], 0),
            "samples": [{"sample_idx": i, "correct": bool(i), "truncated": False} for i in range(2)]}


class AlignmentGuards(unittest.TestCase):
    def test_explicit_bos_and_legacy_label(self):
        tok = Tokenizer()
        self.assertEqual(common.prompt_ids(tok, "q", 1), [1, 3])
        self.assertEqual(common.prompt_ids(tok, "q", 2), [1, 1, 3])
        self.assertEqual(common.trace_inputs(tok, trace())[3], "reconstructed_legacy_unverified")

    def test_policy_is_bound_to_source(self):
        with tempfile.TemporaryDirectory() as d:
            source, policy = Path(d) / "source.jsonl", Path(d) / "bos.json"
            source.write_text(json.dumps(trace()) + "\n")
            value = {"traces_sha256": sha_file(source), "basis": "likelihood inference", "traces": {"1_0": 1}}
            policy.write_text(json.dumps(value))
            self.assertEqual(common.load_bos_policy(policy, [trace()], source), value)
            source.write_text(json.dumps(trace()) + "\n\n")
            with self.assertRaises(ValueError):
                common.load_bos_policy(policy, [trace()], source)

    def test_saved_generation_ids_preferred_and_policy_conflict_rejected(self):
        tok = Tokenizer()
        saved = attach_token_inputs(trace(), tok, [1, 1, 3], [10, 11, 2])
        prompt, ids, offsets, provenance = common.trace_inputs(tok, saved, need_offsets=True)
        self.assertEqual((prompt, ids, provenance), ([1, 1, 3], [10, 11], "recorded_at_generation"))
        self.assertEqual(offsets, [(0, 1), (1, 2)])
        with self.assertRaises(ValueError):
            common.trace_inputs(tok, saved, {"traces": {"1_0": 1}})
        changed = copy.deepcopy(saved)
        changed["text"] = "changed"
        with self.assertRaises(ValueError):
            common.trace_inputs(tok, changed)

    def test_shared_t0_requires_same_prompt(self):
        traces = [trace("1_0"), trace("1_1")]
        with self.assertRaises(ValueError):
            bp.problem_prompt(Tokenizer(), traces, {"traces": {"1_0": 1, "1_1": 2}})

    def test_recorded_count_accepts_hf_and_vllm_eos_conventions(self):
        saved = attach_token_inputs(trace(), Tokenizer(), [1, 3], [10, 11, 2])
        for count in (2, 3):
            saved["n_gen"] = count
            self.assertTrue(common.canonical(saved, [10, 11]))
            self.assertFalse(common.canonical(saved, [10, 12]))
        saved["n_gen"] = 99
        self.assertFalse(common.canonical(saved, [10, 11]))

    def test_structured_legacy_prompt_is_forwarded(self):
        class StructuredTokenizer(Tokenizer):
            def apply_chat_template(self, messages, **kwargs):
                self.message = messages[0]["content"]
                return "<think>"
        tok = StructuredTokenizer()
        common.trace_inputs(tok, {**trace(), "structured": True})
        self.assertIn("Number each step", tok.message)

    def test_cut_checkpoint_and_duplicate_prefix_guards(self):
        p = prefix()
        rr.validate_prefixes([p])
        for changed in ({**p, "cut_token_idx": 1}, {**p, "checkpoint": "p3"},
                        {**p, "prefix_gen_token_ids": [True, 11]}, {**p, "gold_answer": None},
                        {**p, "gold_answer": " "}):
            with self.assertRaises(StaleCache):
                rr.validate_prefixes([changed])
        with self.assertRaises(StaleCache):
            rr.validate_prefixes([p, p])

    def test_resume_rejects_bad_metadata_samples_and_conflicts(self):
        p, row = prefix(), completed(prefix())
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "rollouts"
            path.write_text(json.dumps(row) + "\n")
            self.assertEqual(rr.done_rows(path, [p], "cfg", 0), {(p["prefix_id"], 0)})
            bad = [dict(row, problem_id=99), dict(row, n=3), dict(row, seed=1)]
            duplicate = copy.deepcopy(row)
            duplicate["samples"][1]["sample_idx"] = 0
            bad.append(duplicate)
            for item in bad:
                path.write_text(json.dumps(item) + "\n")
                with self.assertRaises(StaleCache):
                    rr.done_rows(path, [p], "cfg", 0)
            conflict = copy.deepcopy(row)
            conflict["samples"][0]["correct"] = True
            path.write_text(json.dumps(row) + "\n" + json.dumps(conflict) + "\n")
            with self.assertRaises(StaleCache):
                rr.done_rows(path, [p])

    def test_valid_last_record_is_not_deleted(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "rollouts"
            payload = json.dumps(completed(prefix()))
            path.write_text(payload)
            self.assertEqual(rr.repair_tail(path), 0)
            self.assertEqual(path.read_text(), payload + "\n")
            with path.open("a") as f:
                f.write('{"torn":')
            self.assertGreater(rr.repair_tail(path), 0)
            self.assertEqual(path.read_text(), payload + "\n")
            self.assertEqual(len(list(Path(d).glob("*.torn_tail.*.bin"))), 1)

    def test_resume_binds_continuation_budget_and_gold_answer(self):
        p, row = prefix(), completed(prefix())
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "rollouts"
            path.write_text(json.dumps(row) + "\n")
            for changed in ({**p, "max_tokens": 100}, {**p, "gold_answer": "8"}):
                with self.assertRaisesRegex(StaleCache, "budget or gold answer changed"):
                    rr.done_rows(path, [changed], "cfg", 0, 0)
            with self.assertRaisesRegex(StaleCache, "effective continuation budget"):
                rr.done_rows(path, [p], "cfg", 0, 5)
            row["max_tokens"] = 5
            path.write_text(json.dumps(row) + "\n")
            self.assertEqual(rr.done_rows(path, [p], "cfg", 0, 5), {(p["prefix_id"], 0)})

    def test_legacy_rows_are_readable_for_status_but_not_certified_for_resume(self):
        p, row = prefix(), completed(prefix())
        row.pop("prefix_definition_sha")
        row.pop("gold_answer")
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "rollouts"
            path.write_text(json.dumps(row) + "\n")
            self.assertEqual(rr.done_rows(path, [p]), {(p["prefix_id"], 0)})
            with self.assertRaisesRegex(StaleCache, "legacy row lacks"):
                rr.done_rows(path, [p], "cfg", 0, 0)

    def test_prefix_definition_cannot_skip_torn_record(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "prefixes"
            path.write_text(json.dumps(prefix()) + '\n{"prefix_id":')
            with self.assertRaises(json.JSONDecodeError):
                rr.read_prefixes(path)

    def test_sidecar_cannot_replace_prefix_file(self):
        with self.assertRaises(StaleCache):
            rr.check_paths("/tmp/rollouts.config.json", "/tmp/rollouts")

    def test_prompt_check_uses_explicit_ids(self):
        class FakeLLM:
            def generate(self, requests, params):
                self.requests = requests
                return [types.SimpleNamespace(prompt_token_ids=r["prompt_token_ids"]) for r in requests]
        llm = FakeLLM()
        recs = [prefix(), {**prefix(), "prompt_token_ids": [1, 1, 3]}]
        rr.check_prompts(llm, lambda **kw: kw, recs)
        self.assertEqual([r["prompt_token_ids"] for r in llm.requests], [[1, 3], [1, 1, 3]])


if __name__ == "__main__":
    unittest.main()
