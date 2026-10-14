"""CPU-only tests of exact-token provenance/BOS/cache guards; no model downloads."""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def stub(name, **attrs):
    module = types.ModuleType(name)
    vars(module).update(attrs)
    sys.modules[name] = module


stub("torch", Tensor=type("Tensor", (), {}), __version__="0.0.0")
stub("datasets", load_dataset=lambda *a, **k: None)
stub("math_verify", parse=lambda x: x, verify=lambda a, b: a == b)
stub("transformers", AutoTokenizer=object, AutoModelForCausalLM=object)

import numpy as np
import pandas as pd
import cache
import extract
import generate


class Tokenizer:
    bos_token_id, eos_token_id = 1, 2
    name_or_path, chat_template = "mock", "mock template"
    special_tokens_map = {"bos_token": "<BOS>", "eos_token": "<EOS>"}

    def get_vocab(self):
        return {"<BOS>": 1, "<EOS>": 2, "a": 10, "b": 11, "ab": 12, "prompt": 20}

    def __len__(self):
        return len(self.get_vocab())

    def apply_chat_template(self, *args, **kwargs):
        return "prompt<think>"

    def __call__(self, text, add_special_tokens=True, return_offsets_mapping=False):
        if text.startswith("prompt"):
            return {"input_ids": [1, 1, 20] if add_special_tokens else [1, 20]}
        out = {"input_ids": [10, 11]}
        if return_offsets_mapping:
            out["offset_mapping"] = [(0, 1), (1, 2)]
        return out

    def decode(self, ids, skip_special_tokens=False):
        pieces = {1: "<BOS>", 2: "<EOS>", 10: "a", 11: "b", 12: "ab", 20: "prompt"}
        return "".join(pieces[x] for x in ids if not skip_special_tokens or x not in (1, 2))


class TraceInputs(unittest.TestCase):
    def setUp(self):
        self.tok = Tokenizer()
        self.trace = {"trace_id": "7_0", "problem_id": 7, "problem": "question", "text": "ab",
                      "n_gen": 3, "trunc": False, "is_correct": True, "level": 3, "structured": False}

    def recorded(self, prompt=None, generated=None):
        return generate.attach_token_inputs(copy.deepcopy(self.trace), self.tok,
                                            [1, 1, 20] if prompt is None else prompt,
                                            [10, 11, 2] if generated is None else generated)

    def test_exact_generation_ids_and_terminal_eos(self):
        trace = self.recorded()
        self.assertEqual(trace["completion_token_ids"], [10, 11, 2])
        self.assertEqual(trace["generated_token_ids"], [10, 11])
        self.assertEqual(generate.recorded_token_inputs(self.tok, trace), ([1, 1, 20], [10, 11]))
        prompt, ids, offsets = extract.encode(self.tok, trace)
        self.assertEqual(prompt, [1, 1, 20])  # no extra BOS added
        self.assertEqual(offsets, [(0, 1), (1, 2)])

    def test_source_and_recorded_ids_are_bound(self):
        for field, value in (("text", "changed"), ("problem", "another question"), ("structured", True)):
            trace = self.recorded()
            trace[field] = value
            with self.assertRaisesRegex(ValueError, "different problem"):
                generate.recorded_token_inputs(self.tok, trace)
        trace = self.recorded()
        trace["prompt_token_ids"] = [1, 20]
        with self.assertRaisesRegex(ValueError, "ordered token IDs changed"):
            generate.recorded_token_inputs(self.tok, trace)

    def test_partial_recording_and_wrong_tokenizer_fail(self):
        trace = dict(self.trace, prompt_token_ids=[1, 20])
        with self.assertRaisesRegex(ValueError, "incomplete"):
            extract.encode(self.tok, trace)
        trace = self.recorded()
        self.tok.name_or_path = "another-tokenizer"
        with self.assertRaisesRegex(ValueError, "different tokenizer"):
            extract.encode(self.tok, trace)

    def test_tokenizer_identity_ignores_only_runtime_padding_and_truncation(self):
        config = {"model": {"type": "BPE", "vocab": {"a": 10, "b": 11}},
                  "added_tokens": [{"id": 1, "content": "<BOS>", "special": True}],
                  "padding": None, "truncation": None}
        self.tok.backend_tokenizer = types.SimpleNamespace(to_str=lambda: json.dumps(config))
        before = generate.tokenizer_identity(self.tok)
        trace = self.recorded()
        config["padding"] = {"strategy": "BatchLongest", "pad_id": 2, "pad_token": "<EOS>"}
        config["truncation"] = {"max_length": 999, "strategy": "LongestFirst"}
        self.tok.special_tokens_map = dict(self.tok.special_tokens_map, pad_token="<EOS>")
        self.assertEqual(before, generate.tokenizer_identity(self.tok))
        self.assertEqual(generate.recorded_token_inputs(self.tok, trace), ([1, 1, 20], [10, 11]))
        config["model"]["vocab"]["new"] = 42
        self.assertNotEqual(before, generate.tokenizer_identity(self.tok))
        del config["model"]["vocab"]["new"]
        config["added_tokens"].append({"id": 43, "content": "<new>", "special": True})
        self.assertNotEqual(before, generate.tokenizer_identity(self.tok))
        config["added_tokens"].pop()
        self.tok.chat_template = "different template"
        self.assertNotEqual(before, generate.tokenizer_identity(self.tok))

    def test_legacy_default_and_explicit_bos_policy(self):
        self.assertEqual(extract.encode(self.tok, self.trace)[0], [1, 20])
        self.assertEqual(extract.encode(self.tok, self.trace, legacy_n_bos=2)[0], [1, 1, 20])
        self.assertEqual(extract.encode(self.tok, dict(self.trace, n_bos=2))[0], [1, 1, 20])
        with self.assertRaisesRegex(ValueError, "conflicts"):
            extract.encode(self.tok, self.recorded(), legacy_n_bos=1)
        with self.assertRaisesRegex(ValueError, "integer 1 or 2"):
            extract.encode(self.tok, self.trace, legacy_n_bos=True)

    def test_retokenization_drift_does_not_replace_recorded_tokens(self):
        # Alternative actual tokenization decodes to the same text.
        trace = self.recorded(generated=[12, 2])
        prompt, ids, offsets = extract.encode(self.tok, trace)
        self.assertEqual(ids, [12])
        self.assertIsNone(offsets)
        with self.assertRaisesRegex(ValueError, "offsets are unverified"):
            extract.segment(trace["text"], offsets)

    def test_part_fingerprint_changes_with_bos_or_same_length_tokens(self):
        base = extract.expect_part(self.trace, [10, 11], [1], [1, 20])
        other = extract.expect_part(self.trace, [10, 11], [1], [1, 1, 20])
        self.assertNotEqual(base["prompt_ids_sha"], other["prompt_ids_sha"])
        other = extract.expect_part(self.trace, [11, 10], [1], [1, 20])
        self.assertNotEqual(base["trace_ids_sha"], other["trace_ids_sha"])

    def test_metadata_mutation_is_rejected_even_with_same_trace_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            meta, acts = str(Path(tmp) / "meta.parquet"), str(Path(tmp) / "acts.npy")
            Path(meta).write_bytes(b"original labels")
            Path(acts).write_bytes(b"activations")
            cache.write_dataset_manifest(meta, acts, {}, cache.sha_obj({}), ["7_0"], (1, 1, 1))
            Path(meta).write_bytes(b"different labels")
            with self.assertRaisesRegex(cache.StaleCache, "metadata contents changed"):
                cache.check_dataset_manifest(meta, acts, ["7_0"], (1, 1, 1))

    def test_fixed_extraction_reuses_only_matching_inputs_and_records_prefix_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            traces = [dict(self.trace, n_gen=2, n_bos=1), dict(self.trace, trace_id="7_1", n_gen=2, n_bos=1)]
            paths = [str(root / n) for n in ("acts.npy", "meta.parquet", "parts")]
            calls = []
            def fake_activations(model, device, prompt, ids, indices):
                calls.append((prompt[:], ids[:]))
                return np.zeros((len(indices), len(extract.LAYERS), 3), dtype=np.float16)
            with patch.object(extract, "POSITIONS", [1, 2]), patch.object(extract, "activations", fake_activations):
                extract.fixed(self.tok, None, "cpu", traces, *paths)
                self.assertEqual(len(calls), 2)
                extract.fixed(self.tok, None, "cpu", traces, *paths)
                self.assertEqual(len(calls), 2)
                traces[0]["n_bos"] = 2
                extract.fixed(self.tok, None, "cpu", traces, *paths)
                self.assertEqual(len(calls), 3)
            meta = pd.read_parquet(paths[1])
            self.assertEqual(meta.iloc[0].prefix_sha, cache.sha_obj([[1, 1, 20], [10]]))
            self.assertEqual(meta.iloc[0].token_input_source, "reconstructed_explicit_bos_policy")

    def test_legacy_adoption_cannot_certify_historical_token_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            traces = [self.trace, dict(self.trace, trace_id="7_1")]
            for n in range(2):
                np.save(Path(tmp) / f"{n}.npy", np.zeros((1, len(extract.LAYERS), 3)))
            with patch.object(extract, "POSITIONS", [1]):
                for sample_size in (1, 2):
                    with self.assertRaisesRegex(cache.StaleCache, "--adopt is disabled"):
                        extract.verify(self.tok, None, "cpu", traces, tmp, "fixed", sample_size, True)


if __name__ == "__main__":
    unittest.main()
