"""Tiny CPU fixtures for the read-only full-cache auditor."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
from audit_full_cache import audit_cache, sha256


class FullCacheAudit(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.cfg = {"layers": list(range(-1, 28)), "layer_convention": "hidden_states[l+1]",
                    "positions": [128], "rows": 4, "traces": 2}
        (self.root / "full_config.json").write_text(json.dumps(self.cfg))
        self.meta = pd.DataFrame({"trace_id": ["1_0", "1_0", "1_1", "1_1"],
                                  "ckpt": ["128", "end", "128", "end"],
                                  "pos": [128, 200, 128, 210], "row": [0, 1, 2, 3]})
        self.meta.to_parquet(self.root / "meta_full.parquet")
        np.save(self.root / "acts_full.npy", np.zeros((4, 29, 1536), dtype=np.float16))

    def tearDown(self):
        self.temp.cleanup()

    def test_good_fixture_is_read_only_and_never_claims_historical_provenance(self):
        before = {p.name: (sha256(p), p.stat().st_mtime_ns) for p in self.root.iterdir()}
        capture = io.StringIO()
        with contextlib.redirect_stdout(capture):
            result = audit_cache(self.root, (self.root / "acts_full.npy").stat().st_size, chunk_rows=2)
        after = {p.name: (sha256(p), p.stat().st_mtime_ns) for p in self.root.iterdir()}
        self.assertEqual(before, after)
        self.assertEqual(capture.getvalue(), "")
        self.assertTrue(result["structural_checks_passed"])
        self.assertEqual(result["scope"], "structural_only")
        self.assertEqual(result["historical_input_provenance"], "unverified")
        self.assertIn("sha256_observed_now", result["files"]["acts_full.npy"])

    def test_appended_bytes_and_wrong_reported_size_fail(self):
        with (self.root / "acts_full.npy").open("ab") as handle:
            handle.write(b"unexpected trailing bytes")
        result = audit_cache(self.root, expected_acts_bytes=1)
        self.assertFalse(result["checks"]["activations.exact_header_plus_payload_size"]["passed"])
        self.assertFalse(result["checks"]["activations.matches_reported_drive_size"]["passed"])

    def test_shape_dtype_and_nonfinite_values_fail(self):
        array = np.zeros((4, 28, 1536), dtype=np.float32)
        array[3, 1, 0] = np.nan
        np.save(self.root / "acts_full.npy", array)
        result = audit_cache(self.root, chunk_rows=2)
        for name in ("activations.fp16", "activations.expected_shape", "activations.all_finite"):
            self.assertFalse(result["checks"][name]["passed"])
        self.assertEqual(result["checks"]["activations.all_finite"]["detail"]["first_bad_rows"], [3])

    def test_metadata_duplicate_order_position_and_config_counts_fail(self):
        self.meta.loc[1, ["row", "ckpt", "pos"]] = [3, "128", 127]
        self.meta.to_parquet(self.root / "meta_full.parquet")
        self.cfg.update(rows=9, traces=7, layers=list(range(28, -1, -1)))
        (self.root / "full_config.json").write_text(json.dumps(self.cfg))
        result = audit_cache(self.root)
        for name in ("metadata.sequential_rows", "metadata.unique_trace_checkpoint",
                     "metadata.fixed_checkpoint_matches_position", "config.rows_match_metadata",
                     "config.traces_match_metadata", "config.layer_order"):
            self.assertFalse(result["checks"][name]["passed"])

    def test_missing_and_truncated_arrays_are_reported_without_fabrication(self):
        path = self.root / "acts_full.npy"
        path.unlink()
        result = audit_cache(self.root)
        self.assertFalse(result["structural_checks_passed"])
        self.assertFalse(result["checks"]["acts_full.npy.exists"]["passed"])
        self.assertNotIn("acts_full.npy", result["files"])
        path.write_bytes(b"broken NPY")
        result = audit_cache(self.root)
        self.assertFalse(result["checks"]["activations.readable"]["passed"])


if __name__ == "__main__":
    unittest.main()
