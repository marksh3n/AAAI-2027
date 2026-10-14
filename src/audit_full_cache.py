"""Read-only structural audit of an existing e0_full cache; no model or GPU.

Usage: python -B src/audit_full_cache.py --cache-dir /path/to/e0_full
Optional: --expected-acts-bytes 1370173568 (the size reported by Drive).

Prints JSON to stdout and exits nonzero on failed checks. It never writes a
manifest, edits metadata or certifies historical token/activation alignment.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import numpy as np
import pandas as pd


def sha256(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def audit_cache(cache_dir, expected_acts_bytes=None, chunk_rows=128):
    root = Path(cache_dir)
    result = {
        "scope": "structural_only",
        "historical_input_provenance": "unverified",
        "cache_dir": str(root),
        "read_only": True,
        "checks": {},
        "files": {},
        "limitations": [
            "Matching file size, shape and fresh checksums do not prove historical token inputs.",
            "New checksums describe the files read now; no prior extraction manifest is created or inferred.",
            "This audit does not compare activations with a fresh model pass or rerun probe experiments.",
        ],
    }
    if type(chunk_rows) is not int or chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")

    def check(name, passed, detail=None):
        result["checks"][name] = {"passed": bool(passed)}
        if detail is not None:
            result["checks"][name]["detail"] = detail

    paths = {name: root / name for name in ("full_config.json", "meta_full.parquet", "acts_full.npy")}
    initial_stats = {}
    for name, path in paths.items():
        check(name + ".exists", path.is_file())
        if path.is_file():
            stat = path.stat()
            initial_stats[name] = (stat.st_size, stat.st_mtime_ns, stat.st_ino)
            result["files"][name] = {"bytes": stat.st_size}
    cfg = meta = acts = None
    if paths["full_config.json"].is_file():
        try:
            with paths["full_config.json"].open() as handle:
                cfg = json.load(handle)
            if not isinstance(cfg, dict):
                raise ValueError("Configuration must be a JSON object")
            check("config.readable", True)
            check("config.layer_order", cfg.get("layers") == list(range(-1, 28)),
                  {"expected": list(range(-1, 28)), "actual": cfg.get("layers")})
            check("config.layer_convention", cfg.get("layer_convention") == "hidden_states[l+1]")
        except Exception as exc:
            check("config.readable", False, f"{type(exc).__name__}: {exc}")
            cfg = None
    if paths["meta_full.parquet"].is_file():
        try:
            meta = pd.read_parquet(paths["meta_full.parquet"])
            check("metadata.readable", True)
            required = {"row", "trace_id", "ckpt", "pos"}
            check("metadata.required_columns", required <= set(meta.columns),
                  {"missing": sorted(required - set(meta.columns))})
            if required <= set(meta.columns):
                check("metadata.nonempty", len(meta) > 0)
                check("metadata.sequential_rows", np.array_equal(meta["row"].to_numpy(), np.arange(len(meta))))
                check("metadata.nonnull_trace_checkpoint", not meta[["trace_id", "ckpt"]].isna().any().any())
                check("metadata.unique_trace_checkpoint", not meta.duplicated(["trace_id", "ckpt"]).any())
                pos = pd.to_numeric(meta["pos"], errors="coerce").to_numpy(dtype=float)
                good_pos = np.isfinite(pos) & (pos > 0) & (pos == np.floor(pos))
                check("metadata.positive_integer_positions", good_pos.all())
                checkpoints = meta["ckpt"].astype(str)
                fixed = checkpoints != "end"
                fixed_pos = pd.to_numeric(checkpoints[fixed], errors="coerce").to_numpy(dtype=float)
                check("metadata.fixed_checkpoint_matches_position",
                      np.isfinite(fixed_pos).all() and np.array_equal(fixed_pos, pos[fixed.to_numpy()]))
                if cfg is not None:
                    check("config.rows_match_metadata", type(cfg.get("rows")) is int and cfg["rows"] == len(meta),
                          {"configured": cfg.get("rows"), "observed": len(meta)})
                    count = int(meta["trace_id"].nunique())
                    check("config.traces_match_metadata", type(cfg.get("traces")) is int and cfg["traces"] == count,
                          {"configured": cfg.get("traces"), "observed": count})
                    allowed = {str(v) for v in cfg.get("positions", [])}
                    unknown = sorted(set(checkpoints[fixed]) - allowed)
                    check("metadata.checkpoints_in_config", not unknown, {"unknown": unknown})
        except Exception as exc:
            check("metadata.readable", False, f"{type(exc).__name__}: {exc}")
            meta = None
    if paths["acts_full.npy"].is_file():
        size = paths["acts_full.npy"].stat().st_size
        if expected_acts_bytes is not None:
            check("activations.matches_reported_drive_size", size == expected_acts_bytes,
                  {"expected": expected_acts_bytes, "observed": size,
                   "meaning": "file size only; not token-input provenance"})
        try:
            acts = np.load(paths["acts_full.npy"], mmap_mode="r", allow_pickle=False)
            if not isinstance(acts, np.memmap):
                raise ValueError("Expected one memory-mapped NPY array, not an NPZ container")
            check("activations.readable", True)
            check("activations.fp16", acts.dtype == np.dtype("float16"), str(acts.dtype))
            check("activations.three_dimensions", acts.ndim == 3, list(acts.shape))
            if cfg is not None and meta is not None:
                expected_shape = (len(meta), len(cfg.get("layers", [])), 1536)
                check("activations.expected_shape", acts.shape == expected_shape,
                      {"expected": list(expected_shape), "observed": list(acts.shape)})
            payload = int(acts.size) * acts.dtype.itemsize
            check("activations.exact_header_plus_payload_size", size == int(acts.offset) + payload,
                  {"header_bytes": int(acts.offset), "payload_bytes": payload, "file_bytes": size})
            nonfinite_rows = []
            if acts.ndim >= 1:
                for start in range(0, len(acts), chunk_rows):
                    block = np.asarray(acts[start:start + chunk_rows])
                    valid = np.isfinite(block).reshape(len(block), -1).all(axis=1)
                    nonfinite_rows.extend((start + np.flatnonzero(~valid)).tolist())
                check("activations.all_finite", not nonfinite_rows,
                      {"nonfinite_row_count": len(nonfinite_rows), "first_bad_rows": nonfinite_rows[:20]})
        except Exception as exc:
            check("activations.readable", False, f"{type(exc).__name__}: {exc}")
        finally:
            if isinstance(acts, np.memmap):
                acts._mmap.close()
    # Hashes describe present bytes; they must not be relabelled as extraction-time evidence.
    for name, path in paths.items():
        if name not in initial_stats:
            continue
        try:
            result["files"][name]["sha256_observed_now"] = sha256(path)
            stat = path.stat()
            check(name + ".unchanged_during_audit",
                  initial_stats[name] == (stat.st_size, stat.st_mtime_ns, stat.st_ino))
        except Exception as exc:
            check(name + ".hash_readable", False, f"{type(exc).__name__}: {exc}")
    result["structural_checks_passed"] = all(item["passed"] for item in result["checks"].values())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--expected-acts-bytes", type=int)
    parser.add_argument("--chunk-rows", type=int, default=128)
    args = parser.parse_args()
    if args.chunk_rows < 1 or (args.expected_acts_bytes is not None and args.expected_acts_bytes < 1):
        parser.error("chunk rows and expected byte count must be positive")
    result = audit_cache(args.cache_dir, args.expected_acts_bytes, args.chunk_rows)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["structural_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
