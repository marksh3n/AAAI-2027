[README.md](https://github.com/user-attachments/files/33419517/README.md)
# Ghait---MJJS

The r(t) ceiling run on Modal: see [RUN_MODAL.md](RUN_MODAL.md).

For the reviewed data/alignment fixes and remaining verification limits, see
[Jaden's handoff](JADEN_DATA_FIXES.md). No GPU jobs were run for that review.

## Activation cache

Activation parts are keyed by line number in `traces.jsonl`, so a part file existing is
not evidence it holds the right activations. Every parts directory carries a
`manifest.json`, and a part is reused only when the manifest proves it matches:

- **config hash** — extraction mode, model, rendered prompts (chat template +
  `INSTRUCTION`/`STRUCTURED`), `LAYERS`, compute dtype, tokenizer, `EXTRACTOR_VERSION`,
  and whichever of `POSITIONS` (fixed mode) or `MIN_STEP_TOKENS` (steps mode) that run
  actually reads — the other is inert, so it is hashed as null. A change to any of these
  refuses the whole directory.
- **per part** — `trace_id`, a hash of the trace's problem + text + `structured` flag, the
  trace's token count, ordered prompt/generated-token hashes, the exact token indices
  extracted at, the array shape, and the file's sha256.

`meta.parquet` and `acts.npy` are joined by row index alone, so extraction also writes
`meta*.manifest.json` binding the two together. Strict checking requires hashes of
both files. Newly created manifests prevent later mismatches; a manifest fabricated
after extraction cannot establish historical provenance.

Nothing is ever deleted: a refused cache is moved to `<dir>.stale-<timestamp>`.

```bash
# normal run: reuses whatever the manifest can validate, extracts the rest
python src/extract.py --fixed --tag B

# diagnostic comparison only; this does not certify or adopt a legacy cache
python src/extract.py --fixed --tag B --verify --verify-n 8

# config changed: set the old parts aside and extract again
python src/extract.py --fixed --tag B --rebuild

# probe, optionally re-hashing acts.npy against its manifest
python src/probe_fixed.py --tag B --strict

# tests are plain scripts, no pytest; each exits non-zero on failure
python tests/test_cache.py       # manifest and stale-reuse paths
python tests/test_ablation.py    # yield table and TR vs TRH on planted signal
python tests/test_positions.py   # --positions scoping and per-run output files
python tests/test_recover.py     # recover.py logic, with torch/vLLM stubbed
python tests/test_recovery.py    # src/recovery logic: prefixes, scoring, resume, temp MLE
python tests/test_slip_scan.py   # injection gate 1: claim detection, skips, load-bearing flags
python tests/test_ceiling.py     # r(t) ceiling vs probes: pooling, alignment, AUROC, paired deltas
python tests/test_probe_cmp.py   # probe comparison across checkpoints, synthetic data
```

`--verify` compares per-(row, layer) cosine rather than exact equality. Approximate
agreement does not prove exact historical BOS/token inputs, so `--adopt` is disabled
in extractor v3. Fresh extraction is required to establish new input-bound records.
Legacy text-only traces remain explicitly unverified; use a source-bound
`--bos-policy` when reconstructing an agreed legacy prompt convention.
