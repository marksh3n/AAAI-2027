# Ghait---MJJS

## Activation cache

Activation parts are keyed by line number in `traces.jsonl`, so a part file existing is
not evidence it holds the right activations. Every parts directory therefore carries a
`manifest.json`, and a part is reused only when the manifest proves it matches:

- **config hash** — model, rendered prompts (chat template + `INSTRUCTION`/`STRUCTURED`),
  `LAYERS`, `POSITIONS`, `MIN_STEP_TOKENS`, compute dtype, tokenizer, and
  `EXTRACTOR_VERSION`. A change to any of these refuses the whole directory.
- **per part** — `trace_id`, a hash of the trace's problem + text + `structured` flag, the
  trace's token count, the exact token indices extracted at, the array shape, and the
  file's sha256.

`meta.parquet` and `acts.npy` are joined by row index alone, so extraction also writes
`meta*.manifest.json` binding the two together; `probe.py` and `probe_fixed.py` refuse to
run on a pair that cannot be shown to come from one extraction under one config.

Nothing is ever deleted: a refused cache is moved to `<dir>.stale-<timestamp>`.

```bash
# normal run: reuses whatever the manifest can validate, extracts the rest
python src/extract.py --fixed --tag B

# a directory written before manifests existed (run B) -- re-extract a sample and
# compare it against the cache, then stamp a manifest if it reproduces
python src/extract.py --fixed --tag B --verify --verify-n 8
python src/extract.py --fixed --tag B --verify --verify-n 8 --adopt

# config genuinely changed: set the old parts aside and extract again
python src/extract.py --fixed --tag B --rebuild

# probe, optionally re-hashing acts.npy against its manifest
python src/probe_fixed.py --tag B --strict

python tests/test_cache.py
```

`--verify` compares per-(row, layer) cosine rather than exact equality: fp16 and a
different `CHUNK` move the last bits, but a part belonging to a different trace or a
different model moves the direction, so cosine separates the two cleanly.
