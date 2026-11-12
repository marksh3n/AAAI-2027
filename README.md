# Ghait---MJJS

## Results map

Sections of [notebooks/E0_reproduce.ipynb](notebooks/E0_reproduce.ipynb).
Scripts formerly only `%%writefile` cells in `notebooks/E0_colab.ipynb`: `scripts/colab/`.

| # | result | code | notebook section |
|---|---|---|---|
| 0 | data creation: traces, activations, rollouts | `src/generate_vllm.py`, `src/extract.py`, `src/recovery/` | 0 |
| 1 | between problems, L16 probe 0.887 vs step count 0.854 | `src/probe_cmp/iter1.py` | 1 |
| 2 | within problem, no checkpoint beats len; end 0.809 vs 0.787 | `src/probe_cmp/run.py`, `report.py` | 2 |
| 3 | 29 layers × 13 checkpoints; 6144: 0.486 | `scripts/colab/extract_full.py`, `e0_full_table.py` | 3 |
| 4 | no window/delta view beats chance after Holm; all trail r̂ | `src/probe_cmp/extract_reps.py`, `reps.py` | 4 |
| 5 | hgb/topk50 at 6144 picked as best cell | `src/probe_cmp/run.py --exploratory` | 5 |
| 6 | 342/74: tree 0.523 vs len 0.554; no rule gives 197/53 | `src/confirm/provenance_6144.py` | 6 |
| 7 | frozen fit: tree@6144 0.560 vs len 0.554 | `src/confirm/fit_monitors.py` | 7 |
| 8 | S7 pre-answer tree − len +0.090 [−0.029, +0.211] | `src/confirm/dev_bundle.py`, `analyze.py` | 8 |
| 9 | r̂ 0.626 / 0.655 / 0.811; +0.262 over best probe at 4096 | `src/probe_cmp/ceiling.py` | 9 |
| 10 | rollout-ranked restart +5.7 pp at 4096, q 0.25 | `src/probe_cmp/oracle.py` | 10 |
| 11 | oracle v1–v3, length-matched: superseded | `src/probe_cmp/oracle.py` | 11 (pointer) |
| 12 | monitor restart +1.1 pp vs random, +1.6 vs maj@3 | `src/confirm/analyze.py`, `policy.py`, `dev_bundle.py` | 12 (run in 8) |
| 13 | detect power 0.18–0.51; rule-out of ≥ 5 pp 0.99 | `src/confirm/discordance.py`, `power.py` | 13 |
| 14 | latest fit, dry run, hashes | `src/confirm/` | 14 |
| 15 | T̂ ≈ 0.70 in every block | `scripts/colab/temp_blocks.py`, `src/recovery/check_temp.py` | 15 |
| 16 | lines 0–479 one BOS, 480+ two | `scripts/colab/bos_check.py` | 16 |
| 17 | rollout − source gap 7.3 / 5.1 / 3.0 pp, unexplained | `scripts/colab/lenfilter.py`, `mixed_sel.py`, `src/probe_cmp/oracle.py --length-matched` | 17 |
| 18 | 17.6 vs 15.7 slips per 1k claims | `src/injection/slip_scan.py` | 18 |
| 19 | injected digit error: not run | `src/injection/inject.py` | 19 |

The r(t) ceiling run on Modal: see [RUN_MODAL.md](RUN_MODAL.md).

Confirmation design (monitors, policies, cost accounting, analysis plan):
[docs/CONFIRMATION_SPEC.md](docs/CONFIRMATION_SPEC.md); its GPU run:
[docs/RUN_CONFIRM_GPU.md](docs/RUN_CONFIRM_GPU.md); its dev runs: [docs/CONFIRM_DEV_COLAB.md](docs/CONFIRM_DEV_COLAB.md).

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
python tests/test_inject.py      # injection gate 2: matched edit sites, pairing, report metrics
python tests/test_ceiling.py     # r(t) ceiling vs probes: pooling, alignment, AUROC, paired deltas
python tests/test_probe_cmp.py   # probe comparison across checkpoints, synthetic data
python tests/test_oracle.py      # oracle-r_hat intervention bound: halves, ties, cross-fit, resample control
python tests/test_reps.py        # representation test: window / last-token arithmetic, planted signal
python tests/test_confirm.py     # confirmation design: monitors == probe_cmp fits, vote, costs, token matching
python tests/test_confirm_pipeline.py  # fit_monitors -> dev_bundle -> analyze end to end on synthetic files
python tests/test_replication.py # E0_MODEL profiles, Ministral prompt / [THINK] tags, level filter, bank plan
```

`--verify` compares per-(row, layer) cosine rather than exact equality: fp16 and a
different `CHUNK` move the last bits, but a part belonging to a different trace or a
different model moves the direction, so cosine separates the two cleanly.
