# Jaden's data-checking handoff

This is a reviewed copy of repository commit `732da15decc8433226210a3e835f618c57db3543`. The original uploaded ZIP was not overwritten; changes are in this separate repository copy. No language-model inference, new continuation sampling or research-data probe retraining was run for this audit. CPU tests use small synthetic fixtures.

## What we checked

- The source file contains **2,202 unique traces from 367 problems**. IDs, labels and the 11,409 legacy metadata rows agree; the legacy activation array contains no NaN or infinite values.
- All **11,409 stored prefix strings** match the decoded token prefixes reconstructed with the pinned tokenizer.
- All **1,430 complete rollout records** have input hashes matching the reconstructed **two-BOS** prompt and their stated reasoning prefix. This includes 183 source traces from the first 480 source-file lines and 265 from later lines.
- The supplied repository's legacy `extract.encode` default uses **one BOS**. The rollout hashes establish two BOS for those saved rollout inputs. This is a real difference between the code paths; it does **not**, by itself, prove which inputs produced an old activation array.

The actual shared `e0_full` metadata, config, tables and logs have now been inspected read-only. All **15,380 metadata rows** pass source/row checks: **13,219 fixed prefixes** and **2,161 end strings** agree with the pinned-tokenizer/source reconstruction. The folder contains the **29-layer activation file**, but its 1,370,173,568 bytes exceed the 256 MiB download limit here, so its numerical contents have not been inspected.

On the rollout overlap, **794 main-run records** match the inputs declared by metadata; **549 pilot records from 183 traces** disagree because metadata declares one BOS while those rollout hashes match two. The latest recovery table already excludes the pilot: 265 traces / 52 problems at 512 and 2048; 264 / 52 at 4096. Do not mix these rows with the old 87-problem comparison or the larger grid table.

Original generation token IDs and extraction-time input hashes were not saved in these legacy full-cache files. Matching metadata or current checksums cannot certify the historical activations' exact inputs. Full numerical and historical-provenance verification remain open. Nothing in the shared Drive folder was changed.

## Data issue already isolated

The original rollout file has **10 malformed opening lines**, including an unrelated configuration header. They were quarantined without inventing replacement data. The cleaned file retains **1,430 complete records / 11,440 continuations**. One complete record remains missing: `ceil/216_2/p4096`.

Use the cleaned file whose SHA256 is:

`0cbd7c1a9b2d27d54b7949e192a76979d740405641ca95a4a769bf706f2cafc4`

It is in the prior handoff's `cleaned_data/recovery_ceiling_clean.jsonl`, beside `cleaning_manifest.json`. Do not silently fill in the missing prefix or mix the unrelated opening header with these records.

## What changed in the code

- `generate.py` and `generate_vllm.py`: future records save the actual prompt and generated token IDs with hashes.
- `extract.py` and `cache.py`: extraction records input hashes; cache reuse checks inputs and metadata. `--adopt` is disabled because approximate activation agreement cannot establish exact historical token inputs.
- `recovery/common.py`, `build_prefixes.py` and `run_rollouts.py`: saved token IDs take priority; reconstructed legacy inputs remain explicitly labeled. Resume checks prevent changed inputs or conflicting results from being silently reused.
- `probe_cmp/ceiling.py`: malformed/conflicting records fail clearly, and recorded extraction/rollout hash disagreements stop the comparison. Missing provenance is reported as unverified.
- `audit_alignment.py`: CPU-only reconstruction of one- and two-BOS candidates, checked against every stored rollout input hash.

## Rerun the CPU hash audit

Run from the repository folder. Replace the input paths below with your local files. The tokenizer folder needs `tokenizer.json`, `tokenizer_config.json`, `config.json` and `tokenizer_manifest.json`.

```bash
python -m pip install pandas pyarrow tokenizers jinja2
python src/audit_alignment.py \
  --traces /path/to/traces_B.jsonl \
  --rollouts /path/to/recovery_ceiling_clean.jsonl \
  --rollout-manifest /path/to/cleaning_manifest.json \
  --tokenizer-dir /path/to/pinned_tokenizer_folder \
  --out data/alignment_audit
```

Optionally add `--meta /path/to/meta_B.parquet` to check that every rollout checkpoint has one matching metadata row. The manifest check rejects an incomplete or changed cleaned file.

The audit writes `alignment_hash_report.json`, `alignment_hash_rows.csv` and `alignment_hash_issues.csv`. A successful hash audit confirms the saved rollout hash candidates; it does not certify historical generation or activation inputs.

Eight focused test scripts passed. Detailed logs and real-data checks are in `audit/`. The hash-audit unit tests can be repeated with:

```bash
python -m unittest discover -s tests -p test_audit_alignment.py
```

## Read-only full-cache check

The cache has been found; no further folder upload is needed. Where the large array is already accessible, run:

```bash
python -B src/audit_full_cache.py --cache-dir /path/to/e0_full --expected-acts-bytes 1370173568
```

This prints JSON and never writes to the cache. It checks shape, dtype, finite values, metadata order, file size and current checksums. It labels the result structural-only and historical provenance unverified. Its tiny-fixture tests do not substitute for running it on the real 1.37 GB array.

The downloaded result tables agree with their completed rerun log; all 805 long-form result cells match the rounded summaries. No probes were retrained. The first table run failed at the missing `216_2/p4096` record; the completed rerun explicitly excluded it. Detailed findings are in `audit/full_metadata_report.json` and `audit/full_table_review.json`.

The metadata audit is reproducible with `scripts/audit_full_metadata.py --help`; use an output directory outside the source/cache folders. The old wrapper has weak part-reuse provenance. Any controlled re-extraction should be a separate coding-team decision; no GPU rerun or Drive modification has been launched by this handoff.


## Independent recheck

An independent reconstruction confirmed the metadata, prefix strings, all saved rollout hash candidates, cohort counts and all 805 table cells. The three main-only recovery AUROCs and bootstrap confidence intervals were independently recomputed from saved continuation outcomes and matched exactly. Count flags remain diagnostic: the EOS-aware check flags 16 recovery-source traces, without proving original token identities.

The code review found and corrected one actual resume bug: token equality alone allowed a changed continuation budget or gold answer to reuse completed samples. `run_rollouts.py` now uses row format 4 and binds tokens, budget and gold answer; both execution call sites validate the effective continuation cap. The analysis loader refuses inconsistent pooling across rounds. Older unbound files remain readable for analysis/status, but a new bound run requires a separate output; never fabricate historical bindings.

All ten selected CPU scripts passed on the original delivered ZIP. After the fix, six targeted suites passed, and the new guard/audit code has 51 regression tests across five files. The final loader accepted the actual cleaned 1,430 records and rejected the original corrupt file. Evidence is in `audit/recheck_summary.json` and the associated `recheck_` files.

Real GPU/Transformers/vLLM integration is not validated by these CPU tests. The full activation file and historical input provenance remain unverified. Model names are not immutable weight revisions. `--rebuild` only replaces incompatible/unmanifested caches and can reuse same-config parts; do not describe it as forcing a new forward pass. No shared Drive files were changed.
