# GPU runs on Modal

Two apps, each with its own Volume (`/inputs`, `/outputs`, `/hf` model cache). Every GPU step resumes:
`run_rollouts.py` skips (prefix, round) rows already written, so a preempted or stopped run picks up where it was.

## Setup

```bash
uv tool install modal && modal setup
```

## DeepSeek re-runs (`scripts/modal_app.py`, Volume `e0-recovery`)

Re-runs of every cohort trace (finished, past 4,096 tokens, problems with both outcomes) cut at 512, 2,048 and
4,096 tokens: 8 per cut, T 0.7, top-p 0.95. Problems run in tiers; a problem is either complete or not started.

```bash
# traces_B.jsonl: 2,202 lines, sha256 starts c8cb3d8f9b21453b
scripts/modal_upload.sh                                   # -> /inputs/traces_B.jsonl
modal run scripts/modal_app.py --step build               # prefixes and tiers, CPU
modal run scripts/modal_app.py --step temp                # exits non-zero unless T 0.7 fits best
modal run scripts/modal_app.py --step smoke               # 2 prefixes; a second run finds nothing to do
modal run scripts/modal_app.py --step bench --gpu h100    # 5 minutes, then cost projections
modal run --detach scripts/modal_app.py --step run --gpu h100 --budget <credit left, $>
modal volume get e0-recovery /outputs ./outputs
```

## Qwen3 and Ministral (`scripts/replication_modal.py`, Volume `e0-<model>`)

Sources (6 per problem), layer states at the cut, the cohort cuts at 4,096 and 8 re-runs each.
`E0_MODEL` picks the profile in `src/replication/spec.py`.

```bash
python src/replication/dev_problems.py --traces data/hf/traces_B.jsonl --out problems.jsonl   # Ministral: --levels 4,5
export E0_MODEL=qwen3-1.7b                                # or ministral3-3b
modal volume put --force e0-qwen3-1-7b problems.jsonl /inputs/problems.jsonl
modal run scripts/replication_modal.py --step smoke
modal run scripts/replication_modal.py --step t0
modal run --detach scripts/replication_modal.py --step sources --samples 6 --budget <$>
modal run scripts/replication_modal.py --step extract
modal run scripts/replication_modal.py --step cuts        # prints the re-run plan and its cost
modal run --detach scripts/replication_modal.py --step bank --gpu h200 --budget <$>
modal run scripts/replication_modal.py --step confidence
```

`E0_POOL_TRUNC=1` at the `cuts` step also re-runs sources that hit the 32,768-token limit (counted as wrong).
For DeepSeek's token confidence, run `--step confidence` with `E0_MODEL` unset, after putting `traces_B.jsonl`
and the cohort cuts (`ds_cohort.jsonl`: trace_id, cut, line, prefix_sha) in `/inputs`.

## Budget guard

- New problems stop once spent + the running work's estimated cost + the next problem reach budget − $2.
- At budget − $0.50 the run stops at once and drops what is still running.
- Spend uses `--rate-usd-hr` (H100 4.40, H200 4.99, T4 1.05) plus 4 minutes of startup per attempt; the ledger
  next to each output file carries spend across preemptions.

Measured: Qwen3-1.7B 41.3M tokens for about $28; Ministral sources about $0.63 per M tokens on H100, re-runs about
$0.50 per M on H200.
