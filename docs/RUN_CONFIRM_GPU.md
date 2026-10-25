# Confirmation data run on Modal (for Mark)

This generates the confirmation sources, their layer-20 states, and the rollout bank, all under
one pinned engine. The design is in `docs/CONFIRMATION_SPEC.md`. Code:

- `scripts/confirm_modal.py`
- `src/confirm/gpu_prep.py`
- `src/recovery/run_rollouts.py`

The Modal app and Volume are `e0-confirm`, separate from the ceiling run's `e0-recovery`.

**Do not start step 1 until both of these are true:**

- (a) Siddhant confirms the spec commit has been pushed, or its hashes sent to Ghait;
- (b) Jaden has filled `spec.JADEN` (problem-list sha256, samples per problem) on that commit.

`gpu_prep.py t0` refuses a problem file whose sha256 differs from `spec.JADEN["problems_file_sha256"]`.

## Once

The Modal setup is the same as `RUN_MODAL.md` ("Once"). Then:

```bash
git switch siddhant/recovery-modal && git pull
modal volume create e0-confirm                      # "already exists" is fine
modal volume put --force e0-confirm <jaden's problems.jsonl> /inputs/problems.jsonl
```

`problems.jsonl` has one JSON object per line with `problem_id`, `problem` and `gold`.

## Steps

Run them in order from the repo root. If a step fails, send Siddhant the full error and don't
work around it. Each GPU step resumes where it stopped, so rerunning the same command is safe.

```bash
# 0. smoke (T4, ~10 min, < $1): the whole chain on 2 bundled problems, small caps; must end with
#    "smoke check passed". Send Siddhant the last 40 lines either way.
modal run scripts/confirm_modal.py --step smoke

# 1. prompts (CPU): one-BOS prompt ids for every problem -> /outputs/t0.jsonl
modal run scripts/confirm_modal.py --step t0

# 2. source traces (GPU): spec.JADEN["samples_per_problem"] per problem, every token id saved
modal run --detach scripts/confirm_modal.py --step sources --gpu h100 --budget <credit left in $>

# 3. layer-20 states at 2048 / 4096 / 6144 from the exact token ids (GPU)
modal run scripts/confirm_modal.py --step extract --gpu h100

# 4. live cuts -> rollout prefixes, max_tokens = 32768 - cut (CPU)
modal run scripts/confirm_modal.py --step cuts

# 5. rollout bank (GPU): 8 fresh samples per problem (t0, round 1), then 8 continuations per live cut
modal run --detach scripts/confirm_modal.py --step bank --gpu h100 --budget <credit left in $>

# progress and spend at any point (CPU)
modal run scripts/confirm_modal.py --step status

# fetch everything for the analysis
modal volume get e0-confirm /outputs ./confirm_outputs
```

The steps 2 and 5 commands above have no `--samples` flag. If `spec.JADEN["samples_per_problem"]`
is still empty when you reach step 2, stop and ask; don't pass `--samples` yourself.

## What the smoke test checks

| Check | Proves |
|---|---|
| exactly one leading BOS, and vLLM echoes the prompt ids unchanged | the BOS convention is the frozen one |
| `len(token_ids) == n_gen` for every source; text == decode(ids) | the saved ids are the sampled ids |
| the stored layer-20 state == one full forward pass at c − 1 (cos > 0.999) | chunked extraction reads the right token |
| a bank row for every live cut, and a t0 round-1 row per problem | nothing is missing from the bank |
| token-id hashes on every bank continuation | bank samples are bound to their tokens |

The sampler config written next to each output (`*.config.json`) records the model revision, the
vLLM version, temperature, top_p and the cap. `bundle.py` refuses sources and bank that differ in
any of them.

## After the GPU steps (CPU, Siddhant)

```bash
python src/confirm/bundle.py --sources confirm_outputs/sources.jsonl --bank confirm_outputs/bank.jsonl \
    --t0 confirm_outputs/t0.jsonl --feats confirm_outputs/feats --monitors <frozen monitors dir> --out confirm_bundle
python src/confirm/analyze.py --bundle confirm_bundle --out confirm_results
# only if the report's resample validity check excludes 0 at 4096 (pre-specified sensitivity):
python src/confirm/analyze.py --bundle confirm_bundle --out confirm_results_keep_rollout --keep rollout
```
