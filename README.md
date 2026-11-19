# Outcome determination in reasoning traces

Code for the paper. How much of a reasoning attempt's outcome its prefix already fixes (outcome
determination D), and how much of that a monitor reads (capture). Three models: DeepSeek-R1-Distill-Qwen-1.5B,
Qwen3-1.7B, Ministral-3-3B-Reasoning; MATH-500 levels 3-5.

## Layout

- `src/generate_vllm.py`, `src/generate.py`: DeepSeek source traces (6 per problem, T 0.7, top-p 0.95)
- `src/extract.py`, `src/cache.py`: hidden states at fixed positions, manifest-checked cache
- `src/probe_cmp/`: probes and baselines, pooled and within problem; `ceiling.py` scores r̂ beside them
- `src/recovery/`: re-runs from a prefix (`build_prefixes.py`, `run_rollouts.py`), temperature check
- `src/replication/`: the same pipeline for Qwen3 and Ministral (`spec.py` profiles), token confidence
- `src/analysis/`: D, capture, re-run cost, restart policies, three-model tables, figures
- `scripts/`: Modal wrappers (`docs/modal.md`), `reproduce_analysis.sh`, `colab/` GPU audits and the layer sweep
- `results/split.json`, `results/split_all.json`: problem splits, fixed before any re-run analysis
- `results/holdout_checks/`: the checks written after the explore half and before the holdout half ran
- `tests/`: plain scripts on synthetic data, CPU only

## Data

`E0_DATA` (default `data/`):

- `hf/traces_B.jsonl`, `hf/meta_B.parquet`, `hf/acts_B.npy`: DeepSeek sources, prefix rows, layer states
- `recovery_ceiling`: DeepSeek re-runs; `rollouts_limit.jsonl`: re-runs of its limit-hit sources
- `qwen3/`, `ministral/`: `replication_modal.py` outputs (`sources.jsonl`, `bank.jsonl`, `feats/`)
- `problems_l45.jsonl`: Ministral's problems (levels 4-5)
- `confidence/{deepseek,qwen3,ministral}/`: `confidence.py` outputs

`src/probe_cmp/fetch.py` downloads the DeepSeek files from the HF dataset named by `E0_HF_REPO`.

## Run

```bash
uv sync                                  # or pip install -e .
for t in tests/test_*.py; do python $t || break; done
E0_DATA=data scripts/reproduce_analysis.sh   # CPU, writes results/ and figures/
```

GPU steps (sources, states, re-runs, token confidence): `docs/modal.md`. Layer sweep and its shuffled-label
control: `scripts/colab/extract_full.py`, then `scripts/colab/layer_sweep.py`.

## Paper map

- Pooled vs within-problem AUROC: `probe_cmp/iter1.py` (pooled 0.887 vs 0.854), `probe_cmp/run.py` + `report.py`
  (end of trace 0.809 vs 0.787), `colab/layer_sweep.py` (block 20 at 6,144: 0.502), `probe_cmp/ceiling.py`
  (r̂ 0.63 / 0.66 / 0.81)
- D and capture by cut, explore and holdout; Figure 1: `analysis/determination.py`, `forward_capture.py`,
  `commitment.py`
- Robustness of D, stronger readouts: `analysis/robustness.py`; limit-hit row `replicate.py --limit-hits wrong`;
  null simulation `sampler_gap.py`
- Re-run cost; Figure 2: `analysis/cheap_rollouts.py`
- Three models; Figure 4: `analysis/replicate.py`, `cohort_stats.py`, `sampler_gap.py`
- Token confidence: `replication/confidence.py` (GPU), `analysis/confidence_capture.py`
- D by position: `analysis/position.py`
- Restart policies at equal compute; Figure 3: `analysis/allocation.py`, `t5_envelope.py`
- Holdout checks: `results/holdout_checks/`

## Notes

- Sources were sampled at T 0.7 with a string prompt: lines 0-479 of `traces_B.jsonl` got one BOS, 480+ two
  (`colab/bos_check.py`); re-runs use two everywhere. `colab/temp_blocks.py` checks T 0.7 per block.
- Qwen3 and Ministral re-runs get 32,768 − cut tokens; DeepSeek's got 32,768 after the cut.
- DeepSeek states come from re-tokenized text; Qwen3 and Ministral from the sampled ids.
