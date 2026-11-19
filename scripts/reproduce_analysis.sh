#!/usr/bin/env bash
# every CPU analysis behind the paper's tables and figures, in order, from the data layout in README.
#   E0_DATA=data scripts/reproduce_analysis.sh            # writes results/ and figures/
# token confidence needs src/replication/confidence.py outputs first (GPU); skipped if they are missing.
set -euo pipefail
cd "$(dirname "$0")/.."
D=${E0_DATA:-data}
export E0_DATA=$D
PY=${PY:-python}
R=results
mkdir -p $R/deepseek $R/models $R/limit_hits
A="--rollouts $D/recovery_ceiling --traces $D/hf/traces_B.jsonl"
S="--meta $D/hf/meta_B.parquet --acts $D/hf/acts_B.npy"

# DeepSeek, explore / holdout halves (results/split.json, fixed before any re-run analysis)
for h in explore holdout; do
  $PY src/analysis/determination.py $A $S --split results/split.json --half $h --out $R/deepseek
  $PY src/analysis/forward_capture.py --rollouts $D/recovery_ceiling --split results/split.json --half $h --out $R/deepseek
  $PY src/analysis/commitment.py $A --meta $D/hf/meta_B.parquet --split results/split.json --half $h --out $R/deepseek
  $PY src/analysis/cheap_rollouts.py $A --split results/split.json --half $h --out $R/deepseek
  $PY src/analysis/allocation.py --traces $D/hf/traces_B.jsonl $S --split results/split_all.json --half $h --out $R/deepseek
done
$PY src/analysis/t5_envelope.py $R/deepseek $D/hf/traces_B.jsonl results/split_all.json $R/deepseek
$PY src/analysis/robustness.py $A $S --out $R/deepseek

# three models at 4,096, cohort rule; limit-hit sources counted as wrong for DeepSeek and Ministral
for m in deepseek qwen3 ministral; do
  $PY src/analysis/replicate.py --model $m --all-problems --out $R/models
done
for m in deepseek ministral; do
  $PY src/analysis/replicate.py --model $m --limit-hits wrong --out $R/limit_hits
done
$PY src/analysis/cohort_stats.py --out $R/cohort_stats.csv
$PY src/analysis/sampler_gap.py $D/recovery_ceiling $D/hf/traces_B.jsonl $D/qwen3 $D/ministral | tee $R/sampler_gap.txt
$PY src/analysis/position.py --out $R/position_bins.csv
if [ -d "$D/confidence/deepseek" ]; then
  $PY src/analysis/confidence_capture.py --out $R/confidence
fi
$PY src/analysis/figures.py --results $R --out figures
