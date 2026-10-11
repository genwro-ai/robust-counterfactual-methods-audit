#!/usr/bin/env bash
# The full protocol: a model bank and a benchmark per dataset and model class,
# then the results packet. Re-running resumes: finished banks are kept and
# full_benchmark skips finished runs.
#
#   scripts/run_benchmark.sh [artifacts]
#   DATASETS="german_credit adult" MODEL_TYPES=mlp scripts/run_benchmark.sh
set -euo pipefail

ARTIFACTS="${1:-artifacts}"
DATASETS="${DATASETS:-breast_cancer diabetes wine_quality heloc german_credit adult taiwan_default}"
MODEL_TYPES="${MODEL_TYPES:-mlp logistic xgboost}"
N_FACTUALS="${N_FACTUALS:-250}"
WORKERS="${WORKERS:-2}"

for model_type in $MODEL_TYPES; do
  for dataset in $DATASETS; do
    bank="$ARTIFACTS/model_bank_${dataset}_${model_type}"
    if [[ ! -f "$bank/manifest.json" ]]; then
      uv run python -m robustness_benchmark.cli.model_bank \
        --dataset "$dataset" \
        --model-type "$model_type" \
        --include-betarce-ensemble \
        --workers "$WORKERS" \
        --output "$bank"
    fi
    uv run python -m robustness_benchmark.cli.full_benchmark \
      --bank "$bank" \
      --output "$ARTIFACTS/full_${dataset}_${model_type}" \
      --n-factuals "$N_FACTUALS"
  done
done

uv run python -m robustness_benchmark.cli.results_packet \
  --artifacts "$ARTIFACTS" \
  --output "$ARTIFACTS/results_packet"
