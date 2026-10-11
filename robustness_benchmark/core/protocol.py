"""Frozen protocol settings shared by the CLIs: split, seeds and directories."""

from pathlib import Path

SPLIT_VERSION = "behavior_v3"
SPLIT_SEED = 2026
BASE_SEEDS = (2026, 2027, 2028, 2029, 2030)
# Seeds of the a priori base-configuration selection.
SELECTION_DEV_SEEDS = (2026, 2027, 2028)

# Offsets added to a base seed so each random draw has its own stream.
EVALUATION_FACTUALS_SEED_OFFSET = 30_000
TUNING_FACTUALS_SEED_OFFSET = 70_000
STABILITY_TAU_SEED_OFFSET = 81_000


def bank_dir(artifacts: Path, dataset: str, model_type: str) -> Path:
    return artifacts / f"model_bank_{dataset}_{model_type}"


def result_dir(artifacts: Path, dataset: str, model_type: str) -> Path:
    return artifacts / f"full_{dataset}_{model_type}"
