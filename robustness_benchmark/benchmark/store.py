"""Finished method runs on disk: one directory per base seed and method.

A run is reused only when it was produced by the current protocol for the
same configuration, factuals, changed models, base model and bank.
"""

import json
from pathlib import Path

import pandas as pd

from robustness_benchmark.benchmark.run import MethodRun

BENCHMARK_PROTOCOL_VERSION = "full_benchmark_v12_restructured"


def run_directory(output: Path, base_seed: int, method: str) -> Path:
    return output / "runs" / f"seed_{base_seed}" / method


def write_run(directory: Path, run: MethodRun, factuals: pd.DataFrame) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(run.generation).to_parquet(
        directory / "generation.parquet", index=False
    )
    pd.DataFrame(run.survival).to_parquet(directory / "survival.parquet", index=False)
    run.counterfactuals.to_parquet(directory / "counterfactuals.parquet", index=False)
    factuals.to_parquet(directory / "factuals.parquet", index=False)


def run_is_current(
    directory: Path,
    config: dict[str, object],
    expected_dataset_row_ids: list[int | str],
    expected_change_ids: set[str],
    expected_base_model_id: str,
    expected_bank_fingerprint: str,
) -> bool:
    generation_path = directory / "generation.parquet"
    survival_path = directory / "survival.parquet"
    if not generation_path.is_file() or not survival_path.is_file():
        return False
    generation = pd.read_parquet(generation_path)
    survival = pd.read_parquet(survival_path)
    generation_columns = {
        "method_config",
        "dataset_row_id",
        "benchmark_protocol_version",
        "base_model_id",
        "bank_fingerprint",
    }
    survival_columns = {
        "change_id",
        "benchmark_protocol_version",
        "base_model_id",
        "bank_fingerprint",
    }
    if not generation_columns <= set(generation.columns) or not survival_columns <= set(
        survival.columns
    ):
        return False
    actual = generation["method_config"].dropna().unique().tolist()
    actual_row_ids = generation["dataset_row_id"].tolist()
    return (
        actual == [json.dumps(config, sort_keys=True)]
        and actual_row_ids == expected_dataset_row_ids
        and set(survival["change_id"].unique()) == expected_change_ids
        and generation["base_model_id"].eq(expected_base_model_id).all()
        and survival["base_model_id"].eq(expected_base_model_id).all()
        and generation["bank_fingerprint"].eq(expected_bank_fingerprint).all()
        and survival["bank_fingerprint"].eq(expected_bank_fingerprint).all()
        and generation["benchmark_protocol_version"]
        .eq(BENCHMARK_PROTOCOL_VERSION)
        .all()
        and survival["benchmark_protocol_version"].eq(BENCHMARK_PROTOCOL_VERSION).all()
    )


def read_protocol_runs(
    paths: list[Path],
    *,
    methods: set[str] | None = None,
    base_model_ids: set[str] | None = None,
) -> pd.DataFrame:
    """Keep only current-protocol rows belonging to this invocation."""

    frames = []
    for path in paths:
        frame = pd.read_parquet(path)
        if "benchmark_protocol_version" not in frame.columns:
            continue
        frame = frame[frame["benchmark_protocol_version"] == BENCHMARK_PROTOCOL_VERSION]
        if methods is not None:
            frame = frame[frame["method"].isin(methods)]
        if base_model_ids is not None:
            frame = frame[frame["base_model_id"].isin(base_model_ids)]
        if len(frame):
            frames.append(frame)
    if not frames:
        raise RuntimeError(
            f"No runs match protocol version {BENCHMARK_PROTOCOL_VERSION!r}"
        )
    return pd.concat(frames, ignore_index=True)
