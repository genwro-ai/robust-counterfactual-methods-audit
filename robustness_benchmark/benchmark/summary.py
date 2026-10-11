"""The benchmark summary: pooled rows of one invocation and their summaries."""

import json
from pathlib import Path

import pandas as pd

from robustness_benchmark.benchmark.store import (
    BENCHMARK_PROTOCOL_VERSION,
    read_protocol_runs,
)
from robustness_benchmark.core.provenance import write_json
from robustness_benchmark.evaluation.aggregate import (
    family_survival_summary,
    generation_summary,
    overall_survival_summary,
    severity_summary,
)


def collect_runs(
    output: Path, methods: list[str], base_model_ids: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generation and survival rows of every (base model, method) invoked.

    Fails unless each invoked method has a current run for every base model.
    """

    invocation_methods = set(methods)
    invocation_base_model_ids = set(base_model_ids)
    generation = read_protocol_runs(
        sorted(output.glob("runs/*/*/generation.parquet")),
        base_model_ids=invocation_base_model_ids,
    )
    survival = read_protocol_runs(
        sorted(output.glob("runs/*/*/survival.parquet")),
        base_model_ids=invocation_base_model_ids,
    )
    generation_runs = set(
        generation[["base_model_id", "method"]].itertuples(index=False, name=None)
    )
    survival_runs = set(
        survival[["base_model_id", "method"]].itertuples(index=False, name=None)
    )
    complete_methods = {
        method
        for method in set(generation["method"]).intersection(survival["method"])
        if all(
            (base_model_id, method) in generation_runs
            and (base_model_id, method) in survival_runs
            for base_model_id in invocation_base_model_ids
        )
    }
    if not invocation_methods.issubset(complete_methods):
        raise RuntimeError(
            "Invoked methods do not have complete current-protocol runs: "
            f"{sorted(invocation_methods - complete_methods)!r}"
        )
    generation = generation[generation["method"].isin(complete_methods)]
    survival = survival[survival["method"].isin(complete_methods)]
    expected_runs = {
        (base_model_id, method)
        for base_model_id in invocation_base_model_ids
        for method in complete_methods
    }
    for name, frame in (("generation", generation), ("survival", survival)):
        actual_runs = set(
            frame[["base_model_id", "method"]].itertuples(index=False, name=None)
        )
        if actual_runs != expected_runs:
            raise RuntimeError(
                f"{name} runs do not match this invocation: "
                f"missing={sorted(expected_runs - actual_runs)!r}, "
                f"unexpected={sorted(actual_runs - expected_runs)!r}"
            )
    return generation, survival


def run_configs(generation: pd.DataFrame) -> list[dict[str, object]]:
    """The single configuration each (base model, method) run used."""

    configs = []
    for values, group in generation.groupby(["base_model_id", "method"], sort=True):
        if not isinstance(values, tuple) or len(values) != 2:
            raise RuntimeError(f"Unexpected run configuration key {values!r}")
        serialized = group["method_config"].dropna().unique().tolist()
        if len(serialized) != 1:
            raise RuntimeError(
                f"Expected one configuration for run {values!r}, got {serialized!r}"
            )
        configs.append(
            {
                "base_model_id": str(values[0]),
                "method": str(values[1]),
                "config": json.loads(str(serialized[0])),
            }
        )
    return configs


def write_summary(
    output: Path,
    methods: list[str],
    configs: dict[str, dict[str, object]],
    base_model_ids: list[str],
    *,
    calibrations: list[dict[str, object]],
    skipped_methods: dict[str, str],
    model_type: str,
) -> None:
    generation, survival = collect_runs(output, methods, base_model_ids)
    generation.to_parquet(output / "generation.parquet", index=False)
    survival.to_parquet(output / "survival.parquet", index=False)
    write_json(
        output / "summary.json",
        {
            "methods": sorted(str(method) for method in generation["method"].unique()),
            "invocation_methods": methods,
            "model_type": model_type,
            "skipped_methods": skipped_methods,
            "invocation_base_configs": configs,
            "run_configs": run_configs(generation),
            "calibrations": calibrations,
            "generation": generation_summary(generation),
            "survival_all_changes": overall_survival_summary(survival),
            "survival_by_family": family_survival_summary(survival),
            "survival_by_severity": severity_summary(survival),
            "benchmark_protocol_version": BENCHMARK_PROTOCOL_VERSION,
        },
    )
