"""Discover benchmark results and summarize them per dataset and model class."""

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from robustness_benchmark.core.protocol import BASE_SEEDS, result_dir
from robustness_benchmark.evaluation.aggregate import (
    family_survival_summary,
    method_summary,
)
from robustness_benchmark.results.labels import (
    DATASET_LABELS,
    MODEL_TYPE_LABELS,
    method_label,
)


@dataclass(frozen=True)
class Results:
    """Summaries of every benchmark found, with dataset and model-type columns."""

    overall: pd.DataFrame  # one row per dataset, model type and method
    family: pd.DataFrame  # one row per dataset, model type, method and family
    seeds: pd.DataFrame  # one row per dataset, model type, seed and method
    runtime: pd.DataFrame  # generation time per model type and method


def discover_runs(artifacts: Path) -> list[tuple[str, str, Path]]:
    """Return (dataset, model type, result root) for every benchmark present."""

    runs = []
    for model_type in MODEL_TYPE_LABELS:
        for dataset in DATASET_LABELS:
            root = result_dir(artifacts, dataset, model_type)
            if not root.is_dir():
                continue
            recorded = json.loads((root / "summary.json").read_text())["model_type"]
            if recorded != model_type:
                raise RuntimeError(f"{root} holds {recorded} results, not {model_type}")
            runs.append((dataset, model_type, root))
    if not runs:
        raise FileNotFoundError(f"No benchmark results found under {artifacts}")
    return runs


def base_seed(base_model_id: str) -> int:
    return int(base_model_id.rsplit("_", maxsplit=1)[-1])


def load_run(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generation and survival rows of one benchmark, checked for every seed."""

    generation = pd.read_parquet(root / "generation.parquet")
    survival = pd.read_parquet(root / "survival.parquet")
    expected = set(BASE_SEEDS)
    for name, frame in (("generation", generation), ("survival", survival)):
        seeds = {base_seed(value) for value in frame["base_model_id"].unique()}
        if seeds != expected:
            raise RuntimeError(
                f"{root} {name} rows cover seeds {sorted(seeds)}, "
                f"not the protocol's {sorted(expected)}"
            )
    return generation, survival


def runtime_summary(generation: pd.DataFrame) -> pd.DataFrame:
    """Observed per-factual generation time, including failed attempts."""

    rows = []
    for (model_type, method), group in generation.groupby(
        ["model_type", "method"], sort=True
    ):
        seconds = group["runtime_seconds"]
        timed_out = group.get(
            "apas_outcome", pd.Series(index=group.index, dtype=object)
        ).eq("timeout")
        rows.append(
            {
                "model_type": str(model_type),
                "method": str(method),
                "attempts_n": len(group),
                "mean_seconds": float(seconds.mean()),
                "standard_deviation_seconds": float(seconds.std(ddof=1)),
                "median_seconds": float(seconds.median()),
                "p90_seconds": float(seconds.quantile(0.9)),
                "maximum_seconds": float(seconds.max()),
                "timeouts_n": int(timed_out.sum()),
            }
        )
    return pd.DataFrame(rows)


def _labelled(frame: pd.DataFrame, dataset: str, model_type: str) -> pd.DataFrame:
    return frame.assign(
        dataset=dataset,
        dataset_label=DATASET_LABELS[dataset],
        model_type=model_type,
    )[["dataset", "dataset_label", "model_type", *frame.columns]]


def load_results(runs: list[tuple[str, str, Path]]) -> Results:
    overall, family, seeds, generations = [], [], [], []
    for dataset, model_type, root in runs:
        generation, survival = load_run(root)
        seed_rows = method_summary(generation, survival, by=("base_model_id", "method"))
        seed_rows.insert(0, "seed", seed_rows.pop("base_model_id").map(base_seed))
        seed_rows.insert(2, "method_label", seed_rows["method"].map(method_label))
        overall.append(
            _labelled(method_summary(generation, survival), dataset, model_type)
        )
        family.append(
            _labelled(
                pd.DataFrame(family_survival_summary(survival)), dataset, model_type
            )
        )
        seeds.append(_labelled(seed_rows, dataset, model_type))
        generations.append(generation.assign(model_type=model_type))
    return Results(
        overall=pd.concat(overall, ignore_index=True),
        family=pd.concat(family, ignore_index=True),
        seeds=pd.concat(seeds, ignore_index=True),
        runtime=runtime_summary(pd.concat(generations, ignore_index=True)),
    )
