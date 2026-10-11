"""One method on one base model: generate counterfactuals, then score them."""

import json
from dataclasses import dataclass

import pandas as pd

from robustness_benchmark.core.model_changes import ChangedModel
from robustness_benchmark.core.task import ClassificationTask
from robustness_benchmark.evaluation.plausibility import PlausibilityReference
from robustness_benchmark.evaluation.proximity import GowerDistance, proximity_metrics
from robustness_benchmark.evaluation.robustness import evaluate_survival
from robustness_benchmark.methods.registry import generate_counterfactuals


@dataclass(frozen=True)
class MethodRun:
    counterfactuals: pd.DataFrame
    generation: list[dict[str, object]]
    survival: list[dict[str, object]]


def base_valid(generation: list[dict[str, object]]) -> list[bool]:
    return [bool(row["base_valid"]) for row in generation]


def score_generation(
    task: ClassificationTask,
    factuals: pd.DataFrame,
    counterfactuals: pd.DataFrame,
    generation: list[dict[str, object]],
    plausibility: PlausibilityReference | None = None,
) -> None:
    """Add proximity, sparsity and (optionally) plausibility to each row."""

    valid = base_valid(generation)
    distance = GowerDistance(task.training_data.X, task.schema)
    metrics = proximity_metrics(distance, factuals, counterfactuals, valid)
    if plausibility is not None:
        metrics = [
            {**row, **support}
            for row, support in zip(
                metrics, plausibility.evaluate(counterfactuals, valid), strict=True
            )
        ]
    for row, values in zip(generation, metrics, strict=True):
        row.update(values)


def run_method(
    method: str,
    *,
    task: ClassificationTask,
    factuals: pd.DataFrame,
    factual_ids: list[int],
    dataset_row_ids: list[int | str],
    changes: list[ChangedModel],
    base_model_id: str,
    X_reference: pd.DataFrame,
    y_reference: pd.Series,
    method_config: dict[str, object],
    generation_models: list | None = None,
    plausibility: PlausibilityReference | None = None,
    provenance: dict[str, object] | None = None,
    progress_description: str | None = None,
) -> MethodRun:
    """Generate, score and evaluate on every changed model; stamp every row.

    Every generation and survival row carries the base model, the serialized
    method configuration and ``provenance``.
    """

    counterfactuals, generation = generate_counterfactuals(
        method,
        task,
        factuals,
        factual_ids=factual_ids,
        dataset_row_ids=dataset_row_ids,
        generation_models=generation_models,
        method_config=method_config,
        progress_description=progress_description,
    )
    score_generation(task, factuals, counterfactuals, generation, plausibility)
    survival = evaluate_survival(
        method,
        factuals,
        counterfactuals,
        changes,
        task.model,
        X_reference,
        y_reference,
        base_valid(generation),
        factual_ids,
        dataset_row_ids,
        base_model_id,
    )
    stamp = {
        "base_model_id": base_model_id,
        "method_config": json.dumps(method_config, sort_keys=True),
        **(provenance or {}),
    }
    for row in (*generation, *survival):
        row.update(stamp)
    return MethodRun(counterfactuals, generation, survival)
