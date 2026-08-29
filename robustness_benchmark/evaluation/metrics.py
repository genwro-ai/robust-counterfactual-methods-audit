import json

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from robustness_benchmark.core.training import ChangedModel


def behavioral_distance(
    base_model, changed_model, X_reference: pd.DataFrame
) -> dict[str, float]:
    base_probability = base_model.predict(X_reference).iloc[:, 0].to_numpy()
    changed_probability = changed_model.predict(X_reference).iloc[:, 0].to_numpy()
    return {
        "hard_disagreement": float(
            np.mean((base_probability >= 0.5) != (changed_probability >= 0.5))
        ),
        "probability_mae": float(
            np.mean(np.abs(base_probability - changed_probability))
        ),
    }


def assess_model_changes(
    changes: list[ChangedModel],
    base_model,
    X_reference: pd.DataFrame,
    y_reference: pd.Series,
    *,
    maximum_accuracy_drop: float = 0.03,
) -> list[dict[str, object]]:
    """Measure candidate quality and relatively stratify the complete model bank."""

    base_probability = base_model.predict(X_reference).iloc[:, 0].to_numpy()
    labels = y_reference.to_numpy()
    base_prediction = (base_probability >= 0.5).astype(int)
    base_accuracy = float(balanced_accuracy_score(labels, base_prediction))
    rows: list[dict[str, object]] = []
    for change in changes:
        changed_probability = change.model.predict(X_reference).iloc[:, 0].to_numpy()
        changed_prediction = (changed_probability >= 0.5).astype(int)
        changed_accuracy = float(balanced_accuracy_score(labels, changed_prediction))
        accuracy_drop = base_accuracy - changed_accuracy
        rows.append(
            {
                "change_id": change.change_id,
                "change_family": change.change_family,
                "change_level": change.change_level,
                **behavioral_distance(base_model, change.model, X_reference),
                "base_reference_balanced_accuracy": base_accuracy,
                "changed_reference_balanced_accuracy": changed_accuracy,
                "balanced_accuracy_drop": accuracy_drop,
                "quality_pass": accuracy_drop <= maximum_accuracy_drop,
                "maximum_accuracy_drop": maximum_accuracy_drop,
                "metadata": change.metadata,
            }
        )

    stratify_severity(rows)
    return rows


def stratify_severity(rows: list[dict[str, object]]) -> None:
    """Relabel every candidate with relative severity from behavioral distance.

    This is the single stratification rule shared by bank construction and
    bank loading; every candidate participates, without quality exclusions.
    """

    ordered = sorted(
        rows,
        key=lambda row: (row["probability_mae"], row["hard_disagreement"]),
    )
    denominator = max(len(ordered), 1)
    for rank, row in enumerate(ordered, start=1):
        percentile = (rank - 0.5) / denominator
        row["severity_percentile"] = percentile
        row["severity"] = (
            "low"
            if percentile <= 1 / 3
            else "medium"
            if percentile <= 2 / 3
            else "high"
        )


def evaluate_survival(
    method: str,
    factuals: pd.DataFrame,
    counterfactuals: pd.DataFrame,
    changes: list[ChangedModel],
    base_model,
    X_reference: pd.DataFrame,
    y_reference: pd.Series,
    base_valid: list[bool],
    factual_ids: list[int],
    dataset_row_ids: list[int | str],
    base_model_id: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    base_probability = base_model.predict(X_reference).iloc[:, 0].to_numpy()
    base_accuracy = float(np.mean((base_probability >= 0.5) == y_reference.to_numpy()))

    for change in changes:
        distance = behavioral_distance(base_model, change.model, X_reference)
        changed_reference = change.model.predict(X_reference).iloc[:, 0].to_numpy()
        changed_accuracy = float(
            np.mean((changed_reference >= 0.5) == y_reference.to_numpy())
        )
        factual_probability = change.model.predict(factuals).iloc[:, 0].to_numpy()
        ce_probability = np.full(len(counterfactuals), np.nan)
        valid_positions = np.flatnonzero(np.asarray(base_valid, dtype=bool))
        if len(valid_positions):
            ce_probability[valid_positions] = (
                change.model.predict(counterfactuals.iloc[valid_positions])
                .iloc[:, 0]
                .to_numpy()
            )

        for position, (p_factual, p_ce) in enumerate(
            zip(factual_probability, ce_probability, strict=True)
        ):
            base_counterfactual_valid = bool(base_valid[position])
            factual_still_adverse = bool(p_factual < 0.5)
            conditional_eligible = base_counterfactual_valid and factual_still_adverse
            ce_valid = bool(p_ce >= 0.5) if base_counterfactual_valid else None
            rows.append(
                {
                    "method": method,
                    "base_model_id": base_model_id,
                    "change_id": change.change_id,
                    "factual_id": factual_ids[position],
                    "dataset_row_id": dataset_row_ids[position],
                    "change_family": change.change_family,
                    "change_level": change.change_level,
                    "change_metadata": json.dumps(change.metadata, sort_keys=True),
                    "change_quality_pass": change.metadata.get("quality_pass"),
                    "change_severity": change.metadata.get("severity"),
                    "severity_percentile": change.metadata.get("severity_percentile"),
                    **distance,
                    "base_reference_plain_accuracy": base_accuracy,
                    "changed_reference_plain_accuracy": changed_accuracy,
                    "plain_accuracy_drop": base_accuracy - changed_accuracy,
                    "factual_probability": float(p_factual),
                    "ce_probability": (
                        float(p_ce) if base_counterfactual_valid else None
                    ),
                    "base_valid": base_counterfactual_valid,
                    "factual_still_adverse": factual_still_adverse,
                    "conditional_eligible": conditional_eligible,
                    "ce_valid": ce_valid,
                    "conditional_survival": ce_valid if conditional_eligible else None,
                }
            )
    return rows
