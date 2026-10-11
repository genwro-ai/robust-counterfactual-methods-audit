"""Robustness of counterfactuals under model changes."""

import json

import numpy as np
import pandas as pd

from robustness_benchmark.core.model_changes import ChangedModel, behavioral_distance


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
