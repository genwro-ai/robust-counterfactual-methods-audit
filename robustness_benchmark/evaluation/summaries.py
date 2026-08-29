import pandas as pd

from robustness_benchmark.evaluation.aggregate import (
    generation_metrics,
    normalize_survival_schema,
)


def ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def generation_summary(generation: pd.DataFrame) -> list[dict[str, object]]:
    """Summarize generation outcomes by method."""

    rows: list[dict[str, object]] = []
    for method, group in generation.groupby("method", sort=True):
        metrics = generation_metrics(group)
        errors = int(group["generation_error"].notna().sum())
        valid_group = group[group["base_valid"]]
        rows.append(
            {
                "method": method,
                **metrics,
                "error_n": errors,
                "mean_runtime_seconds": float(group["runtime_seconds"].mean()),
                "mean_l1_scaled": (
                    float(valid_group["l1_scaled_mean"].mean())
                    if len(valid_group)
                    else None
                ),
                "mean_l1_robust_scale": (
                    float(valid_group["l1_robust_scale_mean"].mean())
                    if len(valid_group)
                    else None
                ),
            }
        )
    return rows


def survival_summary(survival: pd.DataFrame) -> list[dict[str, object]]:
    """Summarize survival without conflating validity and coverage."""

    survival = normalize_survival_schema(survival)
    rows: list[dict[str, object]] = []
    keys = ["base_model_id", "method", "change_id", "change_family", "change_level"]
    for values, group in survival.groupby(keys, sort=True, dropna=False):
        group_values = values if isinstance(values, tuple) else (values,)
        total = len(group)
        base_valid = int(group["base_valid"].sum())
        eligible = int(group["conditional_eligible"].sum())
        survived = int(group["conditional_survival"].eq(True).sum())
        changed_valid = int(group["ce_valid"].eq(True).sum())
        rows.append(
            {
                **dict(zip(keys, group_values, strict=True)),
                "total_n": total,
                "base_valid_pairs_n": base_valid,
                "base_validity_rate": ratio(base_valid, total),
                "conditional_eligible_n": eligible,
                "factual_eligibility_rate": ratio(eligible, base_valid),
                "conditional_survived_n": survived,
                "conditional_survival": ratio(survived, eligible),
                "changed_model_valid_n": changed_valid,
                "changed_validity_given_base_valid": ratio(changed_valid, base_valid),
                "end_to_end_changed_validity": ratio(changed_valid, total),
                "hard_disagreement": float(group["hard_disagreement"].iloc[0]),
                "probability_mae": float(group["probability_mae"].iloc[0]),
                "plain_accuracy_drop": float(group["plain_accuracy_drop"].iloc[0]),
            }
        )
    return rows


def apas_generation_summary(generation: pd.DataFrame) -> list[dict[str, object]]:
    """Summarize calibrated APAS generation outcomes by base model."""

    rows: list[dict[str, object]] = []
    for base_model_id, group in generation.groupby("base_model_id", sort=True):
        generated_valid = group[group["base_valid"]]
        certified = group["method_certified"].eq(True)
        rows.append(
            {
                "base_model_id": str(base_model_id),
                "apas_delta": float(group["apas_delta"].iloc[0]),
                **generation_metrics(group),
                "certified_n": int(certified.sum()),
                "certified_coverage": float(certified.mean()),
                "mean_runtime_seconds": float(group["runtime_seconds"].mean()),
                "mean_l1_robust_scale": (
                    float(generated_valid["l1_robust_scale_mean"].mean())
                    if len(generated_valid)
                    else None
                ),
            }
        )
    return rows


def seed_family_survival_summary(
    survival: pd.DataFrame,
) -> list[dict[str, object]]:
    """Summarize changed-model validity by base model and change family."""

    rows: list[dict[str, object]] = []
    for (base_model_id, family), group in survival.groupby(
        ["base_model_id", "change_family"], sort=True
    ):
        eligible = group["conditional_eligible"].astype(bool)
        base_valid = group["base_valid"].astype(bool)
        changed_valid = group["ce_valid"].eq(True)
        rows.append(
            {
                "base_model_id": str(base_model_id),
                "change_family": str(family),
                "changed_models_n": int(group["change_id"].nunique()),
                "conditional_eligible_n": int(eligible.sum()),
                "pooled_conditional_survival": (
                    float(group.loc[eligible, "conditional_survival"].eq(True).mean())
                    if eligible.any()
                    else None
                ),
                "changed_validity_given_base_valid": (
                    float(changed_valid.sum() / base_valid.sum())
                    if base_valid.any()
                    else None
                ),
                "end_to_end_changed_validity": float(changed_valid.mean()),
            }
        )
    return rows
