"""Aggregation of per-counterfactual rows into method summaries."""

import pandas as pd

GENERATED_STATUSES = frozenset({"success", "base_invalid"})


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def generation_metrics(group: pd.DataFrame) -> dict[str, int | float | None]:
    """Separate candidate production from validity on the generation model."""

    generated = group["generation_status"].isin(GENERATED_STATUSES)
    requested_n = len(group)
    generated_n = int(generated.sum())
    base_valid_n = int(group["base_valid"].sum())
    # Runs predating the column have no pre-projection validity.
    raw_valid = group.get("base_valid_before_projection")
    raw_valid_n = None if raw_valid is None else int(raw_valid.eq(True).sum())
    return {
        "requested_n": requested_n,
        "generated_n": generated_n,
        "generation_coverage": _ratio(generated_n, requested_n),
        "base_valid_n": base_valid_n,
        "validity_given_generated": _ratio(base_valid_n, generated_n),
        "validity_before_projection_given_generated": (
            None if raw_valid_n is None else _ratio(raw_valid_n, generated_n)
        ),
        "end_to_end_validity": _ratio(base_valid_n, requested_n),
    }


def normalize_survival_schema(survival: pd.DataFrame) -> pd.DataFrame:
    """Rename the legacy field that conflated base validity with generation."""

    normalized = survival.copy()
    if "base_valid" not in normalized:
        if "generation_eligible" not in normalized:
            raise ValueError("Survival rows must include base_valid")
        normalized = normalized.rename(columns={"generation_eligible": "base_valid"})
    elif "generation_eligible" in normalized:
        current = normalized["base_valid"].astype("boolean")
        legacy = normalized["generation_eligible"].astype("boolean")
        normalized["base_valid"] = current.combine_first(legacy)
        normalized = normalized.drop(columns="generation_eligible")
    normalized["base_valid"] = normalized["base_valid"].astype(bool)
    return normalized


def aggregate_survival(survival: pd.DataFrame) -> list[dict[str, object]]:
    survival = normalize_survival_schema(survival)
    rows: list[dict[str, object]] = []
    keys = ["method", "change_family"]
    for values, group in survival.groupby(keys, sort=True):
        if not isinstance(values, tuple) or len(values) != 2:
            raise RuntimeError(f"Unexpected survival group key: {values!r}")
        method, family = str(values[0]), str(values[1])
        total = len(group)
        base_valid = int(group["base_valid"].sum())
        eligible = int(group["conditional_eligible"].sum())
        survived = int(group["conditional_survival"].eq(True).sum())
        changed_valid = int(group["ce_valid"].eq(True).sum())
        per_base = (
            group[group["conditional_eligible"]]
            .groupby("base_model_id")["conditional_survival"]
            .mean()
        )
        per_change = (
            group[group["conditional_eligible"]]
            .groupby(["base_model_id", "change_id"])["conditional_survival"]
            .mean()
        )
        unique_changes = group.groupby(["base_model_id", "change_id"], sort=False)
        rows.append(
            {
                "method": method,
                "change_family": family,
                "base_models_n": int(group["base_model_id"].nunique()),
                "total_n": total,
                "base_valid_pairs_n": base_valid,
                "base_validity_rate": _ratio(base_valid, total),
                "conditional_eligible_n": eligible,
                "factual_eligibility_rate": _ratio(eligible, base_valid),
                "conditional_survived_n": survived,
                "pooled_conditional_survival": _ratio(survived, eligible),
                "changed_valid_pairs_n": changed_valid,
                "changed_validity_given_base_valid": _ratio(changed_valid, base_valid),
                "end_to_end_changed_validity": _ratio(changed_valid, total),
                "mean_base_conditional_survival": (
                    float(per_base.mean()) if len(per_base) else None
                ),
                "min_base_conditional_survival": (
                    float(per_base.min()) if len(per_base) else None
                ),
                "max_base_conditional_survival": (
                    float(per_base.max()) if len(per_base) else None
                ),
                "changed_models_n": int(
                    group[["base_model_id", "change_id"]].drop_duplicates().shape[0]
                ),
                "eligible_changed_models_n": len(per_change),
                "mean_changed_model_conditional_survival": (
                    float(per_change.mean()) if len(per_change) else None
                ),
                "median_changed_model_conditional_survival": (
                    float(per_change.median()) if len(per_change) else None
                ),
                "p10_changed_model_conditional_survival": (
                    float(per_change.quantile(0.1)) if len(per_change) else None
                ),
                "min_changed_model_conditional_survival": (
                    float(per_change.min()) if len(per_change) else None
                ),
                "mean_hard_disagreement": float(
                    unique_changes["hard_disagreement"].first().mean()
                ),
                "mean_probability_mae": float(
                    unique_changes["probability_mae"].first().mean()
                ),
            }
        )
    return rows


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
                "mean_gower_distance": (
                    float(valid_group["gower_distance"].mean())
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
                "base_validity_rate": _ratio(base_valid, total),
                "conditional_eligible_n": eligible,
                "factual_eligibility_rate": _ratio(eligible, base_valid),
                "conditional_survived_n": survived,
                "conditional_survival": _ratio(survived, eligible),
                "changed_model_valid_n": changed_valid,
                "changed_validity_given_base_valid": _ratio(changed_valid, base_valid),
                "end_to_end_changed_validity": _ratio(changed_valid, total),
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
                "mean_gower_distance": (
                    float(generated_valid["gower_distance"].mean())
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
