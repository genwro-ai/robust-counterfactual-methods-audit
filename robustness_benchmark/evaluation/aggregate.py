"""Aggregation of per-counterfactual rows into method summaries.

Generation rows hold one counterfactual each; survival rows one
(counterfactual, changed model) pair. Every summary groups them by ``by``,
``("method",)`` unless a caller needs finer groups.
"""

from collections.abc import Sequence

import pandas as pd

from robustness_benchmark.evaluation.plausibility import plausibility_summary
from robustness_benchmark.evaluation.proximity import proximity_summary

GENERATED_STATUSES = frozenset({"success", "base_invalid"})


def ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def _statistic(values: pd.Series, name: str) -> float | None:
    if not len(values):
        return None
    if name == "p10":
        return float(values.quantile(0.1))
    return float(getattr(values, name)())


def _groups(frame: pd.DataFrame, by: Sequence[str]):
    """``(group values as a dict, group)`` pairs in sorted order."""

    for values, group in frame.groupby(list(by), sort=True):
        keys = values if isinstance(values, tuple) else (values,)
        yield dict(zip(by, keys, strict=True)), group


def generation_metrics(group: pd.DataFrame) -> dict[str, int | float | None]:
    """Separate candidate production from validity on the generation model."""

    generated = group["generation_status"].isin(GENERATED_STATUSES)
    requested_n = len(group)
    generated_n = int(generated.sum())
    base_valid_n = int(group["base_valid"].sum())
    raw_valid_n = int(group["base_valid_before_projection"].eq(True).sum())
    return {
        "requested_n": requested_n,
        "generated_n": generated_n,
        "generation_coverage": ratio(generated_n, requested_n),
        "base_valid_n": base_valid_n,
        "validity_given_generated": ratio(base_valid_n, generated_n),
        "validity_before_projection_given_generated": ratio(raw_valid_n, generated_n),
        "end_to_end_validity": ratio(base_valid_n, requested_n),
    }


def certification_metrics(group: pd.DataFrame) -> dict[str, int | float | None]:
    """Independently certified counterfactuals, for methods that certify."""

    if "method_certified" not in group or group["method_certified"].isna().all():
        return {"certified_n": None, "certified_coverage": None}
    certified = group["method_certified"].eq(True)
    return {
        "certified_n": int(certified.sum()),
        "certified_coverage": float(certified.mean()),
    }


def generation_summary(
    generation: pd.DataFrame, by: Sequence[str] = ("method",)
) -> list[dict[str, object]]:
    """Coverage, validity, cost and runtime of the generated counterfactuals."""

    rows = []
    for keys, group in _groups(generation, by):
        valid = group[group["base_valid"]]
        rows.append(
            {
                **keys,
                **generation_metrics(group),
                "error_n": int(group["generation_error"].notna().sum()),
                "mean_runtime_seconds": float(group["runtime_seconds"].mean()),
                **certification_metrics(group),
                **proximity_summary(valid),
                **plausibility_summary(valid),
            }
        )
    return rows


def survival_metrics(group: pd.DataFrame) -> dict[str, int | float | None]:
    """Validity of base-valid counterfactuals on the changed models."""

    total = len(group)
    base_valid = int(group["base_valid"].sum())
    eligible = int(group["conditional_eligible"].sum())
    survived = int(group["conditional_survival"].eq(True).sum())
    changed_valid = int(group["ce_valid"].eq(True).sum())
    return {
        "total_n": total,
        "base_valid_pairs_n": base_valid,
        "base_validity_rate": ratio(base_valid, total),
        "conditional_eligible_n": eligible,
        "factual_eligibility_rate": ratio(eligible, base_valid),
        "conditional_survived_n": survived,
        "pooled_conditional_survival": ratio(survived, eligible),
        "changed_valid_pairs_n": changed_valid,
        "changed_validity_given_base_valid": ratio(changed_valid, base_valid),
        "end_to_end_changed_validity": ratio(changed_valid, total),
    }


def changed_model_survival(group: pd.DataFrame) -> dict[str, int | float | None]:
    """Distribution of conditional survival over the changed models."""

    eligible = group[group["conditional_eligible"]]
    per_change = (
        eligible["conditional_survival"]
        .eq(True)
        .groupby([eligible["base_model_id"], eligible["change_id"]])
        .mean()
    )
    return {
        "changed_models_n": int(
            group[["base_model_id", "change_id"]].drop_duplicates().shape[0]
        ),
        "eligible_changed_models_n": len(per_change),
        **{
            f"{name}_changed_model_conditional_survival": _statistic(per_change, name)
            for name in ("mean", "median", "p10", "min")
        },
    }


def overall_survival_summary(
    survival: pd.DataFrame, by: Sequence[str] = ("method",)
) -> list[dict[str, object]]:
    """Robustness over every changed model in the bank."""

    return [
        {**keys, **survival_metrics(group), **changed_model_survival(group)}
        for keys, group in _groups(survival, by)
    ]


def family_survival_summary(
    survival: pd.DataFrame, by: Sequence[str] = ("method", "change_family")
) -> list[dict[str, object]]:
    """Robustness per change family, with its spread over base models."""

    rows = []
    for keys, group in _groups(survival, by):
        eligible = group[group["conditional_eligible"]]
        per_base = (
            eligible["conditional_survival"]
            .eq(True)
            .groupby(eligible["base_model_id"])
            .mean()
        )
        unique_changes = group.groupby(["base_model_id", "change_id"], sort=False)
        rows.append(
            {
                **keys,
                "base_models_n": int(group["base_model_id"].nunique()),
                **survival_metrics(group),
                **{
                    f"{name}_base_conditional_survival": _statistic(per_base, name)
                    for name in ("mean", "min", "max")
                },
                **changed_model_survival(group),
                "mean_hard_disagreement": float(
                    unique_changes["hard_disagreement"].first().mean()
                ),
                "mean_probability_mae": float(
                    unique_changes["probability_mae"].first().mean()
                ),
            }
        )
    return rows


def severity_summary(survival: pd.DataFrame) -> list[dict[str, object]]:
    """Robustness per method and relative change severity."""

    return [
        {**keys, **survival_metrics(group)}
        for keys, group in _groups(survival, ("method", "change_severity"))
    ]


def change_survival_summary(survival: pd.DataFrame) -> list[dict[str, object]]:
    """Robustness on each changed model separately."""

    by = ("base_model_id", "method", "change_id", "change_family", "change_level")
    return [
        {
            **keys,
            **survival_metrics(group),
            "hard_disagreement": float(group["hard_disagreement"].iloc[0]),
            "probability_mae": float(group["probability_mae"].iloc[0]),
            "plain_accuracy_drop": float(group["plain_accuracy_drop"].iloc[0]),
        }
        for keys, group in _groups(survival, by)
    ]


def method_summary(
    generation: pd.DataFrame,
    survival: pd.DataFrame,
    by: Sequence[str] = ("method",),
) -> pd.DataFrame:
    """Generation and overall-survival summaries side by side, one row a group."""

    return pd.DataFrame(generation_summary(generation, by)).merge(
        pd.DataFrame(overall_survival_summary(survival, by)),
        on=list(by),
        validate="one_to_one",
    )
