import argparse
from pathlib import Path

import pandas as pd

from robustness_benchmark.core.provenance import write_json

GENERATED_STATUSES = frozenset({"success", "base_invalid"})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def generation_metrics(group: pd.DataFrame) -> dict[str, int | float | None]:
    """Separate candidate production from validity on the generation model."""

    generated = group["generation_status"].isin(GENERATED_STATUSES)
    requested_n = len(group)
    generated_n = int(generated.sum())
    base_valid_n = int(group["base_valid"].sum())
    return {
        "requested_n": requested_n,
        "generated_n": generated_n,
        "generation_coverage": _ratio(generated_n, requested_n),
        "base_valid_n": base_valid_n,
        "validity_given_generated": _ratio(base_valid_n, generated_n),
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


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    generation = pd.concat(
        [pd.read_parquet(path / "generation.parquet") for path in args.inputs],
        ignore_index=True,
    )
    survival = pd.concat(
        [pd.read_parquet(path / "survival.parquet") for path in args.inputs],
        ignore_index=True,
    )
    survival = normalize_survival_schema(survival)
    factuals = pd.concat(
        [pd.read_parquet(path / "factuals.parquet") for path in args.inputs],
        ignore_index=True,
    )
    counterfactuals = pd.concat(
        [pd.read_parquet(path / "counterfactuals.parquet") for path in args.inputs],
        ignore_index=True,
    )

    generation.to_parquet(args.output / "generation.parquet", index=False)
    survival.to_parquet(args.output / "survival.parquet", index=False)
    factuals.to_parquet(args.output / "factuals.parquet", index=False)
    counterfactuals.to_parquet(args.output / "counterfactuals.parquet", index=False)

    generation_summary = []
    for values, group in generation.groupby(["base_model_id", "method"], sort=True):
        if not isinstance(values, tuple) or len(values) != 2:
            raise RuntimeError(f"Unexpected generation group key: {values!r}")
        base_model_id, method = str(values[0]), str(values[1])
        generation_summary.append(
            {
                "base_model_id": base_model_id,
                "method": method,
                **generation_metrics(group),
            }
        )

    write_json(
        args.output / "summary.json",
        {
            "input_runs": [str(path.resolve()) for path in args.inputs],
            "base_model_ids": sorted(survival["base_model_id"].unique().tolist()),
            "generation": generation_summary,
            "survival": aggregate_survival(survival),
        },
    )


if __name__ == "__main__":
    main()
