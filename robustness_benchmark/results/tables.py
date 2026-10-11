"""Metric tables of the results packet, as CSV and markdown."""

from pathlib import Path

import numpy as np
import pandas as pd

from robustness_benchmark.results.labels import (
    METHOD_ORDER,
    MODEL_TYPE_LABELS,
    method_label,
    ordered,
)

QUALITY_SUMMARY_COLUMNS = (
    "mean_n_changed_features",
    "mean_knn_distance",
    "mean_knn_distance_relative",
    "lof_inlier_rate",
    "mean_lof_score",
    "mean_ynn",
)


TABLE_COLUMNS = {
    "generation_coverage": ("Coverage", "{:.1%}"),
    "validity_given_generated": ("Base validity", "{:.1%}"),
    "validity_before_projection_given_generated": (
        "Base validity before snapping",
        "{:.1%}",
    ),
    "pooled_conditional_survival": ("Conditional survival", "{:.1%}"),
    "end_to_end_changed_validity": ("End-to-end changed validity", "{:.1%}"),
    "changed_validity_given_base_valid": ("Empirical robustness", "{:.1%}"),
    "mean_gower_distance": ("Gower distance", "{:.3f}"),
    "mean_runtime_seconds": ("Seconds/CFE", "{:.3f}"),
    "mean_n_changed_features": ("Changed features", "{:.2f}"),
    "mean_knn_distance": ("kNN distance", "{:.3f}"),
    "mean_knn_distance_relative": ("Relative kNN distance", "{:.2f}"),
    "lof_inlier_rate": ("LOF inlier rate", "{:.1%}"),
    "mean_lof_score": ("LOF score", "{:.2f}"),
    "mean_ynn": ("yNN", "{:.2f}"),
}


def format_column(values: pd.Series, template: str) -> pd.Series:
    """Format numbers, showing n/a for values a run did not record."""

    return pd.to_numeric(values, errors="coerce").map(
        lambda value: "n/a" if pd.isna(value) else template.format(value)
    )


def save_tables(
    output: Path,
    overall: pd.DataFrame,
    family: pd.DataFrame,
    seeds: pd.DataFrame,
    intervals: pd.DataFrame,
) -> None:
    columns = [
        "dataset",
        "model_type",
        "method",
        "requested_n",
        "generated_n",
        "generation_coverage",
        "validity_given_generated",
        "validity_before_projection_given_generated",
        "certified_n",
        "certified_coverage",
        "end_to_end_validity",
        "pooled_conditional_survival",
        "end_to_end_changed_validity",
        "changed_validity_given_base_valid",
        "mean_gower_distance",
        "mean_runtime_seconds",
        "p10_changed_model_conditional_survival",
        "min_changed_model_conditional_survival",
        *QUALITY_SUMMARY_COLUMNS,
    ]
    overall[columns].to_csv(output / "overall_metrics.csv", index=False)
    availability = method_availability(overall)
    availability.to_csv(output / "method_availability.csv")
    (output / "method_availability.md").write_text(
        availability.rename(index=method_label, columns=MODEL_TYPE_LABELS).to_markdown()
        + "\n"
    )
    family.to_csv(output / "change_family_metrics.csv", index=False)
    seeds.to_csv(output / "seed_metrics.csv", index=False)
    intervals.to_csv(output / "seed_summary.csv", index=False)

    display = overall[["dataset_label", "model_type", "method", *TABLE_COLUMNS]].copy()
    display["method"] = display["method"].map(method_label)
    display["model_type"] = display["model_type"].map(MODEL_TYPE_LABELS)
    for column, (_, template) in TABLE_COLUMNS.items():
        display[column] = format_column(display[column], template)
    display = display.rename(
        columns={
            "dataset_label": "Dataset",
            "model_type": "Model",
            "method": "Method",
            **{column: header for column, (header, _) in TABLE_COLUMNS.items()},
        }
    )
    (output / "overall_metrics.md").write_text(display.to_markdown(index=False) + "\n")


def save_runtime_table(output: Path, runtime: pd.DataFrame) -> None:
    runtime.to_csv(output / "runtime_metrics.csv", index=False)
    display = runtime.copy()
    display["method"] = display["method"].map(method_label)
    display["model_type"] = display["model_type"].map(MODEL_TYPE_LABELS)
    for column in (
        "mean_seconds",
        "standard_deviation_seconds",
        "median_seconds",
        "p90_seconds",
        "maximum_seconds",
    ):
        display[column] = display[column].map(lambda value: f"{value:.4f}")
    display = display.rename(
        columns={
            "model_type": "Model",
            "method": "Method",
            "attempts_n": "Attempts",
            "mean_seconds": "Mean seconds",
            "standard_deviation_seconds": "SD seconds",
            "median_seconds": "Median seconds",
            "p90_seconds": "P90 seconds",
            "maximum_seconds": "Maximum seconds",
            "timeouts_n": "Timeouts",
        }
    )
    (output / "runtime_metrics.md").write_text(display.to_markdown(index=False) + "\n")


def method_availability(overall: pd.DataFrame) -> pd.DataFrame:
    """Method x model-type table: datasets run, or n/a where never run."""

    counts = overall.groupby(["method", "model_type"])["dataset"].nunique()
    table = counts.unstack().reindex(
        index=ordered(overall["method"], METHOD_ORDER),
        columns=ordered(overall["model_type"], tuple(MODEL_TYPE_LABELS)),
    )
    return table.apply(
        lambda column: column.map(
            lambda value: "n/a" if pd.isna(value) else f"{int(value)} datasets"
        )
    )


CROSS_CLASS_METRICS = {
    "empirical_robustness": "changed_validity_given_base_valid",
    "end_to_end_robustness": "end_to_end_changed_validity",
    "distance": "mean_gower_distance",
}


def cross_class_summary(overall: pd.DataFrame) -> pd.DataFrame:
    """Per-method means over datasets, one column per model type and metric.

    Cells stay NaN for methods a model class never ran.
    """

    model_types = ordered(overall["model_type"], tuple(MODEL_TYPE_LABELS))
    numeric = overall.assign(
        **{
            column: pd.to_numeric(overall[column], errors="coerce")
            for column in CROSS_CLASS_METRICS.values()
        }
    )
    means = (
        numeric.groupby(["method", "model_type"])[list(CROSS_CLASS_METRICS.values())]
        .mean()
        .rename(columns={column: name for name, column in CROSS_CLASS_METRICS.items()})
        .unstack("model_type")
    )
    means.columns = [f"{model_type}_{name}" for name, model_type in means.columns]
    return means.reindex(
        index=ordered(overall["method"], METHOD_ORDER),
        columns=[
            f"{model_type}_{name}"
            for model_type in model_types
            for name in CROSS_CLASS_METRICS
        ],
    )


def cross_class_markdown(summary: pd.DataFrame) -> str:
    """Methods by model type; each cell is robustness / end-to-end / distance."""

    model_types = [
        model_type
        for model_type in MODEL_TYPE_LABELS
        if f"{model_type}_distance" in summary.columns
    ]

    def cell(row: pd.Series, model_type: str) -> str:
        values = [row[f"{model_type}_{name}"] for name in CROSS_CLASS_METRICS]
        if all(pd.isna(value) for value in values):
            return "n/a"
        return " / ".join(
            "n/a" if pd.isna(value) else template.format(value)
            for value, template in zip(
                values, ("{:.1%}", "{:.1%}", "{:.3f}"), strict=True
            )
        )

    table = pd.DataFrame(
        {
            MODEL_TYPE_LABELS[model_type]: summary.apply(
                cell, axis=1, model_type=model_type
            )
            for model_type in model_types
        }
    ).rename_axis("Method")
    table.index = table.index.map(method_label)
    return (
        "Cells: empirical robustness / end-to-end robustness / Gower distance, "
        "averaged over datasets.\n\n" + table.to_markdown() + "\n"
    )


def seed_summary(seeds: pd.DataFrame) -> pd.DataFrame:
    """Mean and standard deviation of each metric over the base-model seeds."""

    metrics = (
        "generation_coverage",
        "validity_given_generated",
        "validity_before_projection_given_generated",
        "end_to_end_validity",
        "pooled_conditional_survival",
        "end_to_end_changed_validity",
        "changed_validity_given_base_valid",
        "mean_gower_distance",
        *QUALITY_SUMMARY_COLUMNS,
    )
    rows = []
    for (dataset, model_type), dataset_group in seeds.groupby(
        ["dataset", "model_type"], sort=True
    ):
        seed_order = sorted(dataset_group["seed"].unique())
        for method, group in dataset_group.groupby("method", sort=True):
            aligned = group.set_index("seed").loc[seed_order]
            for metric in metrics:
                values = aligned[metric].to_numpy(dtype=float)
                values = values[np.isfinite(values)]
                rows.append(
                    {
                        "dataset": dataset,
                        "model_type": model_type,
                        "method": method,
                        "metric": metric,
                        "seeds_n": len(values),
                        "total_seeds_n": len(seed_order),
                        "estimate": float(values.mean()) if len(values) else np.nan,
                        "seed_sd": (
                            float(values.std(ddof=1)) if len(values) > 1 else np.nan
                        ),
                    }
                )
    return pd.DataFrame(rows)
