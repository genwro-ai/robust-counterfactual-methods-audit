import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

from robustness_benchmark.core.protocol import BASE_SEEDS, result_dir
from robustness_benchmark.evaluation.aggregate import (
    aggregate_survival,
    generation_metrics,
)
from robustness_benchmark.evaluation.figure_style import (
    AXIS,
    EMPTY,
    INK,
    MARGIN_LABEL,
    MUTED,
    ROBUSTNESS_CMAP,
    SECONDARY,
    SMALL_SIZE,
    SURFACE,
    legend_methods,
    mark_method_ticks,
    method_handle,
    method_style,
    panel_title,
    save_figure,
    scatter_method,
    style_axis,
    styled,
)
from robustness_benchmark.evaluation.plausibility import plausibility_summary
from robustness_benchmark.evaluation.rank_tests import (
    InsufficientData,
    RankTestResult,
    plot_critical_difference,
    rank_test,
)

MARGIN_METHODS = (
    "growing_spheres",
    "gs_margin_60",
    "gs_margin_70",
    "gs_margin_80",
    "gs_margin_90",
)
MARGIN_BASELINES = MARGIN_METHODS[1:]

DATASETS = {
    "breast_cancer": "Breast cancer",
    "diabetes": "Diabetes",
    "wine_quality": "Wine quality",
    "heloc": "HELOC",
    "german_credit": "German credit",
    "adult": "Adult",
    "taiwan_default": "Taiwan default",
}
MODEL_TYPES = {
    "mlp": "MLP",
    "logistic": "Logistic regression",
    "xgboost": "XGBoost",
}
METHOD_ORDER = (
    "wachter",
    "apas",
    "rbr",
    "kdtree",
    "rnce",
    "betarce",
    "roar_lime",
    "robx_balanced",
    "robx_robust",
    "trex",
    "roar",
    *MARGIN_METHODS,
)
FAMILY_ORDER = (
    "architecture",
    "bootstrap",
    "seed",
    "training_config",
    "label_update",
    "deletion",
    "data_addition",
    "bounded_parameter",
    "incremental_update",
)
FAMILY_LABELS = {
    "architecture": "Architecture",
    "bootstrap": "Bootstrap",
    "seed": "New seed",
    "training_config": "Training config",
    "label_update": "Label update",
    "deletion": "Deletion",
    "data_addition": "Data addition",
    "bounded_parameter": "Parameter perturb.",
    "incremental_update": "Incremental update",
}
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
CROSS_CLASS = "cross_class"
SCOPE_LABELS = {**MODEL_TYPES, CROSS_CLASS: "Cross-class"}


@dataclass(frozen=True)
class RankAnalysis:
    """One rank-test metric; ``by_family`` metrics also get per-family analyses."""

    name: str
    label: str
    column: str
    higher_is_better: bool
    by_family: bool


RANK_ANALYSES = (
    RankAnalysis(
        "empirical_robustness",
        "Empirical robustness",
        "changed_validity_given_base_valid",
        True,
        True,
    ),
    RankAnalysis(
        "end_to_end_robustness",
        "End-to-end robustness",
        "end_to_end_changed_validity",
        True,
        True,
    ),
    RankAnalysis("distance", "Gower distance", "mean_gower_distance", False, False),
)


def ordered(values, order: tuple[str, ...]) -> list[str]:
    """Return ``values`` in ``order``, followed by any unknown values sorted."""

    present = set(values)
    return [value for value in order if value in present] + sorted(
        present.difference(order)
    )


def method_label(method: str) -> str:
    return method_style(method).label


def discover_runs(artifacts: Path) -> list[tuple[str, str, Path]]:
    """Return (dataset, model type, result root) for every benchmark present."""

    runs = []
    for model_type in MODEL_TYPES:
        for dataset in DATASETS:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/results_packet"))
    return parser.parse_args()


def load_dataset_results(root: Path, dataset: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    generation = pd.read_parquet(root / "generation.parquet")
    survival = pd.read_parquet(root / "survival.parquet")
    expected_seeds = set(BASE_SEEDS)
    generation_seeds = {
        int(value.rsplit("_", maxsplit=1)[-1])
        for value in generation["base_model_id"].unique()
    }
    survival_seeds = {
        int(value.rsplit("_", maxsplit=1)[-1])
        for value in survival["base_model_id"].unique()
    }
    if generation_seeds != expected_seeds or survival_seeds != expected_seeds:
        raise RuntimeError(
            f"{root} does not contain the expected seeds {sorted(expected_seeds)}: "
            f"generation={sorted(generation_seeds)}, "
            f"survival={sorted(survival_seeds)}"
        )
    return generation, survival


def summarize_generation(generation: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, group in generation.groupby("method", sort=True):
        valid_cost = group.loc[group["base_valid"], "gower_distance"]
        certification_reported = group["method_certified"].notna().any()
        certified = (
            group["method_certified"].eq(True) if certification_reported else None
        )
        rows.append(
            {
                "method": str(method),
                **generation_metrics(group),
                "mean_runtime_seconds": float(group["runtime_seconds"].mean()),
                "mean_gower_distance": float(valid_cost.mean()),
                "certified_n": int(certified.sum()) if certified is not None else None,
                "certified_coverage": (
                    float(certified.mean()) if certified is not None else None
                ),
                **plausibility_summary(group[group["base_valid"]]),
            }
        )
    return pd.DataFrame(rows)


def summarize_overall_survival(survival: pd.DataFrame) -> pd.DataFrame:
    rows = []
    eligible_rows = survival[survival["conditional_eligible"]].copy()
    eligible_rows["survived"] = eligible_rows["conditional_survival"].eq(True)
    per_changed_model = (
        eligible_rows.groupby(["method", "base_model_id", "change_id"], sort=True)[
            "survived"
        ]
        .mean()
        .rename("survival")
        .reset_index()
    )
    for method, group in survival.groupby("method", sort=True):
        eligible = group["conditional_eligible"].astype(bool)
        base_valid = group["base_valid"].astype(bool)
        changed_valid = group["ce_valid"].eq(True)
        model_values = per_changed_model.loc[
            per_changed_model["method"].eq(method), "survival"
        ]
        rows.append(
            {
                "method": str(method),
                "changed_models_n": int(
                    group[["base_model_id", "change_id"]].drop_duplicates().shape[0]
                ),
                "eligible_changed_models_n": len(model_values),
                "eligible_pairs_n": int(eligible.sum()),
                "base_valid_pairs_n": int(base_valid.sum()),
                "base_validity_rate": float(base_valid.mean()),
                "changed_valid_pairs_n": int(changed_valid.sum()),
                "changed_validity_given_base_valid": (
                    float(changed_valid.sum() / base_valid.sum())
                    if base_valid.any()
                    else None
                ),
                "end_to_end_changed_validity": float(changed_valid.mean()),
                "pooled_conditional_survival": (
                    float(group.loc[eligible, "conditional_survival"].eq(True).mean())
                    if eligible.any()
                    else None
                ),
                "mean_changed_model_conditional_survival": float(model_values.mean()),
                "median_changed_model_conditional_survival": float(
                    model_values.median()
                ),
                "p10_changed_model_conditional_survival": float(
                    model_values.quantile(0.1)
                ),
                "min_changed_model_conditional_survival": float(model_values.min()),
            }
        )
    return pd.DataFrame(rows)


def load_results(
    runs: list[tuple[str, str, Path]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    overall_frames: list[pd.DataFrame] = []
    family_frames: list[pd.DataFrame] = []
    for dataset, model_type, root in runs:
        generation_rows, survival_rows = load_dataset_results(root, dataset)
        generation = summarize_generation(generation_rows)
        survival = summarize_overall_survival(survival_rows)
        overall = generation.merge(survival, on="method", validate="one_to_one")
        family = pd.DataFrame(aggregate_survival(survival_rows))
        for frame in (overall, family):
            frame.insert(0, "dataset", dataset)
            frame.insert(1, "dataset_label", DATASETS[dataset])
            frame.insert(2, "model_type", model_type)
        overall_frames.append(overall)
        family_frames.append(family)
    return (
        pd.concat(overall_frames, ignore_index=True),
        pd.concat(family_frames, ignore_index=True),
    )


def runtime_metrics(runs: list[tuple[str, str, Path]]) -> pd.DataFrame:
    """Summarize observed per-factual generation time across all datasets."""

    frames: list[pd.DataFrame] = []
    for _, model_type, root in runs:
        generation = pd.read_parquet(root / "generation.parquet")
        frame = generation[["method", "runtime_seconds"]].copy()
        frame["model_type"] = model_type
        frame["timed_out"] = generation.get(
            "apas_outcome", pd.Series(index=generation.index, dtype=object)
        ).eq("timeout")
        frames.append(frame)

    runtime = pd.concat(frames, ignore_index=True)
    rows: list[dict[str, object]] = []
    for (model_type, method), group in runtime.groupby(
        ["model_type", "method"], sort=True
    ):
        seconds = group["runtime_seconds"]
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
                "timeouts_n": int(group["timed_out"].sum()),
            }
        )
    return pd.DataFrame(rows)


def seed_metrics(runs: list[tuple[str, str, Path]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    generated_statuses = {"success", "base_invalid"}
    for dataset, model_type, root in runs:
        label = DATASETS[dataset]
        for generation_path in sorted(
            (root / "runs").glob("seed_*/*/generation.parquet")
        ):
            run_dir = generation_path.parent
            seed = int(run_dir.parent.name.removeprefix("seed_"))
            method = run_dir.name
            generation = pd.read_parquet(generation_path)
            survival = pd.read_parquet(run_dir / "survival.parquet")
            generated = generation["generation_status"].isin(generated_statuses)
            eligible = survival["conditional_eligible"].astype(bool)
            certification_reported = (
                "method_certified" in generation
                and generation["method_certified"].notna().any()
            )
            rows.append(
                {
                    "dataset": dataset,
                    "dataset_label": label,
                    "model_type": model_type,
                    "seed": seed,
                    "method": method,
                    "method_label": method_label(method),
                    "requested_n": len(generation),
                    "generation_coverage": float(generated.mean()),
                    "validity_given_generated": (
                        float(generation.loc[generated, "base_valid"].mean())
                        if generated.any()
                        else None
                    ),
                    "validity_before_projection_given_generated": generation_metrics(
                        generation
                    )["validity_before_projection_given_generated"],
                    "certified_coverage": (
                        float(generation["method_certified"].eq(True).mean())
                        if certification_reported
                        else None
                    ),
                    "end_to_end_validity": float(generation["base_valid"].mean()),
                    "pooled_conditional_survival": (
                        float(survival.loc[eligible, "conditional_survival"].mean())
                        if eligible.any()
                        else None
                    ),
                    "end_to_end_changed_validity": float(
                        survival["ce_valid"].eq(True).mean()
                    ),
                    "changed_validity_given_base_valid": (
                        float(
                            survival["ce_valid"].eq(True).sum()
                            / survival["base_valid"].sum()
                        )
                        if survival["base_valid"].any()
                        else None
                    ),
                    "mean_gower_distance": float(
                        generation.loc[
                            generation["base_valid"], "gower_distance"
                        ].mean()
                    ),
                    "mean_runtime_seconds": float(generation["runtime_seconds"].mean()),
                    **plausibility_summary(generation[generation["base_valid"]]),
                }
            )
    return pd.DataFrame(rows)


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
        availability.rename(index=method_label, columns=MODEL_TYPES).to_markdown()
        + "\n"
    )
    family.to_csv(output / "change_family_metrics.csv", index=False)
    seeds.to_csv(output / "seed_metrics.csv", index=False)
    intervals.to_csv(output / "seed_summary.csv", index=False)

    display = overall[["dataset_label", "model_type", "method", *TABLE_COLUMNS]].copy()
    display["method"] = display["method"].map(method_label)
    display["model_type"] = display["model_type"].map(MODEL_TYPES)
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
    display["model_type"] = display["model_type"].map(MODEL_TYPES)
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
        columns=ordered(overall["model_type"], tuple(MODEL_TYPES)),
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

    model_types = ordered(overall["model_type"], tuple(MODEL_TYPES))
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
        for model_type in MODEL_TYPES
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
            MODEL_TYPES[model_type]: summary.apply(cell, axis=1, model_type=model_type)
            for model_type in model_types
        }
    ).rename_axis("Method")
    table.index = table.index.map(method_label)
    return (
        "Cells: empirical robustness / end-to-end robustness / Gower distance, "
        "averaged over datasets.\n\n" + table.to_markdown() + "\n"
    )


def rank_test_methods(overall: pd.DataFrame, model_types: list[str]) -> list[str]:
    """Methods run on every model type in scope; margin baselines are excluded."""

    runs = [
        set(overall.loc[overall["model_type"].eq(model_type), "method"])
        for model_type in model_types
    ]
    return ordered(set.intersection(*runs).difference(MARGIN_BASELINES), METHOD_ORDER)


RankKey = tuple[str, str, str]  # (analysis, scope, change family or ALL_FAMILIES)
ALL_FAMILIES = "all"


def run_rank_tests(
    overall: pd.DataFrame, family: pd.DataFrame
) -> dict[RankKey, RankTestResult]:
    """Rank tests with datasets as the independent blocks.

    Families within a dataset share base models and counterfactuals, so they
    are not independent blocks (Demšar 2006). Three analyses:
    - main, per model type: each method's score averaged over change families;
    - per change family and model type (appendix);
    - cross-class: blocks are dataset x model type. Datasets are shared, so this
      one is descriptive.
    Analyses without enough methods or complete blocks are left out.
    """

    model_types = ordered(overall["model_type"], tuple(MODEL_TYPES))
    scopes = {model_type: [model_type] for model_type in model_types}
    if len(model_types) > 1:
        scopes[CROSS_CLASS] = model_types
    results: dict[RankKey, RankTestResult] = {}

    def test(key: RankKey, rows: pd.DataFrame, analysis: RankAnalysis, block) -> None:
        try:
            results[key] = rank_test(
                rows,
                block=block,
                method="method",
                score=analysis.column,
                higher_is_better=analysis.higher_is_better,
            )
        except InsufficientData:
            pass

    for analysis in RANK_ANALYSES:
        dataset_level = (
            family.groupby(["dataset", "model_type", "method"], as_index=False)[
                analysis.column
            ].mean()
            if analysis.by_family
            else overall
        )
        for scope, scope_types in scopes.items():
            methods = rank_test_methods(overall, scope_types)
            rows = dataset_level[
                dataset_level["model_type"].isin(scope_types)
                & dataset_level["method"].isin(methods)
            ]
            block = ["dataset", "model_type"] if scope == CROSS_CLASS else "dataset"
            test((analysis.name, scope, ALL_FAMILIES), rows, analysis, block)
            if not analysis.by_family or scope == CROSS_CLASS:
                continue
            scoped = family[
                family["model_type"].eq(scope) & family["method"].isin(methods)
            ]
            for change_family in ordered(scoped["change_family"], FAMILY_ORDER):
                test(
                    (analysis.name, scope, change_family),
                    scoped[scoped["change_family"].eq(change_family)],
                    analysis,
                    "dataset",
                )
    return results


def rank_test_table(results: dict[RankKey, RankTestResult]) -> pd.DataFrame:
    """One row per analysis with test statistics and a ``rank_<method>`` column each."""

    methods = ordered(
        {
            method
            for result in results.values()
            for method in result.average_ranks.index
        },
        METHOD_ORDER,
    )
    rows = [
        {
            "analysis": analysis,
            "scope": scope,
            "change_family": change_family,
            "statistic": result.statistic,
            "p_value": result.p_value,
            "n_blocks": result.n_blocks,
            "n_blocks_dropped": result.n_blocks_dropped,
            "k_methods": result.k_methods,
            "critical_difference": result.critical_difference,
            "alpha": result.alpha,
            **{f"rank_{method}": rank for method, rank in result.average_ranks.items()},
        }
        for (analysis, scope, change_family), result in results.items()
    ]
    return pd.DataFrame(rows).reindex(
        columns=[
            "analysis",
            "scope",
            "change_family",
            "statistic",
            "p_value",
            "n_blocks",
            "n_blocks_dropped",
            "k_methods",
            "critical_difference",
            "alpha",
            *(f"rank_{method}" for method in methods),
        ]
    )


def rank_test_markdown(table: pd.DataFrame) -> str:
    rank_columns = [column for column in table if column.startswith("rank_")]
    best = table[rank_columns].idxmin(axis=1).str.removeprefix("rank_")
    best_rank = table[rank_columns].min(axis=1)
    analysis_labels = {analysis.name: analysis.label for analysis in RANK_ANALYSES}
    display = pd.DataFrame(
        {
            "Analysis": table["analysis"].map(analysis_labels),
            "Scope": table["scope"].map(SCOPE_LABELS),
            "Change family": table["change_family"].map(
                lambda value: (
                    "averaged"
                    if value == ALL_FAMILIES
                    else FAMILY_LABELS.get(value, value)
                )
            ),
            "Blocks (dropped)": table["n_blocks"].astype(str)
            + " ("
            + table["n_blocks_dropped"].astype(str)
            + ")",
            "Methods": table["k_methods"],
            "Friedman chi2": table["statistic"].map("{:.2f}".format),
            "p": table["p_value"].map("{:.3g}".format),
            "CD": table["critical_difference"].map("{:.2f}".format),
            "Best method (average rank)": [
                f"{method_label(method)} ({rank:.2f})"
                for method, rank in zip(best, best_rank, strict=True)
            ],
        }
    )
    return display.to_markdown(index=False)


def rank_diagram_stem(analysis: str, scope: str, change_family: str) -> str:
    suffix = "" if change_family == ALL_FAMILIES else f"_{change_family}"
    return f"critical_difference_{analysis}_{scope}{suffix}"


@styled
def plot_rank_diagrams(output: Path, results: dict[RankKey, RankTestResult]) -> None:
    analysis_labels = {analysis.name: analysis.label for analysis in RANK_ANALYSES}
    for (analysis, scope, change_family), result in results.items():
        family_label = (
            ""
            if change_family == ALL_FAMILIES
            else f", {FAMILY_LABELS.get(change_family, change_family)}"
        )
        figure = plot_critical_difference(
            result,
            title=f"{SCOPE_LABELS[scope]}: {analysis_labels[analysis]}{family_label} "
            f"(N = {result.n_blocks}, Friedman p = {result.p_value:.2g})",
        )
        save_figure(figure, output / rank_diagram_stem(analysis, scope, change_family))


def model_datasets(frame: pd.DataFrame, model_type: str) -> list[tuple[str, str]]:
    present = set(frame.loc[frame["model_type"].eq(model_type), "dataset"])
    return [
        (dataset, label) for dataset, label in DATASETS.items() if dataset in present
    ]


def panel_axes(count: int, row_height: float, key_height: float = 0.55):
    """Two panels per row plus an empty key axis for a legend or colour bar.

    The key takes the spare cell when ``count`` is odd, else a row beneath.
    """

    columns = min(2, count)
    rows = math.ceil(count / columns)
    spare = rows * columns > count
    heights = [row_height] * rows + ([] if spare else [key_height])
    figure = plt.figure(figsize=(7.2, sum(heights)), layout="constrained")
    grid = figure.add_gridspec(len(heights), columns, height_ratios=heights)
    axes = [
        figure.add_subplot(grid[index // columns, index % columns])
        for index in range(count)
    ]
    key = figure.add_subplot(grid[rows - 1, 1] if spare else grid[rows, :])
    key.axis("off")
    return figure, axes, key


def finish_panels(figure, model_type: str, title: str, path: Path) -> None:
    figure.suptitle(
        f"{MODEL_TYPES[model_type]}: {title}", x=0.01, ha="left", fontweight="bold"
    )
    save_figure(figure, path)


def family_matrix(
    family: pd.DataFrame, dataset: str, model_type: str
) -> tuple[pd.DataFrame, list[str]]:
    """Robustness by method and change family, with rows for methods not run.

    Rows cover every heatmap method run on any model class, so methods a model
    class cannot serve appear as n/a; columns are the families of that bank.
    """

    methods = [
        method
        for method in ordered(family["method"], METHOD_ORDER)
        if method not in MARGIN_BASELINES
    ]
    selected = family[
        family["dataset"].eq(dataset) & family["model_type"].eq(model_type)
    ]
    matrix = selected.pivot(
        index="method",
        columns="change_family",
        values="changed_validity_given_base_valid",
    ).reindex(index=methods, columns=ordered(selected["change_family"], FAMILY_ORDER))
    not_run = [method for method in methods if method not in set(selected["method"])]
    return matrix, not_run


def mark_not_run(axis, matrix: pd.DataFrame, not_run: list[str]) -> None:
    for row, method in enumerate(matrix.index):
        if method in not_run:
            for column in range(matrix.shape[1]):
                axis.text(
                    column + 0.5,
                    row + 0.5,
                    "n/a",
                    ha="center",
                    va="center",
                    color=MUTED,
                    fontsize=SMALL_SIZE - 0.5,
                )


HEATMAP_FAMILY_LABELS = {
    **FAMILY_LABELS,
    "training_config": "Train config",
    "data_addition": "Data add.",
    "bounded_parameter": "Param. perturb.",
    "incremental_update": "Incr. update",
}


@styled
def plot_family_heatmaps(output: Path, family: pd.DataFrame, model_type: str) -> None:
    """Empirical robustness by method and change family, one panel per dataset.

    Grey cells have no value: n/a where the model class cannot run the method.
    """

    datasets = model_datasets(family, model_type)
    figure, axes, key = panel_axes(len(datasets), 2.75)
    for index, (dataset, dataset_label) in enumerate(datasets):
        axis = axes[index]
        matrix, not_run = family_matrix(family, dataset, model_type)
        sns.heatmap(
            100 * matrix.rename(columns=HEATMAP_FAMILY_LABELS),
            ax=axis,
            annot=True,
            annot_kws={"fontsize": SMALL_SIZE},
            fmt=".0f",
            cmap=ROBUSTNESS_CMAP,
            vmin=0,
            vmax=100,
            linewidths=1,
            linecolor=SURFACE,
            cbar=False,
        )
        axis.set_facecolor(EMPTY)
        mark_not_run(axis, matrix, not_run)
        axis.set(xlabel="", ylabel="")
        axis.tick_params(length=0)
        plt.setp(
            axis.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor"
        )
        # rows are the same in every panel, so only the left column names them
        if index % 2:
            axis.set_yticklabels([])
        else:
            axis.set_yticklabels(
                [method_label(method) for method in matrix.index], rotation=0
            )
            mark_method_ticks(axis, list(matrix.index))
        panel_title(axis, index, dataset_label)
    colour_bar = figure.colorbar(
        ScalarMappable(Normalize(0, 100), ROBUSTNESS_CMAP),
        cax=key.inset_axes((0.2, 0.55, 0.6, 0.12)),
        orientation="horizontal",
        ticks=(0, 25, 50, 75, 100),
    )
    colour_bar.outline.set_visible(False)
    colour_bar.ax.tick_params(length=2, color=AXIS)
    colour_bar.set_label("Empirical robustness (%)")
    finish_panels(
        figure,
        model_type,
        "robustness profiles change with the model-update mechanism",
        output / f"paper_robustness_by_change_family_{model_type}",
    )


def tradeoff_point(
    selected: pd.DataFrame, method: str, x_column: str = "mean_gower_distance"
) -> tuple[float, float] | None:
    x = float(selected.loc[method, x_column])
    y = 100 * float(selected.loc[method, "changed_validity_given_base_valid"])
    return (x, y) if np.isfinite(x) and np.isfinite(y) else None


MARGIN_LINE = {"color": MUTED, "linestyle": "--", "linewidth": 1.0, "marker": "."}
FRONTIER_LINE = {"color": AXIS, "linewidth": 1.2}


def plot_margin_baseline(
    axis, selected: pd.DataFrame, x_column: str = "mean_gower_distance"
) -> list[tuple[float, float]]:
    """Draw the confidence-margin baselines as one dashed curve."""

    points = [
        point
        for method in MARGIN_METHODS
        if method in selected.index
        and (point := tradeoff_point(selected, method, x_column)) is not None
    ]
    if points:
        x_values, y_values = zip(*points, strict=True)
        axis.plot(x_values, y_values, zorder=2, **MARGIN_LINE)
    return points


def pareto_frontier(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Points no other point beats on both lower x and higher y."""

    frontier, best_y = [], -np.inf
    for x, y in sorted(points):
        if y > best_y:
            frontier.append((x, y))
            best_y = y
    return frontier


@styled
def plot_tradeoff(
    output: Path,
    overall: pd.DataFrame,
    model_type: str,
    *,
    x_column: str = "mean_gower_distance",
    x_label: str = r"Mean Gower distance  $\leftarrow$ closer",
    title: str = "robustness and proximity form dataset-specific frontiers",
    stem: str = "paper_robustness_proximity_tradeoff",
) -> None:
    """Robustness against a lower-is-better ``x_column``, one panel per dataset."""

    datasets = model_datasets(overall, model_type)
    figure, axes, key = panel_axes(len(datasets), 2.6)
    drawn = set()
    for index, (dataset, dataset_label) in enumerate(datasets):
        axis = axes[index]
        style_axis(axis)
        selected = overall[
            overall["dataset"].eq(dataset) & overall["model_type"].eq(model_type)
        ].set_index("method")
        margin_points = plot_margin_baseline(axis, selected, x_column)
        points = []
        for method in ordered(selected.index, METHOD_ORDER):
            point = tradeoff_point(selected, method, x_column)
            if method in MARGIN_BASELINES or point is None:
                continue
            points.append(point)
            drawn.add(method)
            scatter_method(axis, method, *point, zorder=4)
        if frontier := pareto_frontier(points):
            axis.plot(*zip(*frontier, strict=True), zorder=3, **FRONTIER_LINE)

        all_points = np.array(points + margin_points)
        x_padding = max(0.05, 0.08 * float(np.ptp(all_points[:, 0])))
        axis.set_xlim(
            max(0, all_points[:, 0].min() - x_padding),
            all_points[:, 0].max() + x_padding,
        )
        axis.set_ylim(min(45, all_points[:, 1].min() - 3), 102.5)
        axis.set_xlabel(x_label)
        axis.set_ylabel("Empirical robustness (%)")
        panel_title(axis, index, dataset_label)
    key.legend(
        handles=[method_handle(method) for method in legend_methods(drawn)]
        + [
            Line2D([], [], label=MARGIN_LABEL, **MARGIN_LINE),
            Line2D([], [], label="Pareto frontier", **FRONTIER_LINE),
        ],
        loc="center",
        ncol=2 if len(datasets) % 2 else 5,
        columnspacing=1.2,
        handletextpad=0.4,
    )
    finish_panels(figure, model_type, title, output / f"{stem}_{model_type}")


def plot_plausibility(output: Path, overall: pd.DataFrame, model_type: str) -> None:
    """Robustness against distance to the target-class data."""

    x_column = "mean_knn_distance_relative"
    recorded = overall.assign(
        **{x_column: pd.to_numeric(overall[x_column], errors="coerce")}
    )
    recorded = recorded.dropna(subset=[x_column, "changed_validity_given_base_valid"])
    if not recorded["model_type"].eq(model_type).any():
        return
    plot_tradeoff(
        output,
        recorded,
        model_type,
        x_column=x_column,
        x_label=r"Relative kNN distance  $\leftarrow$ more plausible",
        title="robustness and plausibility",
        stem="paper_robustness_plausibility",
    )


DECOMPOSITION_METRICS = (
    ("generation_coverage", "Coverage", {"marker": "o", "color": SURFACE}),
    ("validity_given_generated", "Base validity", {"marker": "s", "color": MUTED}),
    (
        "changed_validity_given_base_valid",
        "Empirical robustness",
        {"marker": "D", "color": INK},
    ),
)


@styled
def plot_metric_decomposition(
    output: Path, overall: pd.DataFrame, model_type: str
) -> None:
    """Coverage, base validity, and robustness per method, one panel per dataset."""

    datasets = model_datasets(overall, model_type)
    methods = ordered(
        overall.loc[overall["model_type"].eq(model_type), "method"], METHOD_ORDER
    )
    rows = np.arange(len(methods))
    figure, axes, key = panel_axes(len(datasets), 2.75)
    for index, (dataset, dataset_label) in enumerate(datasets):
        axis = axes[index]
        style_axis(axis, grid="x")
        selected = (
            overall[
                overall["dataset"].eq(dataset) & overall["model_type"].eq(model_type)
            ]
            .set_index("method")
            .reindex(methods)
        )
        values = 100 * selected[[column for column, *_ in DECOMPOSITION_METRICS]]
        axis.hlines(
            rows, values.min(axis=1), values.max(axis=1), color=AXIS, linewidth=1.0
        )
        for column, label, look in DECOMPOSITION_METRICS:
            axis.scatter(
                values[column],
                rows,
                s=22,
                edgecolor=SECONDARY,
                linewidth=0.7,
                label=label,
                zorder=3,
                **look,
            )
        axis.set_yticks(rows, [])
        axis.tick_params(axis="y", length=0)
        if index % 2 == 0:  # rows are shared, so only the left column names them
            axis.set_yticklabels([method_label(method) for method in methods])
            mark_method_ticks(axis, methods)
        axis.set_ylim(len(methods) - 0.5, -0.5)
        axis.set_xlim(-3, 103)
        axis.set_xlabel("Rate (%)")
        axis.spines["left"].set_visible(False)
        panel_title(axis, index, dataset_label)
    key.legend(
        *axes[0].get_legend_handles_labels(),
        loc="center",
        ncol=1 if len(datasets) % 2 else 3,
    )
    finish_panels(
        figure,
        model_type,
        "coverage, validity, and robustness answer different questions",
        output / f"paper_metric_decomposition_{model_type}",
    )


def protocol_lines(overall: pd.DataFrame, family: pd.DataFrame) -> list[str]:
    lines = []
    for model_type in ordered(overall["model_type"], tuple(MODEL_TYPES)):
        rows = overall[overall["model_type"].eq(model_type)]
        families = family.loc[family["model_type"].eq(model_type), "change_family"]
        not_run = sorted(set(overall["method"]) - set(rows["method"]))
        lines.append(
            f"- {MODEL_TYPES[model_type]}: {rows['dataset'].nunique()} datasets, "
            f"{families.nunique()} change families, {rows['method'].nunique()} "
            "methods"
            + (f" (n/a: {', '.join(map(method_label, not_run))})" if not_run else "")
            + "."
        )
    return lines


def rank_test_section(table: pd.DataFrame) -> str:
    intro = (
        "Datasets are the independent blocks: change families within a dataset "
        "share base models and counterfactuals. The main analysis ranks methods "
        "per model type on their score averaged over change families; the "
        "per-family analyses (in `rank_tests.csv`) repeat it for each change "
        "family; the cross-class analysis uses dataset x model type blocks and "
        "is descriptive, since datasets are shared. The Friedman test asks "
        "whether average ranks differ, and two methods differ when their average "
        "ranks are further apart than the Nemenyi critical difference at "
        "alpha = 0.05 (Demšar 2006, JMLR 7:1-30). Blocks where a compared method "
        "is missing are dropped. Confidence-margin baselines are excluded."
    )
    if table.empty:
        return f"{intro}\n\nNo analysis had enough methods and complete blocks."
    main = table[table["change_family"].eq(ALL_FAMILIES)]
    analysis_labels = {analysis.name: analysis.label for analysis in RANK_ANALYSES}
    diagrams = "\n".join(
        f"- [{SCOPE_LABELS[scope]}: {analysis_labels[analysis]}]"
        f"({rank_diagram_stem(analysis, scope, ALL_FAMILIES)}.png)"
        for analysis, scope in zip(main["analysis"], main["scope"], strict=True)
    )
    return (
        f"{intro}\n\n{rank_test_markdown(main)}\n\n"
        f"Critical-difference diagrams:\n\n{diagrams}"
    )


def write_brief(
    output: Path,
    overall: pd.DataFrame,
    family: pd.DataFrame,
    cross_class: pd.DataFrame,
    rank_tests: pd.DataFrame,
) -> None:
    averages = (
        overall.groupby(["model_type", "method"], as_index=False)
        .agg(
            empirical_robustness=("changed_validity_given_base_valid", "mean"),
            coverage=("generation_coverage", "mean"),
            base_validity=("validity_given_generated", "mean"),
            distance=("mean_gower_distance", "mean"),
        )
        .sort_values(["model_type", "empirical_robustness"], ascending=[True, False])
    )
    averages["Model"] = averages["model_type"].map(MODEL_TYPES)
    averages["Method"] = averages["method"].map(method_label)
    for column in ("empirical_robustness", "coverage", "base_validity"):
        averages[column] = averages[column].map(lambda value: f"{100 * value:.1f}%")
    averages["distance"] = averages["distance"].map(lambda value: f"{value:.3f}")
    table = averages[
        [
            "Model",
            "Method",
            "coverage",
            "base_validity",
            "empirical_robustness",
            "distance",
        ]
    ].rename(
        columns={
            "coverage": "Mean coverage",
            "base_validity": "Mean base validity",
            "empirical_robustness": "Mean empirical robustness",
            "distance": "Mean Gower distance",
        }
    )
    apas_certification = overall.loc[
        overall["method"].eq("apas"),
        [
            "dataset_label",
            "model_type",
            "generated_n",
            "certified_n",
            "certified_coverage",
        ],
    ].copy()
    apas_certification["model_type"] = apas_certification["model_type"].map(MODEL_TYPES)
    apas_certification["certified_coverage"] = apas_certification[
        "certified_coverage"
    ].map(lambda value: f"{100 * value:.1f}%")
    apas_certification = apas_certification.rename(
        columns={
            "dataset_label": "Dataset",
            "model_type": "Model",
            "generated_n": "Author-sample accepted",
            "certified_n": "Independent holdout certified",
            "certified_coverage": "Certified coverage",
        }
    )
    figures = "\n\n".join(
        f"![{MODEL_TYPES[model_type]}: robustness by change family]"
        f"(paper_robustness_by_change_family_{model_type}.png)\n\n"
        f"![{MODEL_TYPES[model_type]}: coverage, validity, and robustness]"
        f"(paper_metric_decomposition_{model_type}.png)\n\n"
        f"![{MODEL_TYPES[model_type]}: robustness and proximity]"
        f"(paper_robustness_proximity_tradeoff_{model_type}.png)"
        + (
            f"\n\n![{MODEL_TYPES[model_type]}: robustness and plausibility]"
            f"(paper_robustness_plausibility_{model_type}.png)"
            if (output / f"paper_robustness_plausibility_{model_type}.png").is_file()
            else ""
        )
        for model_type in ordered(overall["model_type"], tuple(MODEL_TYPES))
    )
    protocol = "\n".join(protocol_lines(overall, family))
    text = f"""# Robust to which model change?

## Motivation

Counterfactual explanations promise actionable advice, but the model that produced that advice will eventually be retrained or replaced. Existing robust-CFE methods protect against different notions of model change, so a single robustness score can conceal both family-specific failures and failures to generate any counterfactual at all. We therefore test the same methods against controlled families of changed models, measure model difference through prediction behavior, and ask: **robust to which model change, at what coverage and proximity cost?**

## Protocol

{protocol}

Each model class uses five independently trained base models per dataset and 25 held-out variants per change family and base model. Empirical robustness is reported with coverage, base validity, and proximity; seed-wise means and standard deviations use the five base-model seeds. Methods a model class cannot serve are n/a (`method_availability.md`).

## Main results

{figures}

## Cross-dataset headline averages

{table.to_markdown(index=False)}

RobX thresholds are calibrated from the base model's target-class training stability distribution. The balanced and robustness-first settings use its median and upper decile, respectively.

Sparsity and plausibility columns (changed features, kNN distance, LOF, yNN) are in `overall_metrics.md`.

## Cross-class summary

{cross_class_markdown(cross_class)}
## Rank tests

{rank_test_section(rank_tests)}

## APAS certification audit

The author-faithful APAS margin search reuses its model-parameter sample. We
therefore certify the selected CFE once more on an independent sample that
cannot affect selection. Empirical survival continues to use every returned
CFE; `Certified coverage` records only independent holdout successes.

{apas_certification.to_markdown(index=False)}
"""
    (output / "results_brief.md").write_text(text)


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    runs = discover_runs(args.artifacts)
    overall, family = load_results(runs)
    seeds = seed_metrics(runs)
    intervals = seed_summary(seeds)
    runtime = runtime_metrics(runs)
    save_tables(args.output, overall, family, seeds, intervals)
    save_runtime_table(args.output, runtime)
    cross_class = cross_class_summary(overall)
    cross_class.to_csv(args.output / "cross_class_summary.csv", index_label="method")
    (args.output / "cross_class_summary.md").write_text(
        cross_class_markdown(cross_class)
    )
    rank_results = run_rank_tests(overall, family)
    rank_tests = rank_test_table(rank_results)
    rank_tests.to_csv(args.output / "rank_tests.csv", index=False)
    plot_rank_diagrams(args.output, rank_results)
    for model_type in ordered(overall["model_type"], tuple(MODEL_TYPES)):
        plot_family_heatmaps(args.output, family, model_type)
        plot_tradeoff(args.output, overall, model_type)
        plot_plausibility(args.output, overall, model_type)
        plot_metric_decomposition(args.output, overall, model_type)
    write_brief(args.output, overall, family, cross_class, rank_tests)


if __name__ == "__main__":
    main()
