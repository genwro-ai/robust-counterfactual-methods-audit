import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

from robustness_benchmark.evaluation.aggregate import (
    aggregate_survival,
    generation_metrics,
)
from robustness_benchmark.evaluation.plausibility import plausibility_summary
from robustness_benchmark.evaluation.rank_tests import (
    InsufficientData,
    RankTestResult,
    plot_critical_difference,
    rank_test,
)

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})

PAPER_INK = "#172033"
PAPER_MUTED = "#657089"
PAPER_GRID = "#DDE3EC"
PAPER_FACE = "#FBFCFF"
PAPER_METHOD_COLORS = {
    "wachter": "#64748B",
    "apas": "#E76F51",
    "rbr": "#2A9D8F",
    "kdtree": "#4C78A8",
    "rnce": "#8E5BB7",
    "betarce": "#E9A23B",
    "roar_lime": "#D45087",
    "robx_balanced": "#2457C5",
    "robx_robust": "#19A7A0",
    "trex": "#1D3557",
    "roar": "#9B2226",
    "growing_spheres": "#94A3B8",
}
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
PAPER_METHOD_MARKERS = dict(
    zip(
        METHOD_ORDER,
        (
            "o",
            "s",
            "^",
            "v",
            "D",
            "P",
            "X",
            "<",
            ">",
            "h",
            "*",
            ".",
            ".",
            ".",
            ".",
            ".",
        ),
        strict=True,
    )
)
METHOD_LABELS = {
    "wachter": "Wachter",
    "apas": "APΔS",
    "rbr": "RBR",
    "kdtree": "KD-tree",
    "rnce": "RNCE",
    "betarce": "BetaRCE",
    "roar_lime": "ROAR-LIME",
    "robx_balanced": "RobX balanced",
    "robx_robust": "RobX robust-first",
    "trex": "T:Rex",
    "roar": "ROAR",
    "growing_spheres": "Growing spheres",
    **{
        method: f"GS margin 0.{method.removeprefix('gs_margin_')}"
        for method in MARGIN_BASELINES
    },
}
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
    "pooled_conditional_survival": ("Conditional survival", "{:.1%}"),
    "end_to_end_changed_validity": ("End-to-end changed validity", "{:.1%}"),
    "changed_validity_given_base_valid": ("Empirical robustness", "{:.1%}"),
    "mean_l1_robust_scale": ("Scaled L1", "{:.3f}"),
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
    """One rank-test metric; family-level metrics use dataset x family blocks."""

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
    RankAnalysis(
        "distance", "Scaled L1 distance", "mean_l1_robust_scale", False, False
    ),
)


def ordered(values, order: tuple[str, ...]) -> list[str]:
    """Return ``values`` in ``order``, followed by any unknown values sorted."""

    present = set(values)
    return [value for value in order if value in present] + sorted(
        present.difference(order)
    )


def method_label(method: str) -> str:
    return METHOD_LABELS.get(method, method)


def result_root(artifacts: Path, dataset: str, model_type: str) -> Path:
    """MLP results keep the paper's ``full_<dataset>`` directory."""

    legacy = artifacts / f"full_{dataset}"
    if model_type == "mlp" and legacy.is_dir():
        return legacy
    return artifacts / f"full_{dataset}_{model_type}"


def discover_runs(artifacts: Path) -> list[tuple[str, str, Path]]:
    """Return (dataset, model type, result root) for every benchmark present."""

    runs = []
    for model_type in MODEL_TYPES:
        for dataset in DATASETS:
            root = result_root(artifacts, dataset, model_type)
            if not root.is_dir():
                continue
            summary = root / "summary.json"
            recorded = (
                json.loads(summary.read_text()).get("model_type", "mlp")
                if summary.is_file()
                else "mlp"
            )
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
    expected_seeds = {2026, 2027, 2028, 2029, 2030}
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
            f"{root} does not contain the expected five seeds: "
            f"generation={sorted(generation_seeds)}, "
            f"survival={sorted(survival_seeds)}"
        )
    return generation, survival


def summarize_generation(generation: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, group in generation.groupby("method", sort=True):
        valid_cost = group.loc[group["base_valid"], "l1_robust_scale_mean"]
        certification_reported = group["method_certified"].notna().any()
        certified = (
            group["method_certified"].eq(True) if certification_reported else None
        )
        rows.append(
            {
                "method": str(method),
                **generation_metrics(group),
                "mean_runtime_seconds": float(group["runtime_seconds"].mean()),
                "mean_l1_robust_scale": float(valid_cost.mean()),
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
                    "mean_l1_robust_scale": float(
                        generation.loc[
                            generation["base_valid"], "l1_robust_scale_mean"
                        ].mean()
                    ),
                    "mean_runtime_seconds": float(generation["runtime_seconds"].mean()),
                }
            )
    return pd.DataFrame(rows)


def seed_summary(seeds: pd.DataFrame) -> pd.DataFrame:
    """Mean and standard deviation of each metric over the base-model seeds."""

    metrics = (
        "generation_coverage",
        "validity_given_generated",
        "end_to_end_validity",
        "pooled_conditional_survival",
        "end_to_end_changed_validity",
        "changed_validity_given_base_valid",
        "mean_l1_robust_scale",
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
        "certified_n",
        "certified_coverage",
        "end_to_end_validity",
        "pooled_conditional_survival",
        "end_to_end_changed_validity",
        "changed_validity_given_base_valid",
        "mean_l1_robust_scale",
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
    "distance": "mean_l1_robust_scale",
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
        "Cells: empirical robustness / end-to-end robustness / scaled L1, "
        "averaged over datasets.\n\n" + table.to_markdown() + "\n"
    )


def rank_test_methods(overall: pd.DataFrame, model_types: list[str]) -> list[str]:
    """Methods run on every model type in scope; margin baselines are excluded."""

    runs = [
        set(overall.loc[overall["model_type"].eq(model_type), "method"])
        for model_type in model_types
    ]
    return ordered(set.intersection(*runs).difference(MARGIN_BASELINES), METHOD_ORDER)


def run_rank_tests(
    overall: pd.DataFrame, family: pd.DataFrame
) -> dict[tuple[str, str], RankTestResult]:
    """Rank tests per analysis, for each model type and across all of them.

    Blocks are dataset x change family (dataset alone for distance, which has no
    family-level value) and, in the cross-class scope, also model type. Analyses
    without enough methods or complete blocks are left out.
    """

    model_types = ordered(overall["model_type"], tuple(MODEL_TYPES))
    scopes = {model_type: [model_type] for model_type in model_types}
    if len(model_types) > 1:
        scopes[CROSS_CLASS] = model_types
    results = {}
    for analysis in RANK_ANALYSES:
        source = family if analysis.by_family else overall
        block = ["dataset", "model_type"] + (
            ["change_family"] if analysis.by_family else []
        )
        for scope, scope_types in scopes.items():
            methods = rank_test_methods(overall, scope_types)
            rows = source[
                source["model_type"].isin(scope_types) & source["method"].isin(methods)
            ]
            try:
                results[analysis.name, scope] = rank_test(
                    rows,
                    block=block,
                    method="method",
                    score=analysis.column,
                    higher_is_better=analysis.higher_is_better,
                )
            except InsufficientData:
                continue
    return results


def rank_test_table(results: dict[tuple[str, str], RankTestResult]) -> pd.DataFrame:
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
            "statistic": result.statistic,
            "p_value": result.p_value,
            "n_blocks": result.n_blocks,
            "n_blocks_dropped": result.n_blocks_dropped,
            "k_methods": result.k_methods,
            "critical_difference": result.critical_difference,
            "alpha": result.alpha,
            **{f"rank_{method}": rank for method, rank in result.average_ranks.items()},
        }
        for (analysis, scope), result in results.items()
    ]
    return pd.DataFrame(rows).reindex(
        columns=[
            "analysis",
            "scope",
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


def plot_rank_diagrams(
    output: Path, results: dict[tuple[str, str], RankTestResult]
) -> None:
    analysis_labels = {analysis.name: analysis.label for analysis in RANK_ANALYSES}
    for (analysis, scope), result in results.items():
        figure = plot_critical_difference(
            result,
            title=f"{SCOPE_LABELS[scope]}: {analysis_labels[analysis]} "
            f"(N = {result.n_blocks}, Friedman p = {result.p_value:.2g})",
            labels=METHOD_LABELS,
            colors=PAPER_METHOD_COLORS,
            ink=PAPER_INK,
            muted=PAPER_MUTED,
        )
        for suffix, options in (("png", {"dpi": 300}), ("pdf", {})):
            figure.savefig(
                output / f"critical_difference_{analysis}_{scope}.{suffix}",
                bbox_inches="tight",
                pad_inches=0.06,
                **options,
            )
        plt.close(figure)


def model_datasets(frame: pd.DataFrame, model_type: str) -> list[tuple[str, str]]:
    present = set(frame.loc[frame["model_type"].eq(model_type), "dataset"])
    return [
        (dataset, label) for dataset, label in DATASETS.items() if dataset in present
    ]


def panel_axes(count: int, width: float, row_height: float, **kwargs):
    """Two panels per row; unused trailing axes are hidden."""

    columns = min(2, count)
    rows = math.ceil(count / columns)
    figure, axes = plt.subplots(
        rows, columns, figsize=(width, row_height * rows), squeeze=False, **kwargs
    )
    for axis in axes.flat[count:]:
        axis.set_visible(False)
    return figure, list(axes.flat[:count])


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


def mark_not_run(axis, matrix: pd.DataFrame, not_run: list[str], **text_kwargs) -> None:
    for row, method in enumerate(matrix.index):
        if method in not_run:
            for column in range(matrix.shape[1]):
                axis.text(
                    column + 0.5,
                    row + 0.5,
                    "n/a",
                    ha="center",
                    va="center",
                    color=PAPER_MUTED,
                    **text_kwargs,
                )


def plot_family_heatmaps(output: Path, family: pd.DataFrame, model_type: str) -> None:
    sns.set_theme(style="white", context="paper", font_scale=1.05)
    datasets = model_datasets(family, model_type)
    figure, axes = panel_axes(len(datasets), 15.8, 5.6)
    color_axis = figure.add_axes((0.925, 0.18, 0.015, 0.64))
    for index, (dataset, dataset_label) in enumerate(datasets):
        axis = axes[index]
        matrix, not_run = family_matrix(family, dataset, model_type)
        sns.heatmap(
            100
            * matrix.rename(
                index=method_label,
                columns=lambda value: FAMILY_LABELS.get(value, value),
            ),
            ax=axis,
            annot=True,
            fmt=".0f",
            cmap="viridis",
            vmin=0,
            vmax=100,
            linewidths=0.4,
            linecolor="white",
            cbar=index == 0,
            cbar_ax=color_axis if index == 0 else None,
            cbar_kws={"label": "Empirical robustness (%)"},
        )
        mark_not_run(axis, matrix, not_run, fontsize=8)
        axis.set_title(dataset_label, fontweight="bold")
        axis.set_xlabel("")
        axis.set_ylabel("")
        axis.tick_params(axis="x", rotation=35)
        axis.tick_params(axis="y", rotation=0)
    figure.suptitle(
        f"{MODEL_TYPES[model_type]}: empirical robustness depends on how the model changes",
        fontsize=17,
        fontweight="bold",
    )
    figure.subplots_adjust(
        left=0.075,
        right=0.90,
        bottom=0.07,
        top=0.90,
        hspace=0.42,
        wspace=0.24,
    )
    figure.savefig(output / f"robustness_by_change_family_{model_type}.png", dpi=220)
    figure.savefig(output / f"robustness_by_change_family_{model_type}.pdf")
    plt.close(figure)


def plot_paper_family_heatmaps(
    output: Path, family: pd.DataFrame, model_type: str
) -> None:
    sns.set_theme(style="white", context="paper", font="DejaVu Sans")
    cmap = LinearSegmentedColormap.from_list(
        "midnight_sun",
        ("#111A3A", "#3B2D78", "#7B4AA8", "#C35C9A", "#F09A6A", "#F8E7A1"),
    )
    datasets = model_datasets(family, model_type)
    figure, axes = panel_axes(len(datasets), 7.5, 2.68, facecolor="white")
    height = figure.get_figheight()  # margins are fixed in inches for any panel count
    color_axis = figure.add_axes((0.33, 0.35 / height, 0.40, 0.1 / height))
    paper_family_labels = {
        **FAMILY_LABELS,
        "training_config": "Train config",
        "data_addition": "Data add.",
        "bounded_parameter": "Param. perturb.",
        "incremental_update": "Incr. update",
    }
    for index, (dataset, dataset_label) in enumerate(datasets):
        axis = axes[index]
        matrix, not_run = family_matrix(family, dataset, model_type)
        sns.heatmap(
            100
            * matrix.rename(
                index=method_label,
                columns=lambda value: paper_family_labels.get(value, value),
            ),
            ax=axis,
            annot=True,
            annot_kws={"fontsize": 7.0, "fontweight": "bold"},
            fmt=".0f",
            cmap=cmap,
            vmin=0,
            vmax=100,
            linewidths=0.7,
            linecolor="white",
            cbar=index == 0,
            cbar_ax=color_axis if index == 0 else None,
            cbar_kws={
                "label": "Empirical robustness (%)",
                "orientation": "horizontal",
                "ticks": (0, 25, 50, 75, 100),
            },
        )
        annotation_values = (100 * matrix).to_numpy().ravel()
        annotation_values = annotation_values[np.isfinite(annotation_values)]
        for text_item, value in zip(axis.texts, annotation_values, strict=True):
            text_item.set_color("white" if value < 76 else PAPER_INK)
        mark_not_run(axis, matrix, not_run, fontsize=6.0)
        axis.set_title(
            dataset_label,
            fontsize=9.2,
            fontweight="bold",
            color=PAPER_INK,
            loc="left",
            pad=6,
        )
        axis.set_xlabel("")
        axis.set_ylabel("")
        axis.tick_params(
            axis="x", labelsize=6.6, labelcolor=PAPER_MUTED, rotation=32, pad=1
        )
        axis.tick_params(
            axis="y", labelsize=6.8, labelcolor=PAPER_INK, rotation=0, pad=2
        )
        axis.text(
            -0.17,
            1.055,
            chr(ord("A") + index),
            transform=axis.transAxes,
            fontsize=8,
            fontweight="bold",
            color="#C35C9A",
            va="bottom",
        )
    color_axis.tick_params(labelsize=6.5, colors=PAPER_MUTED, length=2)
    color_axis.xaxis.label.set_size(7.2)
    color_axis.xaxis.label.set_color(PAPER_INK)
    figure.suptitle(
        f"{MODEL_TYPES[model_type]}: robustness profiles change with the "
        "model-update mechanism",
        x=0.11,
        y=1 - 0.04 / height,
        ha="left",
        fontsize=10.2,
        fontweight="bold",
        color=PAPER_INK,
    )
    figure.subplots_adjust(
        left=0.11,
        right=0.985,
        bottom=1.02 / height,
        top=1 - 0.43 / height,
        hspace=0.47,
        wspace=0.44,
    )
    figure.savefig(
        output / f"paper_robustness_by_change_family_{model_type}.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.06,
    )
    figure.savefig(
        output / f"paper_robustness_by_change_family_{model_type}.pdf",
        bbox_inches="tight",
        pad_inches=0.06,
    )
    plt.close(figure)


def tradeoff_point(
    selected: pd.DataFrame, method: str, x_column: str = "mean_l1_robust_scale"
) -> tuple[float, float] | None:
    x = float(selected.loc[method, x_column])
    y = 100 * float(selected.loc[method, "changed_validity_given_base_valid"])
    return (x, y) if np.isfinite(x) and np.isfinite(y) else None


def plot_margin_baseline(
    axis, selected: pd.DataFrame, x_column: str = "mean_l1_robust_scale", **line_kwargs
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
        axis.plot(
            x_values,
            y_values,
            linestyle="--",
            marker=".",
            label="Confidence-margin baseline",
            zorder=2,
            **line_kwargs,
        )
    return points


def plot_tradeoff(output: Path, overall: pd.DataFrame, model_type: str) -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)
    palette = dict(
        zip(
            METHOD_ORDER,
            sns.color_palette("colorblind", n_colors=len(METHOD_ORDER)),
            strict=True,
        )
    )
    datasets = model_datasets(overall, model_type)
    figure, axes = panel_axes(len(datasets), 13.8, 5.2, constrained_layout=True)
    offsets = {
        "breast_cancer": {
            "robx_balanced": (4, -11),
            "robx_robust": (4, 7),
            "kdtree": (4, -11),
            "rnce": (4, 7),
            "betarce": (4, -10),
        },
        "diabetes": {
            "robx_balanced": (4, 7),
            "robx_robust": (4, 7),
            "rnce": (4, -11),
            "betarce": (4, 7),
        },
        "wine_quality": {
            "robx_balanced": (4, 7),
            "robx_robust": (4, 7),
            "rnce": (4, -11),
            "betarce": (4, 7),
            "roar_lime": (4, -11),
        },
        "heloc": {
            "robx_balanced": (4, 7),
            "robx_robust": (4, 7),
            "betarce": (4, 7),
            "rnce": (4, -11),
        },
    }
    for index, (dataset, dataset_label) in enumerate(datasets):
        axis = axes[index]
        selected = overall[
            overall["dataset"].eq(dataset) & overall["model_type"].eq(model_type)
        ].set_index("method")
        plot_margin_baseline(axis, selected, color="#94A3B8")
        for method in ordered(selected.index, METHOD_ORDER):
            point = tradeoff_point(selected, method)
            if method in MARGIN_METHODS or point is None:
                continue
            x, y = point
            axis.scatter(x, y, s=55, color=palette.get(method, "black"), zorder=3)
            axis.annotate(
                method_label(method),
                (x, y),
                xytext=offsets.get(dataset, {}).get(method, (4, 4)),
                textcoords="offset points",
                fontsize=8,
            )
        axis.set_title(dataset_label, fontweight="bold")
        axis.set_xlabel("Mean scaled L1 distance")
        axis.set_ylabel("Empirical robustness (%)")
        axis.margins(x=0.08)
        axis.set_ylim(-3, 108)
    figure.suptitle(
        f"{MODEL_TYPES[model_type]}: robustness must be interpreted together "
        "with proximity and coverage",
        fontsize=17,
        fontweight="bold",
    )
    figure.savefig(output / f"robustness_proximity_tradeoff_{model_type}.png", dpi=220)
    figure.savefig(output / f"robustness_proximity_tradeoff_{model_type}.pdf")
    plt.close(figure)


def plot_paper_tradeoff(
    output: Path,
    overall: pd.DataFrame,
    model_type: str,
    *,
    x_column: str = "mean_l1_robust_scale",
    x_label: str = r"Mean scaled $L_1$ distance  $\leftarrow$ closer",
    title: str = "robustness and proximity form dataset-specific frontiers",
    stem: str = "paper_robustness_proximity_tradeoff",
) -> None:
    """Robustness against a lower-is-better ``x_column``, one panel per dataset."""

    sns.set_theme(style="white", context="paper", font="DejaVu Sans")
    datasets = model_datasets(overall, model_type)
    figure, axes = panel_axes(len(datasets), 7.2, 2.72, facecolor="white")
    height = figure.get_figheight()
    for index, (dataset, dataset_label) in enumerate(datasets):
        axis = axes[index]
        axis.set_facecolor(PAPER_FACE)
        selected = overall[
            overall["dataset"].eq(dataset) & overall["model_type"].eq(model_type)
        ].set_index("method")
        margin_points = plot_margin_baseline(
            axis, selected, x_column, color="#94A3B8", linewidth=1.0
        )
        points = []
        for method in ordered(selected.index, METHOD_ORDER):
            point = tradeoff_point(selected, method, x_column)
            if method in MARGIN_METHODS or point is None:
                continue
            x, y = point
            points.append((x, y))
            axis.scatter(
                x,
                y,
                s=52,
                color=PAPER_METHOD_COLORS.get(method, PAPER_INK),
                marker=PAPER_METHOD_MARKERS.get(method, "o"),
                edgecolor="white",
                linewidth=0.8,
                label=method_label(method),
                zorder=4,
            )

        frontier = []
        best_y = -np.inf
        for x, y in sorted(points):
            if y > best_y:
                frontier.append((x, y))
                best_y = y
        if frontier:
            frontier_x, frontier_y = zip(*frontier, strict=True)
            axis.plot(
                frontier_x,
                frontier_y,
                color="#AAB4C5",
                linewidth=1.2,
                zorder=2,
            )

        all_points = points + margin_points
        x_values = np.array([point[0] for point in all_points])
        x_padding = max(0.05, 0.08 * float(np.ptp(x_values)))
        axis.set_xlim(
            max(0, float(x_values.min()) - x_padding), x_values.max() + x_padding
        )
        axis.set_ylim(min(43, min(point[1] for point in all_points) - 3), 102.5)
        axis.set_yticks((50, 60, 70, 80, 90, 100))
        axis.set_title(
            dataset_label,
            fontsize=9.2,
            fontweight="bold",
            color=PAPER_INK,
            loc="left",
            pad=7,
        )
        axis.set_xlabel(x_label, fontsize=7.1)
        axis.set_ylabel("Empirical robustness (%)", fontsize=7.1)
        axis.tick_params(labelsize=6.7, colors=PAPER_MUTED, length=2.5)
        axis.grid(True, color=PAPER_GRID, linewidth=0.7, alpha=0.85)
        axis.set_axisbelow(True)
        for spine in axis.spines.values():
            spine.set_color("#C9D1DE")
            spine.set_linewidth(0.7)
        axis.text(
            -0.15,
            1.055,
            chr(ord("A") + index),
            transform=axis.transAxes,
            fontsize=8,
            fontweight="bold",
            color="#2457C5",
            va="bottom",
        )
    figure.subplots_adjust(
        left=0.10,
        right=0.985,
        bottom=1.03 / height,
        top=1 - 0.54 / height,
        hspace=0.45,
        wspace=0.29,
    )
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=5,
        frameon=False,
        fontsize=7,
        columnspacing=1.1,
        handletextpad=0.4,
        labelcolor=PAPER_INK,
    )
    figure.suptitle(
        f"{MODEL_TYPES[model_type]}: {title}",
        x=0.10,
        y=1 - 0.065 / height,
        ha="left",
        fontsize=10.2,
        fontweight="bold",
        color=PAPER_INK,
    )
    figure.savefig(
        output / f"{stem}_{model_type}.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.06,
    )
    figure.savefig(
        output / f"{stem}_{model_type}.pdf",
        bbox_inches="tight",
        pad_inches=0.06,
    )
    plt.close(figure)


def plot_paper_plausibility(
    output: Path, overall: pd.DataFrame, model_type: str
) -> None:
    """Robustness against distance to the target-class data.

    Runs that predate the plausibility metrics have no values and are skipped.
    """

    x_column = "mean_knn_distance_relative"
    if x_column not in overall:
        return
    recorded = overall.assign(
        **{x_column: pd.to_numeric(overall[x_column], errors="coerce")}
    )
    recorded = recorded.dropna(subset=[x_column, "changed_validity_given_base_valid"])
    if not recorded["model_type"].eq(model_type).any():
        return
    plot_paper_tradeoff(
        output,
        recorded,
        model_type,
        x_column=x_column,
        x_label=r"Relative kNN distance to target-class data  $\leftarrow$ more plausible",
        title="robustness and plausibility",
        stem="paper_robustness_plausibility",
    )


def plot_metric_decomposition(
    output: Path, overall: pd.DataFrame, model_type: str
) -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)
    metrics = (
        ("generation_coverage", "Coverage", "o"),
        ("validity_given_generated", "Base validity", "s"),
        ("changed_validity_given_base_valid", "Empirical robustness", "D"),
    )
    colors = sns.color_palette("colorblind", n_colors=len(metrics))
    datasets = model_datasets(overall, model_type)
    figure, axes = panel_axes(len(datasets), 13.6, 5.1, constrained_layout=True)
    for index, (dataset, dataset_label) in enumerate(datasets):
        axis = axes[index]
        selected = overall[
            overall["dataset"].eq(dataset) & overall["model_type"].eq(model_type)
        ].set_index("method")
        methods = ordered(selected.index, METHOD_ORDER)
        y_positions = np.arange(len(methods))
        for (column, label, marker), color in zip(metrics, colors, strict=True):
            values = 100 * selected.loc[methods, column].to_numpy(dtype=float)
            axis.scatter(
                values,
                y_positions,
                label=label,
                marker=marker,
                color=color,
                s=42,
                zorder=3,
            )
        axis.set_yticks(y_positions, [method_label(method) for method in methods])
        axis.invert_yaxis()
        axis.set_xlim(-3, 103)
        axis.set_xlabel("Rate (%)")
        axis.set_title(dataset_label, fontweight="bold")
        if index == 0:
            axis.legend(loc="lower left", frameon=False, fontsize=8.5)
    figure.suptitle(
        f"{MODEL_TYPES[model_type]}: coverage, validity, and robustness answer "
        "different questions",
        fontsize=17,
        fontweight="bold",
    )
    figure.savefig(output / f"metric_decomposition_{model_type}.png", dpi=220)
    figure.savefig(output / f"metric_decomposition_{model_type}.pdf")
    plt.close(figure)


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
        "Methods are ranked within each block (dataset x change family; dataset "
        "alone for distance; also model type in the cross-class scope), and the "
        "Friedman test asks whether average ranks differ. Two methods differ "
        "when their average ranks are further apart than the Nemenyi critical "
        "difference at alpha = 0.05 (Demšar 2006, JMLR 7:1-30). Blocks where a "
        "compared method is missing are dropped. Confidence-margin baselines "
        "are excluded. Full results are in `rank_tests.csv`."
    )
    if table.empty:
        return f"{intro}\n\nNo analysis had enough methods and complete blocks."
    analysis_labels = {analysis.name: analysis.label for analysis in RANK_ANALYSES}
    diagrams = "\n".join(
        f"- [{SCOPE_LABELS[scope]}: {analysis_labels[analysis]}]"
        f"(critical_difference_{analysis}_{scope}.png)"
        for analysis, scope in zip(table["analysis"], table["scope"], strict=True)
    )
    return f"{intro}\n\n{rank_test_markdown(table)}\n\nCritical-difference diagrams:\n\n{diagrams}"


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
            distance=("mean_l1_robust_scale", "mean"),
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
            "distance": "Mean scaled L1",
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
        f"(robustness_by_change_family_{model_type}.png)\n\n"
        f"![{MODEL_TYPES[model_type]}: coverage, validity, and robustness]"
        f"(metric_decomposition_{model_type}.png)\n\n"
        f"![{MODEL_TYPES[model_type]}: robustness and proximity]"
        f"(robustness_proximity_tradeoff_{model_type}.png)"
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
        plot_paper_family_heatmaps(args.output, family, model_type)
        plot_tradeoff(args.output, overall, model_type)
        plot_paper_tradeoff(args.output, overall, model_type)
        plot_paper_plausibility(args.output, overall, model_type)
        plot_metric_decomposition(args.output, overall, model_type)
    write_brief(args.output, overall, family, cross_class, rank_tests)


if __name__ == "__main__":
    main()
