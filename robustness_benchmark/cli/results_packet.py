import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

from robustness_benchmark.evaluation.aggregate import aggregate_survival, generation_metrics

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
}

DATASETS = {
    "breast_cancer": "Breast cancer",
    "diabetes": "Diabetes",
    "wine_quality": "Wine quality",
    "heloc": "HELOC",
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
)
PAPER_METHOD_MARKERS = dict(
    zip(
        METHOD_ORDER,
        ("o", "s", "^", "v", "D", "P", "X", "<", ">"),
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
}
RESULT_ROOTS = ("full_{dataset}",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/results_packet"))
    return parser.parse_args()


def dataset_result_roots(artifacts: Path, dataset: str) -> list[Path]:
    roots = [artifacts / pattern.format(dataset=dataset) for pattern in RESULT_ROOTS]
    missing = [root for root in roots if not root.is_dir()]
    if missing:
        raise FileNotFoundError(f"Missing result directories: {missing}")
    return roots


def load_dataset_results(
    artifacts: Path, dataset: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    roots = dataset_result_roots(artifacts, dataset)
    generation = pd.concat(
        [pd.read_parquet(root / "generation.parquet") for root in roots],
        ignore_index=True,
    )
    survival = pd.concat(
        [pd.read_parquet(root / "survival.parquet") for root in roots],
        ignore_index=True,
    )
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
            f"{dataset} does not contain the expected five seeds: "
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


def load_results(artifacts: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    overall_frames: list[pd.DataFrame] = []
    family_frames: list[pd.DataFrame] = []
    for dataset, label in DATASETS.items():
        generation_rows, survival_rows = load_dataset_results(artifacts, dataset)
        generation = summarize_generation(generation_rows)
        survival = summarize_overall_survival(survival_rows)
        overall = generation.merge(survival, on="method", validate="one_to_one")
        overall.insert(0, "dataset", dataset)
        overall.insert(1, "dataset_label", label)
        overall_frames.append(overall)

        family = pd.DataFrame(aggregate_survival(survival_rows))
        family.insert(0, "dataset", dataset)
        family.insert(1, "dataset_label", label)
        family_frames.append(family)
    return (
        pd.concat(overall_frames, ignore_index=True),
        pd.concat(family_frames, ignore_index=True),
    )


def runtime_metrics(artifacts: Path) -> pd.DataFrame:
    """Summarize observed per-factual generation time across all datasets."""

    frames: list[pd.DataFrame] = []
    for dataset in DATASETS:
        for root in dataset_result_roots(artifacts, dataset):
            generation = pd.read_parquet(root / "generation.parquet")
            frame = generation[["method", "runtime_seconds"]].copy()
            frame["timed_out"] = generation.get(
                "apas_outcome", pd.Series(index=generation.index, dtype=object)
            ).eq("timeout")
            frames.append(frame)

    runtime = pd.concat(frames, ignore_index=True)
    rows: list[dict[str, object]] = []
    for method, group in runtime.groupby("method", sort=True):
        seconds = group["runtime_seconds"]
        rows.append(
            {
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


def seed_metrics(artifacts: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    generated_statuses = {"success", "base_invalid"}
    for dataset, label in DATASETS.items():
        roots = dataset_result_roots(artifacts, dataset)
        generation_paths = [
            path
            for root in roots
            for path in sorted((root / "runs").glob("seed_*/*/generation.parquet"))
        ]
        for generation_path in generation_paths:
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
                    "seed": seed,
                    "method": method,
                    "method_label": METHOD_LABELS[method],
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
    for dataset, dataset_group in seeds.groupby("dataset", sort=True):
        seed_order = sorted(dataset_group["seed"].unique())
        for method, group in dataset_group.groupby("method", sort=True):
            aligned = group.set_index("seed").loc[seed_order]
            for metric in metrics:
                values = aligned[metric].to_numpy(dtype=float)
                values = values[np.isfinite(values)]
                rows.append(
                    {
                        "dataset": dataset,
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


def save_tables(
    output: Path,
    overall: pd.DataFrame,
    family: pd.DataFrame,
    seeds: pd.DataFrame,
    intervals: pd.DataFrame,
) -> None:
    columns = [
        "dataset",
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
    ]
    overall[columns].to_csv(output / "overall_metrics.csv", index=False)
    family.to_csv(output / "change_family_metrics.csv", index=False)
    seeds.to_csv(output / "seed_metrics.csv", index=False)
    intervals.to_csv(output / "seed_summary.csv", index=False)

    display = overall[
        [
            "dataset_label",
            "method",
            "generation_coverage",
            "validity_given_generated",
            "pooled_conditional_survival",
            "end_to_end_changed_validity",
            "changed_validity_given_base_valid",
            "mean_l1_robust_scale",
            "mean_runtime_seconds",
        ]
    ].copy()
    display["method"] = display["method"].map(METHOD_LABELS)
    for column in (
        "generation_coverage",
        "validity_given_generated",
        "pooled_conditional_survival",
        "end_to_end_changed_validity",
        "changed_validity_given_base_valid",
    ):
        display[column] = display[column].map(lambda value: f"{100 * value:.1f}%")
    display["mean_l1_robust_scale"] = display["mean_l1_robust_scale"].map(
        lambda value: f"{value:.3f}"
    )
    display["mean_runtime_seconds"] = display["mean_runtime_seconds"].map(
        lambda value: f"{value:.3f}"
    )
    display = display.rename(
        columns={
            "dataset_label": "Dataset",
            "method": "Method",
            "generation_coverage": "Coverage",
            "validity_given_generated": "Base validity",
            "pooled_conditional_survival": "Conditional survival",
            "end_to_end_changed_validity": "End-to-end changed validity",
            "changed_validity_given_base_valid": "Empirical robustness",
            "mean_l1_robust_scale": "Scaled L1",
            "mean_runtime_seconds": "Seconds/CFE",
        }
    )
    (output / "overall_metrics.md").write_text(display.to_markdown(index=False) + "\n")


def save_runtime_table(output: Path, runtime: pd.DataFrame) -> None:
    runtime.to_csv(output / "runtime_metrics.csv", index=False)
    display = runtime.copy()
    display["method"] = display["method"].map(METHOD_LABELS)
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


def plot_family_heatmaps(output: Path, family: pd.DataFrame) -> None:
    sns.set_theme(style="white", context="paper", font_scale=1.05)
    figure, axes = plt.subplots(2, 2, figsize=(15.8, 11.2))
    color_axis = figure.add_axes((0.925, 0.18, 0.015, 0.64))
    for index, (dataset, dataset_label) in enumerate(DATASETS.items()):
        axis = axes.flat[index]
        selected = family[family["dataset"] == dataset]
        matrix = selected.pivot(
            index="method",
            columns="change_family",
            values="changed_validity_given_base_valid",
        ).loc[list(METHOD_ORDER), list(FAMILY_ORDER)]
        matrix.index = [METHOD_LABELS[value] for value in matrix.index]
        matrix.columns = [FAMILY_LABELS[value] for value in matrix.columns]
        sns.heatmap(
            100 * matrix,
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
        axis.set_title(dataset_label, fontweight="bold")
        axis.set_xlabel("")
        axis.set_ylabel("")
        axis.tick_params(axis="x", rotation=35)
        axis.tick_params(axis="y", rotation=0)
    figure.suptitle(
        "Empirical robustness depends on how the model changes",
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
    figure.savefig(output / "robustness_by_change_family.png", dpi=220)
    figure.savefig(output / "robustness_by_change_family.pdf")
    plt.close(figure)


def plot_paper_family_heatmaps(output: Path, family: pd.DataFrame) -> None:
    sns.set_theme(style="white", context="paper", font="DejaVu Sans")
    cmap = LinearSegmentedColormap.from_list(
        "midnight_sun",
        ("#111A3A", "#3B2D78", "#7B4AA8", "#C35C9A", "#F09A6A", "#F8E7A1"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(7.5, 5.35), facecolor="white")
    color_axis = figure.add_axes((0.33, 0.065, 0.40, 0.018))
    paper_family_labels = {
        **FAMILY_LABELS,
        "training_config": "Train config",
        "data_addition": "Data add.",
        "bounded_parameter": "Param. perturb.",
    }
    for index, (dataset, dataset_label) in enumerate(DATASETS.items()):
        axis = axes.flat[index]
        selected = family[family["dataset"] == dataset]
        matrix = selected.pivot(
            index="method",
            columns="change_family",
            values="changed_validity_given_base_valid",
        ).loc[list(METHOD_ORDER), list(FAMILY_ORDER)]
        matrix.index = [METHOD_LABELS[value] for value in matrix.index]
        matrix.columns = [paper_family_labels[value] for value in matrix.columns]
        sns.heatmap(
            100 * matrix,
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
        "Robustness profiles change with the model-update mechanism",
        x=0.11,
        y=0.992,
        ha="left",
        fontsize=10.2,
        fontweight="bold",
        color=PAPER_INK,
    )
    figure.subplots_adjust(
        left=0.11,
        right=0.985,
        bottom=0.19,
        top=0.92,
        hspace=0.47,
        wspace=0.44,
    )
    figure.savefig(
        output / "paper_robustness_by_change_family.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.06,
    )
    figure.savefig(
        output / "paper_robustness_by_change_family.pdf",
        bbox_inches="tight",
        pad_inches=0.06,
    )
    plt.close(figure)


def plot_tradeoff(output: Path, overall: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)
    palette = dict(
        zip(
            METHOD_ORDER,
            sns.color_palette("colorblind", n_colors=len(METHOD_ORDER)),
            strict=True,
        )
    )
    figure, axes = plt.subplots(2, 2, figsize=(13.8, 10.4), constrained_layout=True)
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
    for index, (dataset, dataset_label) in enumerate(DATASETS.items()):
        axis = axes.flat[index]
        selected = overall[overall["dataset"] == dataset].set_index("method")
        for method in METHOD_ORDER:
            row = selected.loc[method]
            x = float(row["mean_l1_robust_scale"])
            y = 100 * float(row["changed_validity_given_base_valid"])
            if not np.isfinite(x) or not np.isfinite(y):
                continue
            axis.scatter(x, y, s=55, color=palette[method], zorder=3)
            axis.annotate(
                METHOD_LABELS[method],
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
        "Robustness must be interpreted together with proximity and coverage",
        fontsize=17,
        fontweight="bold",
    )
    figure.savefig(output / "robustness_proximity_tradeoff.png", dpi=220)
    figure.savefig(output / "robustness_proximity_tradeoff.pdf")
    plt.close(figure)


def plot_paper_tradeoff(output: Path, overall: pd.DataFrame) -> None:
    sns.set_theme(style="white", context="paper", font="DejaVu Sans")
    figure, axes = plt.subplots(2, 2, figsize=(7.2, 5.45), facecolor="white")
    for index, (dataset, dataset_label) in enumerate(DATASETS.items()):
        axis = axes.flat[index]
        axis.set_facecolor(PAPER_FACE)
        selected = overall[overall["dataset"] == dataset].set_index("method")
        points = []
        for method in METHOD_ORDER:
            row = selected.loc[method]
            x = float(row["mean_l1_robust_scale"])
            y = 100 * float(row["changed_validity_given_base_valid"])
            if not np.isfinite(x) or not np.isfinite(y):
                continue
            points.append((x, y))
            axis.scatter(
                x,
                y,
                s=52,
                color=PAPER_METHOD_COLORS[method],
                marker=PAPER_METHOD_MARKERS[method],
                edgecolor="white",
                linewidth=0.8,
                label=METHOD_LABELS[method],
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

        x_values = np.array([point[0] for point in points])
        x_padding = max(0.05, 0.08 * float(np.ptp(x_values)))
        axis.set_xlim(
            max(0, float(x_values.min()) - x_padding), x_values.max() + x_padding
        )
        axis.set_ylim(43, 102.5)
        axis.set_yticks((50, 60, 70, 80, 90, 100))
        axis.set_title(
            dataset_label,
            fontsize=9.2,
            fontweight="bold",
            color=PAPER_INK,
            loc="left",
            pad=7,
        )
        axis.set_xlabel(
            r"Mean scaled $L_1$ distance  $\leftarrow$ closer", fontsize=7.1
        )
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
        bottom=0.19,
        top=0.90,
        hspace=0.45,
        wspace=0.29,
    )
    handles, labels = axes.flat[0].get_legend_handles_labels()
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
        "Robustness and proximity form dataset-specific frontiers",
        x=0.10,
        y=0.988,
        ha="left",
        fontsize=10.2,
        fontweight="bold",
        color=PAPER_INK,
    )
    figure.savefig(
        output / "paper_robustness_proximity_tradeoff.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.06,
    )
    figure.savefig(
        output / "paper_robustness_proximity_tradeoff.pdf",
        bbox_inches="tight",
        pad_inches=0.06,
    )
    plt.close(figure)


def plot_metric_decomposition(output: Path, overall: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)
    metrics = (
        ("generation_coverage", "Coverage", "o"),
        ("validity_given_generated", "Base validity", "s"),
        ("changed_validity_given_base_valid", "Empirical robustness", "D"),
    )
    colors = sns.color_palette("colorblind", n_colors=len(metrics))
    figure, axes = plt.subplots(2, 2, figsize=(13.6, 10.2), constrained_layout=True)
    y_positions = np.arange(len(METHOD_ORDER))
    for index, (dataset, dataset_label) in enumerate(DATASETS.items()):
        axis = axes.flat[index]
        selected = overall[overall["dataset"] == dataset].set_index("method")
        for (column, label, marker), color in zip(metrics, colors, strict=True):
            values = 100 * selected.loc[list(METHOD_ORDER), column].to_numpy()
            axis.scatter(
                values,
                y_positions,
                label=label,
                marker=marker,
                color=color,
                s=42,
                zorder=3,
            )
        axis.set_yticks(y_positions, [METHOD_LABELS[method] for method in METHOD_ORDER])
        axis.invert_yaxis()
        axis.set_xlim(-3, 103)
        axis.set_xlabel("Rate (%)")
        axis.set_title(dataset_label, fontweight="bold")
        if index == 0:
            axis.legend(loc="lower left", frameon=False, fontsize=8.5)
    figure.suptitle(
        "Coverage, validity, and robustness answer different questions",
        fontsize=17,
        fontweight="bold",
    )
    figure.savefig(output / "metric_decomposition.png", dpi=220)
    figure.savefig(output / "metric_decomposition.pdf")
    plt.close(figure)


def write_brief(output: Path, overall: pd.DataFrame) -> None:
    averages = (
        overall.groupby("method", as_index=False)
        .agg(
            empirical_robustness=("changed_validity_given_base_valid", "mean"),
            coverage=("generation_coverage", "mean"),
            base_validity=("validity_given_generated", "mean"),
            distance=("mean_l1_robust_scale", "mean"),
        )
        .sort_values("empirical_robustness", ascending=False)
    )
    averages["Method"] = averages["method"].map(METHOD_LABELS)
    for column in ("empirical_robustness", "coverage", "base_validity"):
        averages[column] = averages[column].map(lambda value: f"{100 * value:.1f}%")
    averages["distance"] = averages["distance"].map(lambda value: f"{value:.3f}")
    table = averages[
        ["Method", "coverage", "base_validity", "empirical_robustness", "distance"]
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
        ["dataset_label", "generated_n", "certified_n", "certified_coverage"],
    ].copy()
    apas_certification["certified_coverage"] = apas_certification[
        "certified_coverage"
    ].map(lambda value: f"{100 * value:.1f}%")
    apas_certification = apas_certification.rename(
        columns={
            "dataset_label": "Dataset",
            "generated_n": "Author-sample accepted",
            "certified_n": "Independent holdout certified",
            "certified_coverage": "Certified coverage",
        }
    )
    robx_robust_rows = overall[overall["method"].eq("robx_robust")]
    minimum_robx_coverage = 100 * float(robx_robust_rows["generation_coverage"].min())
    text = f"""# Robust to which model change?

## Motivation

Counterfactual explanations promise actionable advice, but the model that produced that advice will eventually be retrained or replaced. Existing robust-CFE methods protect against different notions of model change, so a single robustness score can conceal both family-specific failures and failures to generate any counterfactual at all. We therefore test the same methods against controlled families of changed models, measure model difference through prediction behavior, and ask: **robust to which model change, at what coverage and proximity cost?**

## Final protocol

- Four datasets and five independently trained 32×32 MLP base models per dataset.
- Eight change families with 25 variants per family and base model: 4,000 changed evaluation models.
- Eight methods represented by nine configurations.
- Empirical robustness is reported with coverage, base validity, and proximity.
- Seed-wise means and standard deviations use five independent base-model seeds.

## Main results

![Empirical robustness by model-change family](robustness_by_change_family.png)

![Coverage, validity, and robustness decomposition](metric_decomposition.png)

![Robustness and proximity trade-off](robustness_proximity_tradeoff.png)

## Cross-dataset headline averages

{table.to_markdown(index=False)}

RobX thresholds are calibrated from the base model's target-class training stability distribution. The balanced and robustness-first settings use its median and upper decile, respectively.

## APAS certification audit

The author-faithful APAS margin search reuses its model-parameter sample. We
therefore certify the selected CFE once more on an independent sample that
cannot affect selection. Empirical survival continues to use every returned
CFE; `Certified coverage` records only independent holdout successes.

{apas_certification.to_markdown(index=False)}

## Early findings

1. Robustness is strongly change-family dependent. Bounded parameter perturbations are generally much easier than new seeds, architecture changes, or bootstrap retraining.
2. Higher robustness often requires substantially larger counterfactual changes (worse proximity).
3. Coverage and base validity must accompany empirical robustness. The lowest dataset-level coverage of robustness-first RobX is {minimum_robx_coverage:.1f}%.
4. Rankings change across datasets. For example, RBR exceeds APAS on Wine, while APAS exceeds RBR on HELOC.

All percentages use the complete five-seed benchmark.
"""
    (output / "results_brief.md").write_text(text)


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    overall, family = load_results(args.artifacts)
    seeds = seed_metrics(args.artifacts)
    intervals = seed_summary(seeds)
    runtime = runtime_metrics(args.artifacts)
    save_tables(args.output, overall, family, seeds, intervals)
    save_runtime_table(args.output, runtime)
    plot_family_heatmaps(args.output, family)
    plot_paper_family_heatmaps(args.output, family)
    plot_tradeoff(args.output, overall)
    plot_paper_tradeoff(args.output, overall)
    plot_metric_decomposition(args.output, overall)
    write_brief(args.output, overall)


if __name__ == "__main__":
    main()
