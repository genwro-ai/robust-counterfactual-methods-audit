"""Figures of the results packet, all in the shared figure style."""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from robustness_benchmark.evaluation.rank_tests import RankTestResult
from robustness_benchmark.results.figure_style import (
    AXIS,
    EMPTY,
    INK,
    LABEL_SIZE,
    MARGIN_LABEL,
    MUTED,
    ROBUSTNESS_CMAP,
    SECONDARY,
    SMALL_SIZE,
    SURFACE,
    TICK_SIZE,
    TITLE_SIZE,
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
from robustness_benchmark.results.labels import (
    DATASET_LABELS,
    FAMILY_LABELS,
    FAMILY_ORDER,
    MARGIN_BASELINES,
    MARGIN_METHODS,
    METHOD_ORDER,
    MODEL_TYPE_LABELS,
    SCOPE_LABELS,
    method_label,
    ordered,
)
from robustness_benchmark.results.rank_analysis import (
    ALL_FAMILIES,
    RANK_ANALYSES,
    RankKey,
    rank_diagram_stem,
)


@styled
def plot_critical_difference(result: RankTestResult, *, title: str) -> Figure:
    """Demšar's critical-difference diagram: rank 1 at the right, bars join cliques.

    The better half of the methods is labelled on the right, the rest on the
    left; each label hangs from its average rank on the axis and carries the
    method's marker.
    """

    ranks = result.average_ranks
    k = len(ranks)
    cliques = result.cliques()
    row_height, unit_inches = 0.55, 0.3
    first_row = -0.5 - 0.22 * len(cliques)
    rows = math.ceil(k / 2)
    bottom = first_row - row_height * rows

    figure, axis = plt.subplots(figsize=(7.2, unit_inches * (1.8 - bottom)))
    figure.subplots_adjust(left=0.28, right=0.72, bottom=0.01, top=0.88)
    axis.set(xlim=(k + 0.3, 0.7), ylim=(bottom, 1.4))
    axis.axis("off")
    figure.suptitle(title, x=0.01, ha="left", fontsize=TITLE_SIZE, fontweight="bold")

    def line(x, y, **kwargs):
        axis.plot(x, y, color=kwargs.pop("color", INK), clip_on=False, **kwargs)

    line([1, k], [0, 0], linewidth=0.8, color=SECONDARY)
    for tick in range(1, k + 1):
        line([tick, tick], [0, 0.12], linewidth=0.8, color=SECONDARY)
        axis.text(
            tick,
            0.2,
            str(tick),
            ha="center",
            va="bottom",
            fontsize=TICK_SIZE,
            color=SECONDARY,
        )
    line([k, k - result.critical_difference], [1.0, 1.0], linewidth=1.6)
    axis.text(
        k - result.critical_difference / 2,
        1.1,
        f"CD = {result.critical_difference:.2f}",
        ha="center",
        va="bottom",
        fontsize=TICK_SIZE,
    )

    for level, (start, end) in enumerate(cliques):
        y = -0.25 - 0.22 * level
        line([ranks.iloc[start] - 0.04, ranks.iloc[end] + 0.04], [y, y], linewidth=2.4)

    for position, (name, rank) in enumerate(ranks.items()):
        best_half = position < rows
        row = position if best_half else k - 1 - position
        y = first_row - row_height * row
        edge = 0.55 if best_half else k + 0.45
        line([rank, rank, edge], [0, y, y], linewidth=0.6, color=MUTED)
        scatter_method(axis, name, edge, y, zorder=4, clip_on=False)
        axis.annotate(
            f"{method_style(name).label} ({rank:.2f})",
            (edge, y),
            xytext=(6 if best_half else -6, 0),
            textcoords="offset points",
            annotation_clip=False,
            ha="left" if best_half else "right",
            va="center",
            fontsize=LABEL_SIZE,
        )
    return figure


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
        (dataset, label)
        for dataset, label in DATASET_LABELS.items()
        if dataset in present
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
        f"{MODEL_TYPE_LABELS[model_type]}: {title}",
        x=0.01,
        ha="left",
        fontweight="bold",
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
