"""Rank tests over datasets: the main, per-family and cross-class analyses."""

from dataclasses import dataclass

import pandas as pd

from robustness_benchmark.evaluation.rank_tests import (
    InsufficientData,
    RankTestResult,
    rank_test,
)
from robustness_benchmark.results.labels import (
    CROSS_CLASS,
    FAMILY_LABELS,
    FAMILY_ORDER,
    MARGIN_BASELINES,
    METHOD_ORDER,
    MODEL_TYPE_LABELS,
    SCOPE_LABELS,
    method_label,
    ordered,
)


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

    model_types = ordered(overall["model_type"], tuple(MODEL_TYPE_LABELS))
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
