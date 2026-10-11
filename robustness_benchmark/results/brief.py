"""The results brief: protocol, headline figures, tables and rank tests."""

from pathlib import Path

import pandas as pd

from robustness_benchmark.results.labels import (
    MODEL_TYPE_LABELS,
    SCOPE_LABELS,
    method_label,
    ordered,
)
from robustness_benchmark.results.rank_analysis import (
    ALL_FAMILIES,
    RANK_ANALYSES,
    rank_diagram_stem,
    rank_test_markdown,
)
from robustness_benchmark.results.tables import cross_class_markdown


def protocol_lines(overall: pd.DataFrame, family: pd.DataFrame) -> list[str]:
    lines = []
    for model_type in ordered(overall["model_type"], tuple(MODEL_TYPE_LABELS)):
        rows = overall[overall["model_type"].eq(model_type)]
        families = family.loc[family["model_type"].eq(model_type), "change_family"]
        not_run = sorted(set(overall["method"]) - set(rows["method"]))
        lines.append(
            f"- {MODEL_TYPE_LABELS[model_type]}: {rows['dataset'].nunique()} datasets, "
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
    averages["Model"] = averages["model_type"].map(MODEL_TYPE_LABELS)
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
    apas_certification["model_type"] = apas_certification["model_type"].map(
        MODEL_TYPE_LABELS
    )
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
        f"![{MODEL_TYPE_LABELS[model_type]}: robustness by change family]"
        f"(paper_robustness_by_change_family_{model_type}.png)\n\n"
        f"![{MODEL_TYPE_LABELS[model_type]}: coverage, validity, and robustness]"
        f"(paper_metric_decomposition_{model_type}.png)\n\n"
        f"![{MODEL_TYPE_LABELS[model_type]}: robustness and proximity]"
        f"(paper_robustness_proximity_tradeoff_{model_type}.png)"
        + (
            f"\n\n![{MODEL_TYPE_LABELS[model_type]}: robustness and plausibility]"
            f"(paper_robustness_plausibility_{model_type}.png)"
            if (output / f"paper_robustness_plausibility_{model_type}.png").is_file()
            else ""
        )
        for model_type in ordered(overall["model_type"], tuple(MODEL_TYPE_LABELS))
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
