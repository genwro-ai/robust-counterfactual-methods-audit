import json

import numpy as np
import pandas as pd
import pytest

from robustness_benchmark.cli.results_packet import (
    ALL_FAMILIES,
    DATASETS,
    FAMILY_ORDER,
    MARGIN_METHODS,
    METHOD_ORDER,
    cross_class_markdown,
    cross_class_summary,
    discover_runs,
    family_matrix,
    format_column,
    method_availability,
    plot_paper_family_heatmaps,
    plot_paper_plausibility,
    plot_paper_tradeoff,
    plot_rank_diagrams,
    rank_diagram_stem,
    rank_test_table,
    run_rank_tests,
    seed_summary,
    write_brief,
)

XGBOOST_NOT_RUN = {"wachter", "apas", "rnce", "trex", "roar"}
MODEL_METHODS = {
    "mlp": [method for method in METHOD_ORDER if method != "roar"],
    "logistic": list(METHOD_ORDER),
    "xgboost": [method for method in METHOD_ORDER if method not in XGBOOST_NOT_RUN],
}


def family_rows(model_type: str, value: float = 0.75) -> list[dict[str, object]]:
    methods = [
        method
        for method in METHOD_ORDER
        if model_type != "xgboost" or method not in XGBOOST_NOT_RUN
    ]
    return [
        {
            "dataset": dataset,
            "model_type": model_type,
            "method": method,
            "change_family": family,
            "changed_validity_given_base_valid": value,
        }
        for dataset in DATASETS
        for method in methods
        for family in FAMILY_ORDER
    ]


def test_seed_summary_preserves_an_undefined_metric():
    seeds = pd.DataFrame(
        {
            "dataset": ["heloc"] * 5,
            "model_type": ["mlp"] * 5,
            "seed": [2026, 2027, 2028, 2029, 2030],
            "method": ["robx_robust"] * 5,
            "generation_coverage": [0.0] * 5,
            "validity_given_generated": [np.nan] * 5,
            "end_to_end_validity": [0.0] * 5,
            "pooled_conditional_survival": [np.nan] * 5,
            "end_to_end_changed_validity": [0.0] * 5,
            "changed_validity_given_base_valid": [np.nan] * 5,
            "mean_l1_robust_scale": [np.nan] * 5,
        }
    )

    intervals = seed_summary(seeds)

    survival = intervals[intervals["metric"].eq("pooled_conditional_survival")].iloc[0]
    assert survival["model_type"] == "mlp"
    assert survival["seeds_n"] == 0
    assert np.isnan(survival["estimate"])
    assert np.isnan(survival["seed_sd"])


def test_family_matrix_marks_methods_a_model_class_cannot_run():
    family = pd.DataFrame(family_rows("mlp") + family_rows("xgboost"))

    matrix, not_run = family_matrix(family, "heloc", "xgboost")

    assert set(not_run) == XGBOOST_NOT_RUN
    assert matrix.loc[sorted(XGBOOST_NOT_RUN)].isna().all().all()
    assert not set(MARGIN_METHODS[1:]).intersection(matrix.index)
    assert list(matrix.columns) == list(FAMILY_ORDER)


def test_paper_heatmap_handles_undefined_and_unavailable_cells(tmp_path):
    family = pd.DataFrame(family_rows("mlp") + family_rows("xgboost"))
    undefined = (
        family["dataset"].eq("heloc")
        & family["method"].eq("robx_robust")
        & family["change_family"].eq("architecture")
    )
    family.loc[undefined, "changed_validity_given_base_valid"] = np.nan

    for model_type in ("mlp", "xgboost"):
        plot_paper_family_heatmaps(tmp_path, family, model_type)

    assert (tmp_path / "paper_robustness_by_change_family_mlp.pdf").is_file()
    assert (tmp_path / "paper_robustness_by_change_family_xgboost.pdf").is_file()


def test_paper_tradeoff_omits_a_method_without_proximity(tmp_path):
    rows = [
        {
            "dataset": dataset,
            "model_type": "logistic",
            "method": method,
            "mean_l1_robust_scale": (
                np.nan if (dataset, method) == ("heloc", "robx_robust") else 0.5
            ),
            "changed_validity_given_base_valid": 0.8,
        }
        for dataset in DATASETS
        for method in METHOD_ORDER
    ]

    plot_paper_tradeoff(tmp_path, pd.DataFrame(rows), "logistic")

    assert (tmp_path / "paper_robustness_proximity_tradeoff_logistic.pdf").is_file()


def test_method_availability_reports_not_run_methods():
    overall = pd.DataFrame(
        [
            {"dataset": "heloc", "model_type": "mlp", "method": "wachter"},
            {"dataset": "heloc", "model_type": "mlp", "method": "kdtree"},
            {"dataset": "heloc", "model_type": "xgboost", "method": "kdtree"},
        ]
    )

    table = method_availability(overall)

    assert table.loc["wachter", "xgboost"] == "n/a"
    assert table.loc["kdtree", "xgboost"] == "1 datasets"


def test_discover_runs_keeps_the_legacy_mlp_directory(tmp_path):
    (tmp_path / "full_heloc").mkdir()
    xgboost_root = tmp_path / "full_heloc_xgboost"
    xgboost_root.mkdir()
    (xgboost_root / "summary.json").write_text(json.dumps({"model_type": "xgboost"}))

    runs = discover_runs(tmp_path)

    assert runs == [
        ("heloc", "mlp", tmp_path / "full_heloc"),
        ("heloc", "xgboost", xgboost_root),
    ]


def synthetic_results(plausibility: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Overall and per-family frames where later methods in METHOD_ORDER do better."""

    rng = np.random.default_rng(0)
    overall, family = [], []
    for model_type, methods in MODEL_METHODS.items():
        for dataset, label in DATASETS.items():
            for method in methods:
                skill = 0.02 * METHOD_ORDER.index(method)
                overall.append(
                    {
                        "dataset": dataset,
                        "dataset_label": label,
                        "model_type": model_type,
                        "method": method,
                        "generation_coverage": 1.0,
                        "validity_given_generated": 0.9,
                        "generated_n": 100,
                        "certified_n": 90,
                        "certified_coverage": 0.9,
                        "changed_validity_given_base_valid": 0.5 + skill,
                        "end_to_end_changed_validity": 0.4 + skill,
                        "mean_l1_robust_scale": 1.0 - skill + rng.normal(0, 0.01),
                        "mean_knn_distance_relative": (
                            1.0 + rng.uniform() if plausibility else None
                        ),
                    }
                )
                family.extend(
                    {
                        "dataset": dataset,
                        "model_type": model_type,
                        "method": method,
                        "change_family": change_family,
                        "changed_validity_given_base_valid": 0.5
                        + skill
                        + rng.normal(0, 0.05),
                        "end_to_end_changed_validity": 0.4
                        + skill
                        + rng.normal(0, 0.05),
                    }
                    for change_family in FAMILY_ORDER
                )
    return pd.DataFrame(overall), pd.DataFrame(family)


def test_cross_class_summary_marks_methods_a_model_class_did_not_run():
    overall, _ = synthetic_results()

    summary = cross_class_summary(overall)
    markdown = cross_class_markdown(summary)

    assert summary.loc["kdtree", "xgboost_empirical_robustness"] == pytest.approx(
        0.5 + 0.02 * METHOD_ORDER.index("kdtree")
    )
    assert summary.loc[sorted(XGBOOST_NOT_RUN), "xgboost_distance"].isna().all()
    assert list(summary.columns[:3]) == [
        "mlp_empirical_robustness",
        "mlp_end_to_end_robustness",
        "mlp_distance",
    ]
    wachter = next(
        line for line in markdown.splitlines() if line.startswith("| Wachter")
    )
    assert wachter.split("|")[-2].strip() == "n/a"
    assert "% / " in wachter


def test_format_column_shows_n_a_for_unrecorded_values():
    formatted = format_column(pd.Series([None, None], dtype=object), "{:.2f}")

    assert formatted.tolist() == ["n/a", "n/a"]
    assert format_column(pd.Series([0.5, np.nan]), "{:.1%}").tolist() == [
        "50.0%",
        "n/a",
    ]


def test_rank_tests_use_datasets_as_blocks():
    overall, family = synthetic_results()

    results = run_rank_tests(overall, family)
    table = rank_test_table(results)

    assert set(table["scope"]) == {"mlp", "logistic", "xgboost", "cross_class"}
    assert set(table["analysis"]) == {
        "empirical_robustness",
        "end_to_end_robustness",
        "distance",
    }
    main = results["empirical_robustness", "mlp", ALL_FAMILIES]
    assert main.n_blocks == len(DATASETS)
    # one appendix analysis per change family, robustness metrics only
    per_family = {
        key[2] for key in results if key[:2] == ("empirical_robustness", "mlp")
    }
    assert per_family == {ALL_FAMILIES, *FAMILY_ORDER}
    assert not any(key[0] == "distance" and key[2] != ALL_FAMILIES for key in results)
    cross = results["empirical_robustness", "cross_class", ALL_FAMILIES]
    shared = set(MODEL_METHODS["xgboost"]) - set(MARGIN_METHODS[1:])
    assert set(cross.average_ranks.index) == shared
    assert cross.n_blocks == len(DATASETS) * 3  # dataset x model type
    assert not any(
        key[1] == "cross_class" and key[2] != ALL_FAMILIES for key in results
    )
    distance = results["distance", "mlp", ALL_FAMILIES]
    assert distance.n_blocks == len(DATASETS)
    assert not any(column.startswith("rank_gs_margin") for column in table.columns)
    # skill grows along METHOD_ORDER, so distance (lower is better) favours the last
    assert distance.average_ranks.index[0] == "growing_spheres"
    assert main.average_ranks.index[0] == "growing_spheres"
    assert table.loc[table["scope"].eq("xgboost"), "rank_wachter"].isna().all()


def test_rank_tests_skip_analyses_without_complete_blocks():
    overall, family = synthetic_results()
    one_dataset = family[family["dataset"].eq("heloc")]

    results = run_rank_tests(overall, one_dataset)

    assert ("distance", "mlp", ALL_FAMILIES) in results
    assert ("empirical_robustness", "mlp", ALL_FAMILIES) not in results  # one block
    assert run_rank_tests(overall, family.iloc[0:0]).keys() == {
        key for key in results if key[0] == "distance"
    }


def test_rank_diagrams_are_written_per_analysis(tmp_path):
    overall, family = synthetic_results()
    results = run_rank_tests(overall, family)

    plot_rank_diagrams(tmp_path, results)

    for key in results:
        for suffix in ("pdf", "png"):
            assert (tmp_path / f"{rank_diagram_stem(*key)}.{suffix}").is_file()
    assert (tmp_path / "critical_difference_distance_cross_class.pdf").is_file()
    assert (
        tmp_path / "critical_difference_empirical_robustness_mlp_bootstrap.pdf"
    ).is_file()


def test_plausibility_figure_is_written_when_the_metric_was_recorded(tmp_path):
    overall, _ = synthetic_results()

    for model_type in tuple(MODEL_METHODS):
        plot_paper_plausibility(tmp_path, overall, model_type)

    for model_type in tuple(MODEL_METHODS):
        assert (tmp_path / f"paper_robustness_plausibility_{model_type}.pdf").is_file()


def test_plausibility_figure_is_skipped_for_runs_without_the_metric(tmp_path):
    overall, _ = synthetic_results(plausibility=False)

    plot_paper_plausibility(tmp_path, overall, "mlp")
    plot_paper_plausibility(
        tmp_path, overall.drop(columns="mean_knn_distance_relative"), "mlp"
    )

    assert not list(tmp_path.glob("paper_robustness_plausibility_*"))


def test_brief_mentions_cross_class_summary_and_rank_tests(tmp_path):
    overall, family = synthetic_results()
    results = run_rank_tests(overall, family)

    write_brief(
        tmp_path,
        overall,
        family,
        cross_class_summary(overall),
        rank_test_table(results),
    )

    brief = (tmp_path / "results_brief.md").read_text()
    assert "## Rank tests" in brief
    assert "rank_tests.csv" in brief
    assert "critical_difference_empirical_robustness_cross_class.png" in brief
    assert "## Cross-class summary" in brief


def test_brief_without_rank_tests_says_so(tmp_path):
    overall, family = synthetic_results()

    write_brief(
        tmp_path,
        overall,
        family,
        cross_class_summary(overall),
        rank_test_table({}),
    )

    assert "No analysis had enough" in (tmp_path / "results_brief.md").read_text()
