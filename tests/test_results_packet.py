import json

import numpy as np
import pandas as pd

from robustness_benchmark.cli.results_packet import (
    DATASETS,
    FAMILY_ORDER,
    MARGIN_METHODS,
    METHOD_ORDER,
    discover_runs,
    family_matrix,
    method_availability,
    plot_paper_family_heatmaps,
    plot_paper_tradeoff,
    seed_summary,
)

XGBOOST_NOT_RUN = {"wachter", "apas", "rnce", "trex", "roar"}


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
