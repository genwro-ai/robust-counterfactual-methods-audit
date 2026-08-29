import numpy as np
import pandas as pd

from robustness_benchmark.cli.results_packet import (
    DATASETS,
    FAMILY_ORDER,
    METHOD_ORDER,
    plot_paper_family_heatmaps,
    plot_paper_tradeoff,
    seed_summary,
)


def test_seed_summary_preserves_an_undefined_metric():
    seeds = pd.DataFrame(
        {
            "dataset": ["heloc"] * 5,
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
    assert survival["seeds_n"] == 0
    assert np.isnan(survival["estimate"])
    assert np.isnan(survival["seed_sd"])


def test_paper_heatmap_handles_undefined_empirical_robustness(tmp_path):
    rows = [
        {
            "dataset": dataset,
            "method": method,
            "change_family": family,
            "changed_validity_given_base_valid": (
                np.nan
                if (dataset, method, family) == ("heloc", "robx_robust", "architecture")
                else 0.75
            ),
        }
        for dataset in DATASETS
        for method in METHOD_ORDER
        for family in FAMILY_ORDER
    ]

    plot_paper_family_heatmaps(tmp_path, pd.DataFrame(rows))

    assert (tmp_path / "paper_robustness_by_change_family.pdf").is_file()


def test_paper_tradeoff_omits_a_method_without_proximity(tmp_path):
    rows = [
        {
            "dataset": dataset,
            "method": method,
            "mean_l1_robust_scale": (
                np.nan if (dataset, method) == ("heloc", "robx_robust") else 0.5
            ),
            "changed_validity_given_base_valid": 0.8,
        }
        for dataset in DATASETS
        for method in METHOD_ORDER
    ]

    plot_paper_tradeoff(tmp_path, pd.DataFrame(rows))

    assert (tmp_path / "paper_robustness_proximity_tradeoff.pdf").is_file()
