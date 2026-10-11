import json

import numpy as np
import pandas as pd
import pytest

from robustness_benchmark.benchmark.store import BENCHMARK_PROTOCOL_VERSION
from robustness_benchmark.benchmark.summary import write_summary
from robustness_benchmark.evaluation.plausibility import PlausibilityReference
from robustness_benchmark.evaluation.proximity import (
    GowerDistance,
    feature_range,
    proximity_metrics,
)


class ThresholdModel:
    """Stand-in base model: P(y=1) is a steep sigmoid of the first feature."""

    def predict(self, X) -> pd.DataFrame:
        a = np.asarray(X, dtype=float)[:, 0]
        return pd.DataFrame(1.0 / (1.0 + np.exp(-10.0 * a)))


def gaussian_reference(n: int = 400) -> tuple[PlausibilityReference, pd.DataFrame]:
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=["a", "b", "c"])
    return PlausibilityReference(X, ThresholdModel()), X


def frames(factual, counterfactual) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = ["a", "b", "c"]
    return (
        pd.DataFrame(np.atleast_2d(factual), columns=columns),
        pd.DataFrame(np.atleast_2d(counterfactual), columns=columns),
    )


def test_sparsity_counts_changed_features_above_tolerance():
    _, X = gaussian_reference()
    factual, counterfactual = frames([-1.0, 0.0, 0.0], [1.0, 0.5, 1e-9])

    (row,) = proximity_metrics(GowerDistance(X), factual, counterfactual, [True])

    assert row["n_changed_features"] == 2
    assert row["fraction_changed_features"] == pytest.approx(2 / 3)


def test_knn_distance_matches_hand_computed_gower():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, -5.0, -6.0]})
    reference = PlausibilityReference(X, ThresholdModel(), k=2)
    scale = feature_range(X).iloc[0]
    # Target points are 1..7; the two nearest to 10 are 7 and 6.
    expected = ((10 - 7) + (10 - 6)) / 2 / scale
    counterfactual = pd.DataFrame({"a": [10.0]})

    (row,) = reference.evaluate(counterfactual, [True])

    assert row["knn_distance"] == pytest.approx(expected)


def test_relative_knn_distance_is_about_one_for_typical_target_points():
    reference, X = gaussian_reference()
    typical = X[X["a"] > 0].iloc[:50]

    rows = reference.evaluate(typical, [True] * len(typical))
    far = reference.evaluate(frames([-1, 0, 0], [8, 8, 8])[1], [True])[0]

    # Training points match themselves here, so the median sits slightly below 1.
    assert 0.5 < np.median([row["knn_distance_relative"] for row in rows]) < 1.2
    assert far["knn_distance_relative"] > 5


def test_lof_separates_outliers_from_central_points():
    reference, _ = gaussian_reference()
    factual = np.array([[-1.0, 0.0, 0.0]] * 2)
    counterfactual = np.array([[0.7, 0.0, 0.0], [12.0, 12.0, 12.0]])

    central, outlier = reference.evaluate(
        frames(factual, counterfactual)[1], [True] * 2
    )

    assert central["lof_inlier"] is True
    assert outlier["lof_inlier"] is False
    assert central["lof_score"] < 1.5 < outlier["lof_score"]


def test_ynn_is_one_inside_target_region_and_lower_at_the_boundary():
    reference, _ = gaussian_reference(n=2000)
    factual = np.array([[-1.0, 0.0, 0.0]] * 2)
    counterfactual = np.array([[2.5, 0.0, 0.0], [0.0, 0.0, 0.0]])

    deep, boundary = reference.evaluate(frames(factual, counterfactual)[1], [True] * 2)

    assert deep["ynn"] == 1.0
    assert boundary["ynn"] < 1.0


def test_non_base_valid_rows_are_not_scored():
    reference, _ = gaussian_reference()
    factual = np.array([[-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    counterfactual = np.array([[np.nan, np.nan, np.nan], [1.0, 0.0, 0.0]])

    invalid, valid = reference.evaluate(
        frames(factual, counterfactual)[1], [False, True]
    )

    assert invalid["lof_inlier"] is None
    assert all(np.isnan(value) for key, value in invalid.items() if key != "lof_inlier")
    assert not np.isnan(valid["ynn"])


def test_all_invalid_returns_nan_rows():
    reference, _ = gaussian_reference()
    rows = reference.evaluate(frames([-1, 0, 0], [np.nan] * 3)[1], [False])

    assert np.isnan(rows[0]["knn_distance"])


def write_runs(output, generation_columns: dict) -> None:
    directory = output / "runs" / "seed_1" / "method_a"
    directory.mkdir(parents=True)
    common = {
        "method": ["method_a"] * 2,
        "base_model_id": ["base-1"] * 2,
        "benchmark_protocol_version": [BENCHMARK_PROTOCOL_VERSION] * 2,
    }
    pd.DataFrame(
        {
            **common,
            "generation_status": ["success", "base_invalid"],
            "base_valid": [True, False],
            "base_valid_before_projection": [True, False],
            "generation_error": [None, None],
            "runtime_seconds": [1.0, 2.0],
            "gower_distance": [0.5, None],
            "n_changed_features": [2.0, None],
            "method_config": ["{}"] * 2,
            **generation_columns,
        }
    ).to_parquet(directory / "generation.parquet")
    pd.DataFrame(
        {
            **common,
            "change_id": ["c1", "c1"],
            "change_family": ["f"] * 2,
            "change_severity": ["mild"] * 2,
            "base_valid": [True, False],
            "conditional_eligible": [True, False],
            "conditional_survival": [True, None],
            "ce_valid": [True, None],
            "hard_disagreement": [0.1, 0.1],
            "probability_mae": [0.1, 0.1],
        }
    ).to_parquet(directory / "survival.parquet")


def generation_summary(output) -> dict:
    write_summary(
        output,
        ["method_a"],
        {"method_a": {}},
        ["base-1"],
        calibrations=[],
        skipped_methods={},
        model_type="mlp",
    )
    return json.loads((output / "summary.json").read_text())["generation"][0]


def test_summary_averages_plausibility_over_base_valid_rows(tmp_path):
    write_runs(
        tmp_path,
        {
            "knn_distance": [0.4, np.nan],
            "knn_distance_relative": [1.5, np.nan],
            "lof_inlier": [True, None],
            "lof_score": [1.2, np.nan],
            "ynn": [0.8, np.nan],
        },
    )

    summary = generation_summary(tmp_path)

    assert summary["mean_n_changed_features"] == 2.0
    assert summary["mean_knn_distance"] == pytest.approx(0.4)
    assert summary["mean_knn_distance_relative"] == pytest.approx(1.5)
    assert summary["lof_inlier_rate"] == 1.0
    assert summary["mean_lof_score"] == pytest.approx(1.2)
    assert summary["mean_ynn"] == pytest.approx(0.8)


def test_summary_tolerates_generation_runs_without_plausibility_columns(tmp_path):
    write_runs(tmp_path, {})

    summary = generation_summary(tmp_path)

    for key in (
        "mean_knn_distance",
        "mean_knn_distance_relative",
        "lof_inlier_rate",
        "mean_lof_score",
        "mean_ynn",
    ):
        assert summary[key] is None
