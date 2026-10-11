import math

import numpy as np
import pandas as pd
import pytest

from robustness_benchmark.evaluation.rank_tests import (
    InsufficientData,
    nemenyi_critical_difference,
    plot_critical_difference,
    rank_test,
)

# A is best in every block; B and C swap only in b3.
SCORES = {
    "b1": {"A": 3, "B": 2, "C": 1},
    "b2": {"A": 3, "B": 2, "C": 1},
    "b3": {"A": 3, "B": 1, "C": 2},
    "b4": {"A": 3, "B": 2, "C": 1},
}


def long_frame(scores: dict[str, dict[str, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"block": block, "method": method, "score": value}
            for block, row in scores.items()
            for method, value in row.items()
        ]
    )


def run(scores: dict[str, dict[str, float]], higher_is_better: bool = True):
    return rank_test(
        long_frame(scores),
        block="block",
        method="method",
        score="score",
        higher_is_better=higher_is_better,
    )


def test_hand_made_example_matches_known_ranks_statistic_and_cd():
    result = run(SCORES)

    assert result.average_ranks.to_dict() == {"A": 1.0, "B": 2.25, "C": 2.75}
    # chi2 = 12N / (k(k+1)) * (sum R^2 - k(k+1)^2 / 4) = 4 * (13.625 - 12)
    assert result.statistic == pytest.approx(6.5)
    assert result.p_value == pytest.approx(math.exp(-6.5 / 2))  # chi2 with 2 df
    # Demsar's table: q_0.05 = 2.343 for k = 3.
    assert result.critical_difference == pytest.approx(
        2.343 * math.sqrt(12 / 24), abs=2e-3
    )
    assert (result.n_blocks, result.n_blocks_dropped, result.k_methods) == (4, 0, 3)
    # A-B (1.25) and B-C (0.5) are within CD; A-C (1.75) is not.
    assert result.cliques() == [(0, 1), (1, 2)]


def test_lower_is_better_reverses_ranks():
    result = run(SCORES, higher_is_better=False)

    assert result.average_ranks.to_dict() == {"C": 1.25, "B": 1.75, "A": 3.0}


def test_critical_difference_matches_demsar_table():
    for k, q_alpha in {2: 1.960, 5: 2.728, 10: 3.164}.items():
        expected = q_alpha * math.sqrt(k * (k + 1) / 6)
        assert nemenyi_critical_difference(k, 1) == pytest.approx(expected, rel=2e-3)


def test_ties_share_the_average_rank():
    result = run({"b1": {"A": 1, "B": 1, "C": 0}, "b2": {"A": 2, "B": 1, "C": 0}})

    assert result.average_ranks.to_dict() == {"A": 1.25, "B": 1.75, "C": 3.0}


def test_incomplete_blocks_are_dropped_and_counted():
    result = run(
        {
            **SCORES,
            "missing_method": {"A": 1, "B": 2},
            "nan_score": {"A": 1, "B": np.nan, "C": 3},
        }
    )

    assert (result.n_blocks, result.n_blocks_dropped) == (4, 2)
    assert result.statistic == pytest.approx(6.5)


def test_fully_tied_blocks_give_no_evidence():
    result = run({"b1": dict.fromkeys("ABC", 1), "b2": dict.fromkeys("ABC", 1)})

    assert (result.statistic, result.p_value) == (0.0, 1.0)
    assert result.cliques() == [(0, 2)]


@pytest.mark.parametrize(
    "scores",
    [
        {"b1": {"A": 1, "B": 2, "C": 3}},
        {"b1": {"A": 1, "B": 2}, "b2": {"A": 2, "B": 1}},
        {"b1": {"A": 1, "B": 2, "C": 3}, "b2": {"A": 1, "B": 2}},
    ],
)
def test_too_little_data_raises(scores):
    with pytest.raises(InsufficientData):
        run(scores)


def test_diagram_labels_every_method_with_its_rank():
    figure = plot_critical_difference(
        run(SCORES), title="Example", labels={"A": "Method A"}
    )

    texts = {text.get_text() for text in figure.axes[0].texts}
    assert {"Method A (1.00)", "B (2.25)", "C (2.75)"} <= texts
