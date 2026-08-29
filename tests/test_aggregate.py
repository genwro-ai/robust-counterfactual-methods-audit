import pandas as pd

from robustness_benchmark.evaluation.aggregate import aggregate_survival


def test_aggregate_weights_each_changed_model_once_for_model_level_metrics():
    rows = []
    for change_id, distance, outcomes in [
        ("a", 0.1, [True, True]),
        ("b", 0.3, [False, False]),
    ]:
        for outcome in outcomes:
            rows.append(
                {
                    "method": "example",
                    "change_family": "seed",
                    "base_model_id": "base",
                    "change_id": change_id,
                    "base_valid": True,
                    "conditional_eligible": True,
                    "conditional_survival": outcome,
                    "ce_valid": outcome,
                    "hard_disagreement": distance,
                    "probability_mae": distance,
                }
            )

    summary = aggregate_survival(pd.DataFrame(rows))[0]

    assert summary["changed_models_n"] == 2
    assert summary["mean_changed_model_conditional_survival"] == 0.5
    assert summary["median_changed_model_conditional_survival"] == 0.5
    assert summary["p10_changed_model_conditional_survival"] == 0.1
    assert summary["mean_probability_mae"] == 0.2
    assert summary["base_validity_rate"] == 1.0
    assert summary["end_to_end_changed_validity"] == 0.5
