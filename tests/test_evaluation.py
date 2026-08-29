import pandas as pd

from robustness_benchmark.core.model import TorchBinaryModel
from robustness_benchmark.core.training import ChangedModel
from robustness_benchmark.evaluation.metrics import (
    assess_model_changes,
    behavioral_distance,
    evaluate_survival,
    stratify_severity,
)
from robustness_benchmark.evaluation.summaries import survival_summary


def test_behavioral_distance_is_zero_for_identical_model():
    model = TorchBinaryModel(input_dim=2, hidden_dim=(), seed=0)
    X = pd.DataFrame({"a": [0.0, 1.0], "b": [1.0, 0.0]})

    result = behavioral_distance(model, model, X)

    assert result == {"hard_disagreement": 0.0, "probability_mae": 0.0}


class FirstColumnModel:
    def predict(self, X):
        return pd.DataFrame(X.iloc[:, 0].to_numpy())


class ConstantModel:
    def __init__(self, probability):
        self.probability = probability

    def predict(self, X):
        return pd.DataFrame([self.probability] * len(X))


def test_quality_flag_does_not_exclude_change_from_severity_stratification():
    X = pd.DataFrame({"x": [0.1, 0.9, 0.1, 0.9]})
    labels = pd.Series([0, 1, 0, 1])
    base = FirstColumnModel()
    changes = [
        ChangedModel("same", "test", "same", base, {}),
        ChangedModel("bad", "test", "bad", ConstantModel(0.9), {}),
    ]

    rows = assess_model_changes(changes, base, X, labels, maximum_accuracy_drop=0.03)

    assert rows[0]["quality_pass"] is True
    assert rows[0]["severity"] == "low"
    assert rows[1]["quality_pass"] is False
    assert rows[1]["severity"] == "high"


def test_stratify_severity_overrides_stale_exclusion_labels():
    rows = [
        {"probability_mae": 0.3, "hard_disagreement": 0.1, "severity": "excluded"},
        {"probability_mae": 0.1, "hard_disagreement": 0.0, "severity": "low"},
        {"probability_mae": 0.2, "hard_disagreement": 0.0, "severity": "high"},
    ]

    stratify_severity(rows)

    assert [row["severity"] for row in rows] == ["high", "low", "medium"]
    assert [row["severity_percentile"] for row in rows] == [
        2.5 / 3,
        0.5 / 3,
        1.5 / 3,
    ]


def test_invalid_generation_is_excluded_from_survival_denominator():
    model = FirstColumnModel()
    factuals = pd.DataFrame({"x": [0.1, 0.2]})
    counterfactuals = pd.DataFrame({"x": [0.9, float("nan")]})
    reference = pd.DataFrame({"x": [0.1, 0.9]})
    labels = pd.Series([0, 1])
    change = ChangedModel("seed:test", "seed", "test", model, {"seed": 1})

    rows = evaluate_survival(
        "example",
        factuals,
        counterfactuals,
        [change],
        model,
        reference,
        labels,
        [True, False],
        [0, 1],
        [100, 101],
        "base:test",
    )

    assert rows[0]["conditional_eligible"] is True
    assert rows[0]["conditional_survival"] is True
    assert rows[1]["base_valid"] is False
    assert rows[1]["ce_valid"] is None
    assert rows[1]["conditional_survival"] is None


def test_survival_summary_keeps_validity_and_conditioning_denominators():
    rows = []
    for generated, conditionally_eligible, survived, ce_valid in [
        (True, True, True, True),
        (True, False, None, True),
        (False, False, None, None),
    ]:
        rows.append(
            {
                "base_model_id": "base",
                "method": "method",
                "change_id": "change",
                "change_family": "seed",
                "change_level": "one",
                "base_valid": generated,
                "conditional_eligible": conditionally_eligible,
                "conditional_survival": survived,
                "ce_valid": ce_valid,
                "hard_disagreement": 0.1,
                "probability_mae": 0.2,
                "plain_accuracy_drop": 0.0,
            }
        )

    summary = survival_summary(pd.DataFrame(rows))[0]

    assert summary["total_n"] == 3
    assert summary["base_valid_pairs_n"] == 2
    assert summary["base_validity_rate"] == 2 / 3
    assert summary["conditional_eligible_n"] == 1
    assert summary["factual_eligibility_rate"] == 1 / 2
    assert summary["conditional_survival"] == 1.0
    assert summary["changed_validity_given_base_valid"] == 1.0
    assert summary["end_to_end_changed_validity"] == 2 / 3
