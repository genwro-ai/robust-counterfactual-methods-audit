import numpy as np
import pandas as pd
import pytest

from robustness_benchmark.core.model import TorchBinaryModel
from robustness_benchmark.core.training import make_betarce_generation_model
from robustness_benchmark.evaluation.aggregate import generation_metrics
from robustness_benchmark.methods import registry as methods
from robustness_benchmark.methods.betarce import (
    BetaRCEResult,
    beta_left_bound_batch,
    beta_robustness_mask,
)
from robustness_benchmark.methods.registry import generate_counterfactuals, make_task
from robustness_benchmark.methods.robust_adapters import BetaRCEAdapter
from robustness_benchmark.methods.robx import tau_from_stability_quantile


class ExplodingGenerator:
    def generate_for_instance(self, *_args, **_kwargs):
        raise RuntimeError("expected test failure")


class EmptyGenerator:
    def generate_for_instance(self, *_args, **_kwargs):
        return pd.DataFrame(columns=["a", "b"])


class FactualEchoGenerator:
    """Mimic RNCE's fallback of returning the factual unchanged."""

    def generate_for_instance(self, factual, **_kwargs):
        return pd.DataFrame([factual.to_numpy()], columns=factual.index)


def test_generation_coverage_is_separate_from_validity():
    rows = pd.DataFrame(
        {
            "generation_status": ["success", "base_invalid", "exception"],
            "base_valid": [True, False, False],
        }
    )

    summary = generation_metrics(rows)

    assert summary["requested_n"] == 3
    assert summary["generated_n"] == 2
    assert summary["generation_coverage"] == 2 / 3
    assert summary["validity_given_generated"] == 1 / 2
    assert summary["end_to_end_validity"] == 1 / 3


def test_generator_exception_is_missing_and_not_a_factual(monkeypatch):
    X = pd.DataFrame({"a": [0.0, 1.0], "b": [0.0, 1.0]})
    y = pd.Series([0, 1])
    model = TorchBinaryModel(input_dim=2, hidden_dim=(), seed=0)
    task = make_task(model, X, y, seed=0)
    monkeypatch.setattr(
        methods, "create_generator", lambda *_args: (ExplodingGenerator(), {})
    )

    counterfactuals, records = generate_counterfactuals(
        "broken",
        task,
        X.iloc[[0]],
        factual_ids=[4],
        dataset_row_ids=[99],
    )

    assert counterfactuals.iloc[0].isna().all()
    assert records[0]["generation_status"] == "exception"
    assert records[0]["base_valid"] is False
    assert records[0]["factual_id"] == 4
    assert records[0]["dataset_row_id"] == 99


def test_robx_adaptive_variants_require_resolved_tau():
    X = pd.DataFrame({"a": [0.0, 1.0], "b": [0.0, 1.0]})
    y = pd.Series([0, 1])
    model = TorchBinaryModel(input_dim=2, hidden_dim=(), seed=0)
    task = make_task(model, X, y, seed=0)

    with pytest.raises(ValueError, match="requires a calibrated tau"):
        methods.create_generator("robx_balanced", task)

    lower, lower_kwargs = methods.create_generator(
        "robx_balanced", task, method_config={"tau": 0.71}
    )
    upper, upper_kwargs = methods.create_generator(
        "robx_robust", task, method_config={"tau": 0.89}
    )

    assert lower_kwargs == {}
    assert upper_kwargs == {}
    assert lower.tau == 0.71
    assert upper.tau == 0.89


def test_robx_tau_uses_training_stability_quantile_and_probability_floor():
    scores = np.array([0.40, 0.55, 0.70, 0.85, np.nan])

    balanced_tau, balanced = tau_from_stability_quantile(scores, quantile=0.5)
    low_tau, low = tau_from_stability_quantile(scores, quantile=0.1)

    assert balanced_tau == pytest.approx(0.625)
    assert balanced["qualifying_training_points"] == 2
    assert low_tau == 0.5
    assert low["raw_tau"] < 0.5


def test_betarce_paper_confidence_requires_unanimous_32_model_ensemble():
    point = np.zeros((1, 2))
    all_success = [lambda values: np.ones(len(values), dtype=int) for _ in range(32)]
    one_failure = [
        *all_success[:31],
        lambda values: np.zeros(len(values), dtype=int),
    ]

    all_success_bound = beta_left_bound_batch(point, all_success, 1, alpha=0.95)
    one_failure_bound = beta_left_bound_batch(point, one_failure, 1, alpha=0.95)

    assert all_success_bound[0] > 0.9
    assert one_failure_bound[0] < 0.9
    assert beta_robustness_mask(point, all_success, 1, alpha=0.95, delta=0.9)[0]
    assert not beta_robustness_mask(point, one_failure, 1, alpha=0.95, delta=0.9)[0]


def test_betarce_adapter_robustifies_a_base_counterfactual(monkeypatch):
    class TargetModel:
        def predict(self, values):
            return pd.DataFrame(np.ones((len(values), 1)))

    task = make_task(
        TargetModel(),
        pd.DataFrame({"a": [0.0, 1.0], "b": [0.0, 1.0]}),
        pd.Series([0, 1]),
        seed=7,
    )
    adapter = BetaRCEAdapter(
        task,
        [TargetModel() for _ in range(32)],
        seed=7,
        base_gs_max_iter=200,
        gs_max_iter=100,
    )
    base_counterfactual = np.array([0.4, 0.6])
    robust_counterfactual = np.array([0.5, 0.7])
    observed = {}

    base_search = {}

    def fake_base_search(*_args, **kwargs):
        base_search.update(kwargs)
        return base_counterfactual

    monkeypatch.setattr(
        "robustness_benchmark.methods.robust_adapters.growing_spheres_search",
        fake_base_search,
    )

    class StubBetaRCE:
        def generate(self, start_instance, **kwargs):
            observed["start_instance"] = start_instance
            observed.update(kwargs)
            return BetaRCEResult(robust_counterfactual, {"status": "ok"})

    adapter.generator = StubBetaRCE()
    result = adapter.generate_for_instance(pd.Series({"a": 0.0, "b": 0.0}))

    np.testing.assert_array_equal(observed["start_instance"], base_counterfactual)
    assert observed["delta"] == 0.9
    assert observed["alpha"] == 0.95
    assert base_search["max_iter"] == 200
    assert observed["gs_max_iter"] == 100
    np.testing.assert_array_equal(result.iloc[0].to_numpy(), robust_counterfactual)


def test_betarce_bootstrap_members_keep_base_initialization(monkeypatch):
    observed = {}

    def fake_fit_model(X_train, y_train, X_val, y_val, **kwargs):
        observed["X_train"] = X_train.copy()
        observed["y_train"] = y_train.copy()
        observed["X_val"] = X_val
        observed["y_val"] = y_val
        observed.update(kwargs)
        return object()

    monkeypatch.setattr("robustness_benchmark.core.training.fit_model", fake_fit_model)

    class Split:
        X_train = pd.DataFrame({"a": np.arange(12)})
        y_train = pd.Series(np.arange(12) % 2)
        X_val = pd.DataFrame({"a": [12, 13]})
        y_val = pd.Series([0, 1])

    result = make_betarce_generation_model(
        Split(),
        base_seed=2026,
        member=7,
        hidden_dim=(32, 32),
        epochs=300,
    )

    assert result is not None
    assert observed["seed"] == 2026
    assert observed["hidden_dim"] == (32, 32)
    assert observed["epochs"] == 300
    assert len(observed["X_train"]) == len(Split.X_train)
    assert not observed["X_train"]["a"].equals(Split.X_train["a"])


def test_apas_configuration_is_explicit():
    X = pd.DataFrame({"a": [0.0, 1.0], "b": [0.0, 1.0]})
    y = pd.Series([0, 1])
    model = TorchBinaryModel(input_dim=2, hidden_dim=(), seed=0)
    task = make_task(model, X, y, seed=0)

    generator, kwargs = methods.create_generator(
        "apas",
        task,
        method_config={
            "delta": 0.0123,
            "confidence": 0.999,
            "robustness_probability": 0.995,
            "initial_margin": 0.01,
            "increment": 0.2,
            "maximum_margin": 20.0,
            "wall_time_seconds": 30.0,
        },
    )

    assert kwargs == {}
    assert generator.number_of_samples == 1_379
    assert generator.delta == 0.0123


def test_apas_has_no_implicit_fixed_delta():
    X = pd.DataFrame({"a": [0.0, 1.0], "b": [0.0, 1.0]})
    y = pd.Series([0, 1])
    model = TorchBinaryModel(input_dim=2, hidden_dim=(), seed=0)
    task = make_task(model, X, y, seed=0)

    with pytest.raises(ValueError, match="delta must be provided explicitly"):
        methods.create_generator("apas", task, method_config={})


def test_no_counterfactual_is_not_reported_as_an_exception(monkeypatch):
    X = pd.DataFrame({"a": [0.0, 1.0], "b": [0.0, 1.0]})
    y = pd.Series([0, 1])
    model = TorchBinaryModel(input_dim=2, hidden_dim=(), seed=0)
    task = make_task(model, X, y, seed=0)
    monkeypatch.setattr(
        methods, "create_generator", lambda *_args: (EmptyGenerator(), {})
    )

    _, records = generate_counterfactuals("empty", task, X.iloc[[0]])

    assert records[0]["generation_status"] == "no_counterfactual"
    assert records[0]["generation_error"] is None
    assert records[0]["l1_scaled_mean"] is None


def test_factual_returned_unchanged_is_no_counterfactual(monkeypatch):
    X = pd.DataFrame({"a": [0.0, 1.0], "b": [0.0, 1.0]})
    y = pd.Series([0, 1])
    model = TorchBinaryModel(input_dim=2, hidden_dim=(), seed=0)
    task = make_task(model, X, y, seed=0)
    monkeypatch.setattr(
        methods, "create_generator", lambda *_args: (FactualEchoGenerator(), {})
    )

    counterfactuals, records = generate_counterfactuals("echo", task, X.iloc[[0]])

    assert counterfactuals.iloc[0].isna().all()
    assert records[0]["generation_status"] == "no_counterfactual"
    assert records[0]["base_valid"] is False
    assert records[0]["generation_error"] is None
