import pandas as pd
import torch

from robustness_benchmark.core.model import TorchBinaryModel
from robustness_benchmark.methods import apas
from robustness_benchmark.methods.apas import APASAdapter
from robustness_benchmark.methods.registry import make_task


class FixedMCE:
    def __init__(self, candidate: pd.Series):
        self.candidate = candidate
        self.margins: list[float] = []

    def generate_for_instance(self, _instance, **kwargs):
        self.margins.append(float(kwargs["minimum_distance"]))
        return pd.DataFrame([self.candidate])


def test_apas_sample_count_and_successful_generation(monkeypatch):
    X = pd.DataFrame({"a": [-1.0, 1.0], "b": [0.0, 0.0]})
    y = pd.Series([0, 1])
    model = TorchBinaryModel(input_dim=2, hidden_dim=(), seed=0)
    with torch.no_grad():
        linear = model.get_torch_model()[0]
        linear.weight.copy_(torch.tensor([[5.0, 0.0]]))
        linear.bias.zero_()
    task = make_task(model, X, y, seed=7)
    mce = FixedMCE(X.iloc[1])
    monkeypatch.setattr(apas, "_set_default_solver_limit", lambda _seconds: None)
    monkeypatch.setattr(apas, "_reset_default_solver_limit", lambda: None)
    monkeypatch.setattr(apas, "_solver_status", lambda _component: None)
    monkeypatch.setattr(apas, "_time_limit_status", lambda: 9)

    generator = APASAdapter(
        task,
        seed=7,
        delta=0.001,
        confidence=0.9,
        robustness_probability=0.9,
        mce=mce,
    )
    result = generator.generate_for_instance(X.iloc[0])

    assert generator.number_of_samples == 22
    assert not result.empty
    assert mce.margins == [0.01]
    assert generator.last_metadata["method_certified"] is True
    assert generator.last_metadata["apas_outcome"] == "probabilistically_robust"


def test_apas_parameter_samples_are_reused():
    X = pd.DataFrame({"a": [-1.0, 1.0], "b": [0.0, 0.0]})
    y = pd.Series([0, 1])
    model = TorchBinaryModel(input_dim=2, hidden_dim=(), seed=0)
    task = make_task(model, X, y, seed=3)
    generator = APASAdapter(
        task,
        seed=3,
        delta=0.001,
        confidence=0.9,
        robustness_probability=0.9,
        mce=FixedMCE(X.iloc[1]),
    )

    first = generator._parameter_concretizations()
    second = generator._parameter_concretizations()

    assert first is second
    assert all(value.shape[0] == 22 for value in first.values())


def test_apas_confirmation_uses_an_independent_parameter_sample():
    X = pd.DataFrame({"a": [-1.0, 1.0], "b": [0.0, 0.0]})
    y = pd.Series([0, 1])
    model = TorchBinaryModel(input_dim=2, hidden_dim=(), seed=0)
    task = make_task(model, X, y, seed=3)
    generator = APASAdapter(
        task,
        seed=3,
        delta=0.001,
        confidence=0.9,
        robustness_probability=0.9,
        mce=FixedMCE(X.iloc[1]),
    )

    search = generator._parameter_concretizations()
    confirmation = generator._sample_parameter_concretizations(1_000_006)

    assert all(not torch.equal(search[name], confirmation[name]) for name in search)


def test_apas_returns_author_candidate_but_does_not_overclaim_certificate(
    monkeypatch,
):
    X = pd.DataFrame({"a": [-1.0, 1.0], "b": [0.0, 0.0]})
    y = pd.Series([0, 1])
    model = TorchBinaryModel(input_dim=2, hidden_dim=(), seed=0)
    with torch.no_grad():
        linear = model.get_torch_model()[0]
        linear.weight.copy_(torch.tensor([[5.0, 0.0]]))
        linear.bias.zero_()
    task = make_task(model, X, y, seed=7)
    monkeypatch.setattr(apas, "_set_default_solver_limit", lambda _seconds: None)
    monkeypatch.setattr(apas, "_reset_default_solver_limit", lambda: None)
    monkeypatch.setattr(apas, "_solver_status", lambda _component: None)
    monkeypatch.setattr(apas, "_time_limit_status", lambda: 9)
    generator = APASAdapter(
        task,
        seed=7,
        delta=0.001,
        confidence=0.9,
        robustness_probability=0.9,
        mce=FixedMCE(X.iloc[1]),
    )
    monkeypatch.setattr(
        generator,
        "_is_probabilistically_robust",
        lambda _candidate, _desired_output: True,
    )
    monkeypatch.setattr(
        generator,
        "certify_candidate",
        lambda _candidate, _desired_output, *, instance_index: (False, instance_index),
    )

    result = generator.generate_for_instance(X.iloc[0])

    assert not result.empty
    assert generator.last_metadata["apas_author_sample_accepted"] is True
    assert generator.last_metadata["apas_confirmation_passed"] is False
    assert generator.last_metadata["method_certified"] is False
    assert (
        generator.last_metadata["apas_outcome"]
        == "author_sample_accepted_holdout_rejected"
    )
