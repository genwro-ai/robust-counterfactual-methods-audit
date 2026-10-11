import numpy as np
import pandas as pd
import pytest
import torch
from gurobipy import GRB

from robustness_benchmark.core.classifiers import TorchBinaryModel
from robustness_benchmark.methods.calibration import rnce_method_config
from robustness_benchmark.methods.configuration import LOCKED_CONFIGS
from robustness_benchmark.methods.milp.delta_robustness import DeltaRobustnessEvaluator
from robustness_benchmark.methods.registry import (
    create_generator,
    generate_counterfactuals,
    make_task,
)
from robustness_benchmark.methods.rnce import RNCE_INTERVAL_ENCODING


def minimum_pgd_logit(model, instance, delta, bias_delta, restarts=8, steps=100):
    network = model.get_torch_model()
    named_parameters = list(network.named_parameters())
    parameters = [parameter for _, parameter in named_parameters]
    radii = [
        bias_delta if name.endswith(".bias") else delta for name, _ in named_parameters
    ]
    original_parameters = [parameter.detach().clone() for parameter in parameters]
    input_tensor = torch.as_tensor(instance, dtype=torch.float32).reshape(1, -1)
    minimum = float("inf")
    torch.manual_seed(0)
    try:
        for _ in range(restarts):
            with torch.no_grad():
                for parameter, original, radius in zip(
                    parameters, original_parameters, radii, strict=True
                ):
                    perturbation = torch.empty_like(parameter).uniform_(-radius, radius)
                    parameter.copy_(original + perturbation)
            for _ in range(steps):
                network.zero_grad(set_to_none=True)
                logit = network[:-1](input_tensor).squeeze()
                logit.backward()
                with torch.no_grad():
                    for parameter, original, radius in zip(
                        parameters, original_parameters, radii, strict=True
                    ):
                        parameter.add_(parameter.grad.sign(), alpha=-radius / 20)
                        parameter.clamp_(original - radius, original + radius)
            with torch.no_grad():
                minimum = min(minimum, float(network[:-1](input_tensor).item()))
    finally:
        with torch.no_grad():
            for parameter, original in zip(
                parameters, original_parameters, strict=True
            ):
                parameter.copy_(original)
    return minimum


def test_negative_input_uses_sign_aware_first_layer_interval():
    model = TorchBinaryModel(input_dim=1, hidden_dim=(1,), seed=0)
    network = model.get_torch_model()
    with torch.no_grad():
        network[0].weight.fill_(-1.0)
        network[0].bias.zero_()
        network[2].weight.fill_(1.0)
        network[2].bias.zero_()

    X = pd.DataFrame({"feature": [-1.0]})
    y = pd.Series([1], name="target")
    task = make_task(model, X, y, seed=0)
    evaluator = DeltaRobustnessEvaluator(task)

    robust = evaluator.evaluate(
        X.iloc[0],
        desired_output=1,
        delta=0.5,
        bias_delta=0.0,
    )

    assert evaluator.opt.gurobiModel.status == GRB.OPTIMAL
    assert evaluator.opt.outputNode.X == pytest.approx(0.25)
    assert robust is True
    assert (
        minimum_pgd_logit(
            model,
            X.iloc[0].to_numpy(),
            delta=0.5,
            bias_delta=0.0,
        )
        > 0
    )


def test_rnce_rejects_unknown_interval_encoding():
    model = TorchBinaryModel(input_dim=1, hidden_dim=(1,), seed=0)
    X = pd.DataFrame({"feature": [-1.0]})
    task = make_task(model, X, pd.Series([1]), seed=0)
    generator, kwargs = create_generator(
        "rnce",
        task,
        method_config={
            "delta": 0.005,
            "bias_delta": 0.005,
            "interval_encoding": "legacy",
        },
    )

    with pytest.raises(ValueError, match="Unsupported RNCE interval encoding"):
        generator.generate_for_instance(X.iloc[0], **kwargs)

    assert RNCE_INTERVAL_ENCODING == "sign_aware_first_layer_v2"


def test_logistic_interval_bound_matches_closed_form_on_negative_inputs():
    # Without a hidden layer the output layer reads the inputs directly. The
    # worst case over |dw| <= delta, |db| <= bias_delta is
    # w.x + b - delta * ||x||_1 - bias_delta.
    model = TorchBinaryModel(input_dim=2, hidden_dim=(), seed=0)
    linear = model.get_torch_model()[0]
    with torch.no_grad():
        linear.weight.copy_(torch.tensor([[1.0, -2.0]]))
        linear.bias.fill_(0.5)
    X = pd.DataFrame({"a": [2.0], "b": [-0.5]})
    task = make_task(model, X, pd.Series([1]), seed=0)
    evaluator = DeltaRobustnessEvaluator(task)

    robust = evaluator.evaluate(X.iloc[0], desired_output=1, delta=0.5, bias_delta=0.1)

    assert evaluator.opt.gurobiModel.status == GRB.OPTIMAL
    assert evaluator.opt.outputNode.X == pytest.approx(3.5 - 0.5 * 2.5 - 0.1)
    assert robust is True
    assert minimum_pgd_logit(
        model, X.iloc[0].to_numpy(), delta=0.5, bias_delta=0.1
    ) == pytest.approx(2.15, abs=1e-4)


def sum_logit_task():
    """logit = a + b - 1; of the training points only (3, 0) and (2, 2) are robust."""

    model = TorchBinaryModel(input_dim=2, hidden_dim=(), seed=0)
    linear = model.get_torch_model()[0]
    with torch.no_grad():
        linear.weight.fill_(1.0)
        linear.bias.fill_(-1.0)
    X = pd.DataFrame({"a": [0.0, 3.0, 2.0], "b": [0.0, 0.0, 2.0]})
    return X, make_task(model, X, pd.Series([0, 1, 1]), seed=0)


def test_rnce_picks_the_l1_nearest_robust_neighbour():
    # From (0, 0), (3, 0) is nearer in L1 (3 < 4); (2, 2) is nearer in L2.
    X, task = sum_logit_task()

    counterfactuals, _ = generate_counterfactuals(
        "rnce", task, X.iloc[[0]], method_config={"optimal": False}
    )

    assert counterfactuals.iloc[0].tolist() == [3.0, 0.0]


def test_rnce_line_search_returns_the_certified_point_nearest_the_factual():
    # On the segment s * (3, 0) the certified logit is 2.985 s - 1.005, so the
    # 0.05 grid first certifies s = 0.35 (s = 0.30 fails).
    X, task = sum_logit_task()

    counterfactuals, (row,) = generate_counterfactuals(
        "rnce", task, X.iloc[[0]], method_config={"optimal": True}
    )

    np.testing.assert_allclose(counterfactuals.iloc[0], [1.05, 0.0])
    assert row["base_valid"]


def test_rnce_delta_comes_from_its_calibration():
    calibration = {"selected_delta": 0.03, "calibration_fingerprint": "abc"}

    config = rnce_method_config(LOCKED_CONFIGS["rnce"], calibration)

    assert config["delta"] == config["bias_delta"] == pytest.approx(0.03)
    assert config["optimal"] is True
    assert config["calibration_fingerprint"] == "abc"
