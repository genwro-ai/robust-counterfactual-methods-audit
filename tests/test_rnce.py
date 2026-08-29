import pandas as pd
import pytest
import torch
from gurobipy import GRB

from robustness_benchmark.core.model import TorchBinaryModel
from robustness_benchmark.methods.delta_robustness import DeltaRobustnessEvaluator
from robustness_benchmark.methods.registry import create_generator, make_task
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
