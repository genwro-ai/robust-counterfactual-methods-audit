"""ROAR on a logistic base model (Upadhyay et al., NeurIPS 2021).

ROAR was posed for linear models: it minimizes the binary cross-entropy under
the worst-case shift of the coefficients and intercept inside an L-infinity box,
plus an L1 cost. ``RoarLimeAdapter`` applies that objective to a local LIME
surrogate of a black box; when the base model is itself logistic, the
surrogate is unnecessary and the box is centred on the model's exact
parameters. ROAR-LIME uses this module's optimizer, including its masking of
immutable columns and its stopping rule from AI4LIFE's ``get_recourse``: the
relaxed iterate is returned, and the registry's final projection restores
one-hot and integer validity.
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd
import torch

from robustness_benchmark.methods.configuration import ROAR_IMPLEMENTATION


def worst_case_parameter_shift(
    values: torch.Tensor, target: int, delta: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the exact box-constrained shift that maximizes binary CE loss."""

    direction = -1.0 if target == 1 else 1.0
    weight_shift = direction * delta * torch.sign(values)
    intercept_shift = values.new_tensor(direction * delta)
    return weight_shift, intercept_shift


def worst_case_logit(
    values: torch.Tensor,
    weights: torch.Tensor,
    bias: torch.Tensor,
    target: int,
    delta: float,
) -> torch.Tensor:
    weight_shift, intercept_shift = worst_case_parameter_shift(values, target, delta)
    return torch.dot(weights + weight_shift, values) + bias + intercept_shift


def optimize_roar(
    factual: np.ndarray,
    coefficients: np.ndarray,
    intercept: float,
    *,
    target: int,
    delta: float,
    lambda_param: float,
    learning_rate: float,
    loss_threshold: float,
    maximum_iterations: int,
    frozen: Sequence[int] = (),
) -> tuple[np.ndarray, int, bool, float, float]:
    """Optimize the published ROAR min-max binary-cross-entropy objective.

    ``frozen`` columns get zero gradient, so Adam keeps them at the factual.
    """

    x = torch.as_tensor(factual, dtype=torch.float32)
    weights = torch.as_tensor(coefficients, dtype=torch.float32)
    bias = x.new_tensor(intercept)
    target_tensor = x.new_tensor(float(target))
    candidate = x.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([candidate], lr=learning_rate)

    previous_loss: torch.Tensor | None = None
    converged = False
    iterations = 0
    validity_loss = x.new_tensor(float("nan"))
    cost = x.new_tensor(float("nan"))
    for iterations in range(1, maximum_iterations + 1):
        optimizer.zero_grad()
        worst_logit = worst_case_logit(candidate, weights, bias, target, delta)
        validity_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            worst_logit, target_tensor
        )
        cost = torch.linalg.vector_norm(candidate - x, ord=1)
        loss = validity_loss + lambda_param * cost
        loss.backward()
        if frozen:
            candidate.grad[list(frozen)] = 0.0
        optimizer.step()

        if (
            previous_loss is not None
            and torch.abs(previous_loss - loss.detach()) <= loss_threshold
        ):
            converged = True
            break
        previous_loss = loss.detach().clone()

    return (
        candidate.detach().cpu().numpy(),
        iterations,
        converged,
        float(validity_loss.detach()),
        float(cost.detach()),
    )


class RoarAdapter:
    def __init__(
        self,
        task,
        *,
        delta: float,
        lambda_param: float,
        learning_rate: float,
        loss_threshold: float,
        maximum_iterations: int,
    ) -> None:
        self.model = task.model
        self.frozen = task.schema.immutable_columns()
        self.coefficients, self.intercept = task.model.linear_parameters()
        self.delta = delta
        self.lambda_param = lambda_param
        self.learning_rate = learning_rate
        self.loss_threshold = loss_threshold
        self.maximum_iterations = maximum_iterations
        self.last_metadata: dict[str, object] = {}

    def generate_for_instance(
        self, instance: pd.Series, *, neg_value: int = 0, **_kwargs: object
    ) -> pd.DataFrame:
        ce, iterations, converged, validity_loss, cost = optimize_roar(
            instance.to_numpy(dtype=float),
            self.coefficients,
            self.intercept,
            target=1 - neg_value,
            delta=self.delta,
            lambda_param=self.lambda_param,
            learning_rate=self.learning_rate,
            loss_threshold=self.loss_threshold,
            maximum_iterations=self.maximum_iterations,
            frozen=self.frozen,
        )
        result = pd.DataFrame([ce], columns=instance.index)
        self.last_metadata = {
            "roar_implementation": ROAR_IMPLEMENTATION,
            "roar_delta": self.delta,
            "roar_lambda": self.lambda_param,
            "roar_learning_rate": self.learning_rate,
            "roar_loss": "binary_cross_entropy",
            "roar_uncertainty_set": "linf_box_weights_and_intercept",
            "roar_iterations": iterations,
            "roar_converged": converged,
            "roar_final_worst_case_bce": validity_loss,
            "roar_final_l1": cost,
            "base_ce_probability": float(self.model.predict(result).iloc[0, 0]),
        }
        return result
