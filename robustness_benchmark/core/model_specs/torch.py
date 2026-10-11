"""Parameter perturbation and incremental updates of torch networks."""

import copy

import numpy as np
import pandas as pd
import torch

from robustness_benchmark.core.classifiers import Classifier, MLPClassifier
from robustness_benchmark.core.model_specs.base import ModelSpec


def scaled_uniform_direction(
    shapes: list[torch.Size], radius: float, seed: int
) -> list[torch.Tensor]:
    """Draw one uniform direction whose largest component equals ``radius``."""

    generator = torch.Generator().manual_seed(seed)
    directions = [
        torch.empty(shape, dtype=torch.float32).uniform_(-1.0, 1.0, generator=generator)
        for shape in shapes
    ]
    maximum = max(float(direction.abs().max().item()) for direction in directions)
    return [direction * (radius / maximum) for direction in directions]


def perturb_torch_parameters(
    model: MLPClassifier, *, radius: float, seed: int
) -> MLPClassifier:
    """Return a copy with every network parameter moved within an l_inf radius."""

    if radius <= 0:
        raise ValueError("radius must be positive")
    perturbed = copy.deepcopy(model)
    parameters = list(perturbed.get_torch_model().parameters())
    directions = scaled_uniform_direction(
        [parameter.shape for parameter in parameters], radius, seed
    )
    with torch.no_grad():
        for parameter, direction in zip(parameters, directions, strict=True):
            parameter.add_(direction.to(dtype=parameter.dtype))
    perturbed.get_torch_model().eval()
    return perturbed


def torch_incremental_update(
    model: MLPClassifier,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    epochs: int = 1,
    batch_size: int = 8,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
) -> MLPClassifier:
    """Return a copy updated by mini-batch Adam passes over ``(X, y)`` in order."""

    if epochs < 1:
        raise ValueError("epochs must be positive")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if weight_decay < 0:
        raise ValueError("weight_decay must be nonnegative")
    if len(X) < 1:
        raise ValueError("the update sample is empty")
    X_tensor = torch.as_tensor(X.to_numpy(), dtype=torch.float32)
    y_tensor = torch.as_tensor(np.asarray(y), dtype=torch.float32).reshape(-1, 1)
    updated = copy.deepcopy(model)
    network = updated.get_torch_model()
    optimizer = torch.optim.Adam(
        network.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    criterion = torch.nn.BCELoss()
    for _ in range(epochs):
        network.train()
        for start in range(0, len(X_tensor), batch_size):
            stop = min(start + batch_size, len(X_tensor))
            optimizer.zero_grad()
            loss = criterion(network(X_tensor[start:stop]), y_tensor[start:stop])
            loss.backward()
            optimizer.step()
    network.eval()
    return updated


class TorchSpec(ModelSpec):
    """Parameter scale and perturbation through the torch network."""

    @staticmethod
    def _network(model: Classifier) -> MLPClassifier:
        if not isinstance(model, MLPClassifier):
            raise TypeError("expected a torch network")
        return model

    def parameter_scale(self, model: Classifier) -> float:
        parameters = self._network(model).get_torch_model().parameters()
        return float(
            torch.cat([p.detach().abs().reshape(-1) for p in parameters]).mean()
        )

    def perturb_parameters(
        self, model: Classifier, *, radius: float, seed: int
    ) -> Classifier:
        return perturb_torch_parameters(self._network(model), radius=radius, seed=seed)
