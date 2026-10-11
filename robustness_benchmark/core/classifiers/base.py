from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any, ClassVar

import numpy as np
import pandas as pd

# Capabilities a CFE method may require from the classifier it explains.
DIFFERENTIABLE = "differentiable"  # predict_proba_tensor with autograd
TORCH_NETWORK = "torch_network"  # get_torch_model, input_dim/hidden_dim/output_dim
LINEAR = "linear"  # linear_parameters: P(y=1) = sigmoid(w @ x + b)


class Classifier(ABC):
    """Binary classifier shared by every model class in the benchmark.

    ``predict`` returns a one-column frame of P(y=1); every method, metric and
    model-bank routine relies on that contract. Subclasses declare which
    optional capabilities they provide so that methods needing gradients or
    network weights can be gated per model class.
    """

    model_type: ClassVar[str]
    capabilities: ClassVar[frozenset[str]] = frozenset()
    input_dim: int
    seed: int

    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs: object) -> None:
        """Fit the model in place."""

    @abstractmethod
    def predict(self, X) -> pd.DataFrame:
        """Return P(y=1) as a one-column frame."""

    @abstractmethod
    def checkpoint_payload(self) -> dict[str, Any]:
        """Return a ``torch.save(..., weights_only=True)``-compatible state."""

    @classmethod
    @abstractmethod
    def from_checkpoint_payload(cls, payload: dict[str, Any]) -> "Classifier":
        """Restore a model from ``checkpoint_payload`` output."""

    def predict_single(self, X) -> int:
        return int(self.predict(X).iloc[0, 0] >= 0.5)

    def predict_proba(self, X) -> pd.DataFrame:
        p1 = self.predict(X).iloc[:, 0]
        return pd.DataFrame({0: 1.0 - p1, 1: p1})

    def evaluate(self, X: pd.DataFrame, y: pd.DataFrame | pd.Series) -> float:
        predicted = (self.predict(X).iloc[:, 0].to_numpy() >= 0.5).astype(int)
        return float(np.mean(predicted == np.asarray(y)))


def missing_capabilities(
    model_or_capabilities: object, required: Iterable[str]
) -> set[str]:
    """Return the required capabilities a model (or capability set) lacks."""

    if isinstance(model_or_capabilities, (set, frozenset)):
        available = model_or_capabilities
    else:
        available = getattr(model_or_capabilities, "capabilities", frozenset())
    return set(required) - set(available)


def predict_crisp(model: Classifier, values: np.ndarray) -> np.ndarray:
    """Hard labels at the 0.5 threshold, as a 1-D integer array."""

    probability = model.predict(np.asarray(values)).iloc[:, 0].to_numpy()
    return (probability >= 0.5).astype(int)


def predict_proba(model: Classifier, values: np.ndarray) -> np.ndarray:
    """Two-column class probabilities as an array."""

    return model.predict_proba(np.asarray(values)).to_numpy()
