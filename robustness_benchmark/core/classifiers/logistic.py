from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression

from robustness_benchmark.core.classifiers.base import LINEAR
from robustness_benchmark.core.classifiers.mlp import MLPClassifier


class LogisticClassifier(MLPClassifier):
    """Logistic regression fitted by scikit-learn, evaluated as a torch layer.

    The coefficients live in a zero-hidden-layer MLP, so gradients, interval
    abstractions and parameter perturbations apply unchanged. ``l1_ratio`` is
    0 for L2, 1 for L1, and in between for elastic net.
    """

    model_type = "logistic"
    capabilities = frozenset({*MLPClassifier.capabilities, LINEAR})

    def __init__(
        self,
        input_dim: int,
        seed: int = 0,
        *,
        C: float = 1.0,
        l1_ratio: float = 0.0,
        class_weight: str | None = None,
        max_iter: int = 5_000,
    ):
        super().__init__(input_dim, hidden_dim=(), seed=seed)
        self.hyperparameters: dict[str, Any] = {
            "C": float(C),
            "l1_ratio": float(l1_ratio),
            "class_weight": class_weight,
            "max_iter": int(max_iter),
        }

    def train(
        self, X: pd.DataFrame, y: pd.DataFrame | pd.Series, **_kwargs: object
    ) -> None:
        l1_ratio = self.hyperparameters["l1_ratio"]
        estimator = LogisticRegression(
            C=self.hyperparameters["C"],
            l1_ratio=l1_ratio,
            class_weight=self.hyperparameters["class_weight"],
            solver="lbfgs" if l1_ratio == 0.0 else "saga",
            max_iter=self.hyperparameters["max_iter"],
            random_state=self.seed,
        )
        estimator.fit(np.asarray(X, dtype=np.float64), np.asarray(y).astype(int))
        linear = self._model[0]
        with torch.no_grad():
            linear.weight.copy_(torch.as_tensor(estimator.coef_, dtype=torch.float32))
            linear.bias.copy_(
                torch.as_tensor(estimator.intercept_, dtype=torch.float32)
            )
        self._model.eval()

    def linear_parameters(self) -> tuple[np.ndarray, float]:
        """Return ``(w, b)`` such that P(y=1) = sigmoid(w @ x + b)."""

        linear = self._model[0]
        return (
            linear.weight.detach().cpu().numpy().reshape(-1).astype(float),
            float(linear.bias.item()),
        )

    def checkpoint_payload(self) -> dict[str, Any]:
        return {
            **super().checkpoint_payload(),
            "hyperparameters": dict(self.hyperparameters),
        }

    @classmethod
    def from_checkpoint_payload(cls, payload: dict[str, Any]) -> "LogisticClassifier":
        model = cls(
            input_dim=int(payload["input_dim"]),
            seed=int(payload["seed"]),
            **payload["hyperparameters"],
        )
        model.get_torch_model().load_state_dict(payload["state_dict"])
        model.get_torch_model().eval()
        return model
