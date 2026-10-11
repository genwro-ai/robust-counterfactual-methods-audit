from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from robustness_benchmark.core.classifiers.base import (
    DIFFERENTIABLE,
    TORCH_NETWORK,
    Classifier,
)


class MLPClassifier(Classifier):
    """Deterministic binary multilayer perceptron."""

    model_type = "mlp"
    capabilities = frozenset({DIFFERENTIABLE, TORCH_NETWORK})

    def __init__(
        self,
        input_dim: int,
        hidden_dim: tuple[int, ...] = (16, 16),
        seed: int = 0,
    ):
        self.input_dim = input_dim
        self.hidden_dim = list(hidden_dim)
        self.output_dim = 1
        self.seed = seed
        torch.manual_seed(seed)

        layers: list[nn.Module] = []
        previous = input_dim
        for width in hidden_dim:
            layers.extend((nn.Linear(previous, width), nn.ReLU()))
            previous = width
        layers.extend((nn.Linear(previous, 1), nn.Sigmoid()))
        self._model = nn.Sequential(*layers)

    def train(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame | pd.Series,
        *,
        epochs: int = 300,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        optimizer_name: str = "adam",
        patience: int = 30,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
        **_kwargs: object,
    ) -> None:
        torch.manual_seed(self.seed)
        X_tensor = torch.as_tensor(X.to_numpy(), dtype=torch.float32)
        y_tensor = torch.as_tensor(np.asarray(y), dtype=torch.float32).reshape(-1, 1)
        if optimizer_name == "adam":
            optimizer = torch.optim.Adam(
                self._model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
            )
        elif optimizer_name == "sgd":
            optimizer = torch.optim.SGD(
                self._model.parameters(),
                lr=learning_rate,
                momentum=0.9,
                weight_decay=weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer {optimizer_name!r}")
        criterion = nn.BCELoss()

        use_validation = X_val is not None and y_val is not None
        if use_validation:
            X_val_tensor = torch.as_tensor(X_val.to_numpy(), dtype=torch.float32)
            y_val_tensor = torch.as_tensor(
                np.asarray(y_val), dtype=torch.float32
            ).reshape(-1, 1)

        best_loss = float("inf")
        best_state: dict[str, torch.Tensor] | None = None
        stale_epochs = 0
        for _ in range(epochs):
            self._model.train()
            optimizer.zero_grad()
            loss = criterion(self._model(X_tensor), y_tensor)
            loss.backward()
            optimizer.step()

            self._model.eval()
            with torch.no_grad():
                monitored_loss = (
                    criterion(self._model(X_val_tensor), y_val_tensor).item()
                    if use_validation
                    else loss.item()
                )
            if monitored_loss < best_loss - 1e-6:
                best_loss = monitored_loss
                best_state = {
                    key: value.detach().clone()
                    for key, value in self._model.state_dict().items()
                }
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= patience:
                    break

        if best_state is not None:
            self._model.load_state_dict(best_state)
        self._model.eval()

    @staticmethod
    def _tensor(
        X: pd.DataFrame | pd.Series | np.ndarray | torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(X, torch.Tensor):
            tensor = X.to(dtype=torch.float32)
        elif isinstance(X, (pd.DataFrame, pd.Series)):
            tensor = torch.as_tensor(X.to_numpy(), dtype=torch.float32)
        else:
            tensor = torch.as_tensor(X, dtype=torch.float32)
        return tensor.reshape(1, -1) if tensor.ndim == 1 else tensor

    def predict(
        self,
        X: pd.DataFrame | pd.Series | np.ndarray | torch.Tensor,
    ) -> pd.DataFrame:
        with torch.no_grad():
            probabilities = self.predict_proba_tensor(self._tensor(X)).cpu().numpy()
        return pd.DataFrame(probabilities)

    def predict_proba_tensor(self, X: torch.Tensor) -> torch.Tensor:
        return self._model(X)

    def get_torch_model(self) -> nn.Sequential:
        return self._model

    def checkpoint_payload(self) -> dict[str, Any]:
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "seed": self.seed,
            "state_dict": self._model.state_dict(),
        }

    @classmethod
    def from_checkpoint_payload(cls, payload: dict[str, Any]) -> "MLPClassifier":
        model = cls(
            input_dim=int(payload["input_dim"]),
            hidden_dim=tuple(int(width) for width in payload["hidden_dim"]),
            seed=int(payload["seed"]),
        )
        model.get_torch_model().load_state_dict(payload["state_dict"])
        model.get_torch_model().eval()
        return model
