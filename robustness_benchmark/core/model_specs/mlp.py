"""The MLP: a prespecified architecture with Adam or SGD training."""

import copy
import dataclasses
from typing import Any

import pandas as pd

from robustness_benchmark.core.classifiers import Classifier, MLPClassifier
from robustness_benchmark.core.model_specs.base import ALL_CHANGE_FAMILIES
from robustness_benchmark.core.model_specs.torch import TorchSpec

ADAM_TRAINING_CONFIGS = tuple(
    ("adam", learning_rate, weight_decay)
    for learning_rate in (2.5e-4, 5e-4, 1e-3, 2e-3, 4e-3)
    for weight_decay in (0.0, 1e-5, 1e-4, 1e-3)
    if (learning_rate, weight_decay) != (1e-3, 0.0)
)
SGD_TRAINING_CONFIGS = (
    ("sgd", 1e-2, 0.0),
    ("sgd", 1e-2, 1e-4),
    ("sgd", 1e-2, 1e-3),
    ("sgd", 3e-2, 0.0),
    ("sgd", 3e-2, 1e-4),
    ("sgd", 3e-2, 1e-3),
)
TRAINING_CONFIGS = ADAM_TRAINING_CONFIGS + SGD_TRAINING_CONFIGS
# Reserve candidates, used only when grid settings reproduce the base model.
RESERVE_TRAINING_CONFIGS = (
    ("adam", 1.25e-4, 0.0),
    ("adam", 8e-3, 0.0),
    ("adam", 8e-3, 1e-3),
    ("sgd", 1e-1, 0.0),
    ("sgd", 1e-1, 1e-4),
)

MLP_ARCHITECTURE_POOL: tuple[tuple[int, ...], ...] = (
    (8,),
    (16,),
    (32,),
    (64,),
    (128,),
    (8, 8),
    (8, 16),
    (8, 32),
    (16, 8),
    (16, 16),
    (16, 32),
    (32, 8),
    (32, 16),
    (32, 32),
    (32, 64),
    (64, 16),
    (64, 32),
    (64, 64),
    (128, 32),
    (128, 64),
    (128, 128),
    (8, 8, 8),
    (16, 16, 8),
    (16, 16, 16),
    (32, 16, 8),
    (32, 32, 16),
    (32, 32, 32),
    (64, 32, 16),
    (64, 64, 32),
    (16, 16, 16, 16),
    (32, 32, 32, 32),
)
MLP_SELECTION_ARCHITECTURES = ((16, 16), (32, 16), (32, 32), (64, 32), (64, 64))


@dataclasses.dataclass(frozen=True)
class MLPSpec(TorchSpec):
    model_type = "mlp"
    change_families = ALL_CHANGE_FAMILIES

    hidden_dim: tuple[int, ...] = (32, 32)
    epochs: int = 300
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    optimizer_name: str = "adam"
    patience: int = 30

    def __post_init__(self) -> None:
        object.__setattr__(self, "hidden_dim", tuple(int(w) for w in self.hidden_dim))

    def build(self, input_dim: int, seed: int) -> MLPClassifier:
        return MLPClassifier(input_dim, hidden_dim=self.hidden_dim, seed=seed)

    def training_kwargs(self) -> dict[str, Any]:
        return {
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "optimizer_name": self.optimizer_name,
            "patience": self.patience,
        }

    def id_fragment(self) -> str:
        return f"mlp_{'_'.join(map(str, self.hidden_dim))}"

    def training_config_variants(self) -> list[tuple[str, dict[str, Any]]]:
        return [
            (
                f"{optimizer_name}_lr_{learning_rate:g}_wd_{weight_decay:g}",
                {
                    "optimizer_name": optimizer_name,
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                },
            )
            for optimizer_name, learning_rate, weight_decay in (
                TRAINING_CONFIGS + RESERVE_TRAINING_CONFIGS
            )
            if (optimizer_name, learning_rate, weight_decay)
            != (self.optimizer_name, self.learning_rate, self.weight_decay)
        ]

    def architecture_variants(self) -> list[tuple[str, dict[str, Any]]]:
        return [
            ("_".join(map(str, architecture)), {"hidden_dim": architecture})
            for architecture in MLP_ARCHITECTURE_POOL
            if architecture != self.hidden_dim
        ]

    def incremental_update(
        self,
        model: Classifier,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> Classifier:
        """Rerun the training procedure on ``(X, y)`` from the deployed weights."""

        updated = copy.deepcopy(self._network(model))
        updated.train(X, y, X_val=X_val, y_val=y_val, **self.training_kwargs())
        return updated

    @classmethod
    def selection_grid(cls, base: "MLPSpec") -> list["MLPSpec"]:
        return [base.with_overrides(hidden_dim=h) for h in MLP_SELECTION_ARCHITECTURES]
