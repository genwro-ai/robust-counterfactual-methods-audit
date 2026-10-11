"""The interface each model class implements for training and changes."""

import dataclasses
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Self

import pandas as pd

from robustness_benchmark.core.classifiers import Classifier

# Every change family of the bank; a model class runs those that apply to it.
ALL_CHANGE_FAMILIES = (
    "seed",
    "bootstrap",
    "deletion",
    "data_addition",
    "label_update",
    "training_config",
    "architecture",
    "bounded_parameter",
    "incremental_update",
)


@dataclasses.dataclass(frozen=True)
class ModelSpec(ABC):
    """Hyperparameters and change-construction rules of one model class."""

    model_type: ClassVar[str]
    change_families: ClassVar[tuple[str, ...]]

    @abstractmethod
    def build(self, input_dim: int, seed: int) -> Classifier:
        """Return an untrained model."""

    @abstractmethod
    def id_fragment(self) -> str:
        """Return the model part of a base-model identifier."""

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        *,
        seed: int,
    ) -> Classifier:
        model = self.build(X.shape[1], seed)
        model.train(X, y, X_val=X_val, y_val=y_val, **self.training_kwargs())
        return model

    def training_kwargs(self) -> dict[str, Any]:
        return {}

    def with_overrides(self, **overrides: Any) -> Self:
        return dataclasses.replace(self, **overrides)

    def hyperparameters(self) -> dict[str, Any]:
        return {
            field.name: (
                list(value)
                if isinstance(value := getattr(self, field.name), tuple)
                else value
            )
            for field in dataclasses.fields(self)
        }

    def to_dict(self) -> dict[str, Any]:
        return {"model_type": self.model_type, **self.hyperparameters()}

    def training_config_variants(self) -> list[tuple[str, dict[str, Any]]]:
        """Ordered (level, overrides) candidates for the training-config family.

        The bank keeps the first 25 that change the base model, so lists hold
        more than 25 and exclude the base configuration itself.
        """

        return []

    def architecture_variants(self) -> list[tuple[str, dict[str, Any]]]:
        """Ordered (level, overrides) candidates for the architecture family."""

        return []

    @abstractmethod
    def parameter_scale(self, model: Classifier) -> float:
        """Mean absolute value of the parameters ``perturb_parameters`` moves."""

    @abstractmethod
    def perturb_parameters(
        self, model: Classifier, *, radius: float, seed: int
    ) -> Classifier:
        """Return a copy with parameters moved by exactly ``radius`` in l_inf."""

    def incremental_update(
        self,
        model: Classifier,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> Classifier:
        """Return a copy of ``model`` updated, not retrained, on ``(X, y)``."""

        raise NotImplementedError(f"{self.model_type} has no incremental update")

    @classmethod
    def selection_grid(cls, base: Self) -> list[Self]:
        """Candidate base configurations, derived from ``base``."""

        return [base]
