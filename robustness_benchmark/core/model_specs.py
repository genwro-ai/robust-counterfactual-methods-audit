"""How each model class is trained, tuned and changed by the model bank."""

import copy
import dataclasses
import itertools
import math
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Self

import numpy as np
import pandas as pd
import torch

from robustness_benchmark.core.classifiers import (
    Classifier,
    LogisticClassifier,
    TorchBinaryModel,
    XGBoostClassifier,
)

FULL_VARIANTS_PER_FAMILY = 25
# Bounded-parameter radii as fractions of the base model's mean |parameter|,
# so the l_inf box means the same relative change for weights and leaves.
RELATIVE_PARAMETER_RADII = (0.05, 0.1, 0.2, 0.3, 0.5)

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

# (name, l1_ratio, class_weight, C values). Candidates are taken round-robin
# across groups, so any prefix mixes penalties and class weighting.
LOGISTIC_CANDIDATE_GROUPS: tuple[
    tuple[str, float, str | None, tuple[float, ...]], ...
] = (
    ("l2", 0.0, None, (1e-3, 3e-3, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100)),
    ("l1", 1.0, None, (0.01, 0.03, 0.1, 0.3, 1, 10)),
    ("elasticnet", 0.5, None, (0.01, 0.03, 0.1, 0.3, 1, 10)),
    ("l2_balanced", 0.0, "balanced", (1e-3, 0.01, 0.1, 1, 10, 100)),
    ("l1_balanced", 1.0, "balanced", (0.01, 0.1, 1, 10)),
    ("elasticnet_balanced", 0.5, "balanced", (0.01, 0.1, 1, 10)),
)
LOGISTIC_SELECTION_C = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)
MLP_SELECTION_ARCHITECTURES = ((16, 16), (32, 16), (32, 32), (64, 32), (64, 64))

XGBOOST_ARCHITECTURE_DEPTHS = (2, 3, 4, 5, 6, 8)
XGBOOST_ARCHITECTURE_ESTIMATORS = (50, 100, 200, 300, 500)
XGBOOST_SELECTION_DEPTHS = (3, 4, 6)
XGBOOST_SELECTION_LEARNING_RATES = (0.03, 0.1)
XGBOOST_SELECTION_ESTIMATORS = (100, 200, 400)
XGBOOST_SELECTION_MIN_CHILD_WEIGHTS = (1.0, 5.0)


def _scaled_uniform_direction(
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
    model: TorchBinaryModel, *, radius: float, seed: int
) -> TorchBinaryModel:
    """Return a copy with every network parameter moved within an l_inf radius."""

    if radius <= 0:
        raise ValueError("radius must be positive")
    perturbed = copy.deepcopy(model)
    parameters = list(perturbed.get_torch_model().parameters())
    directions = _scaled_uniform_direction(
        [parameter.shape for parameter in parameters], radius, seed
    )
    with torch.no_grad():
        for parameter, direction in zip(parameters, directions, strict=True):
            parameter.add_(direction.to(dtype=parameter.dtype))
    perturbed.get_torch_model().eval()
    return perturbed


def torch_incremental_update(
    model: TorchBinaryModel,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    epochs: int = 1,
    batch_size: int = 8,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
) -> TorchBinaryModel:
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


class _TorchSpec(ModelSpec):
    """Parameter scale and perturbation through the torch network."""

    @staticmethod
    def _network(model: Classifier) -> TorchBinaryModel:
        if not isinstance(model, TorchBinaryModel):
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


@dataclasses.dataclass(frozen=True)
class MLPSpec(_TorchSpec):
    model_type = "mlp"
    change_families = (
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

    hidden_dim: tuple[int, ...] = (32, 32)
    epochs: int = 300
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    optimizer_name: str = "adam"
    patience: int = 30

    def __post_init__(self) -> None:
        object.__setattr__(self, "hidden_dim", tuple(int(w) for w in self.hidden_dim))

    def build(self, input_dim: int, seed: int) -> TorchBinaryModel:
        return TorchBinaryModel(input_dim, hidden_dim=self.hidden_dim, seed=seed)

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


@dataclasses.dataclass(frozen=True)
class LogisticSpec(_TorchSpec):
    """Logistic regression.

    Seed, architecture and incremental-update families do not apply: the fit
    is a convex problem with one optimum, so a new seed or a warm start from
    the deployed coefficients reproduces retraining from scratch.
    """

    model_type = "logistic"
    change_families = (
        "bootstrap",
        "deletion",
        "data_addition",
        "label_update",
        "training_config",
        "bounded_parameter",
    )

    C: float = 1.0
    l1_ratio: float = 0.0
    class_weight: str | None = None
    max_iter: int = 5_000

    def build(self, input_dim: int, seed: int) -> LogisticClassifier:
        return LogisticClassifier(
            input_dim,
            seed=seed,
            C=self.C,
            l1_ratio=self.l1_ratio,
            class_weight=self.class_weight,
            max_iter=self.max_iter,
        )

    def id_fragment(self) -> str:
        return f"logistic_C_{self.C:g}"

    def training_config_variants(self) -> list[tuple[str, dict[str, Any]]]:
        groups = [
            [
                (
                    f"{name}_C_{C:g}",
                    {"C": float(C), "l1_ratio": l1_ratio, "class_weight": class_weight},
                )
                for C in values
            ]
            for name, l1_ratio, class_weight, values in LOGISTIC_CANDIDATE_GROUPS
        ]
        return [
            candidate
            for candidate in itertools.chain.from_iterable(
                itertools.zip_longest(*groups)
            )
            if candidate is not None and self.with_overrides(**candidate[1]) != self
        ]

    @classmethod
    def selection_grid(cls, base: "LogisticSpec") -> list["LogisticSpec"]:
        return [base.with_overrides(C=C) for C in LOGISTIC_SELECTION_C]


@dataclasses.dataclass(frozen=True)
class XGBoostSpec(ModelSpec):
    """Gradient-boosted trees; incremental updates continue boosting."""

    model_type = "xgboost"
    change_families = MLPSpec.change_families

    n_estimators: int = 100
    max_depth: int = 3
    learning_rate: float = 0.1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_lambda: float = 1.0
    reg_alpha: float = 0.0
    min_child_weight: float = 1.0
    gamma: float = 0.0

    def build(self, input_dim: int, seed: int) -> XGBoostClassifier:
        return XGBoostClassifier(input_dim, seed=seed, **self.hyperparameters())

    def id_fragment(self) -> str:
        return (
            f"xgboost_d{self.max_depth}_n{self.n_estimators}_lr{self.learning_rate:g}"
        )

    def training_config_variants(self) -> list[tuple[str, dict[str, Any]]]:
        variants: list[tuple[str, dict[str, Any]]] = []
        for eta_multiplier in (0.5, 1.0, 2.0):
            for reg_lambda in (0.1, 1.0, 10.0):
                for min_child_weight in (1.0, 5.0):
                    if (eta_multiplier, reg_lambda, min_child_weight) == (
                        1.0,
                        self.reg_lambda,
                        self.min_child_weight,
                    ):
                        continue
                    variants.append(
                        (
                            (
                                f"eta_x{eta_multiplier:g}_lambda_{reg_lambda:g}"
                                f"_mcw_{min_child_weight:g}"
                            ),
                            {
                                "learning_rate": self.learning_rate * eta_multiplier,
                                "reg_lambda": reg_lambda,
                                "min_child_weight": min_child_weight,
                            },
                        )
                    )
        variants.extend(
            (f"gamma_{gamma:g}", {"gamma": gamma}) for gamma in (0.5, 1.0, 2.0)
        )
        variants.extend(
            (f"alpha_{alpha:g}", {"reg_alpha": alpha}) for alpha in (0.5, 1.0, 5.0)
        )
        variants.extend(
            (f"subsample_{subsample:g}", {"subsample": subsample})
            for subsample in (0.6, 1.0)
        )
        # Reserve candidates, used only when the ones above reproduce the base.
        variants.extend(
            [
                ("eta_x0.25", {"learning_rate": self.learning_rate * 0.25}),
                ("eta_x4", {"learning_rate": self.learning_rate * 4}),
                ("lambda_100", {"reg_lambda": 100.0}),
                ("mcw_10", {"min_child_weight": 10.0}),
                ("gamma_5", {"gamma": 5.0}),
                ("alpha_10", {"reg_alpha": 10.0}),
                ("subsample_0.5", {"subsample": 0.5}),
            ]
        )
        return variants

    def architecture_variants(self) -> list[tuple[str, dict[str, Any]]]:
        def distance(depth: int, estimators: int) -> float:
            return abs(math.log2(depth / self.max_depth)) + abs(
                math.log2(estimators / self.n_estimators)
            )

        grid = [
            (depth, estimators)
            for depth in XGBOOST_ARCHITECTURE_DEPTHS
            for estimators in XGBOOST_ARCHITECTURE_ESTIMATORS
            if (depth, estimators) != (self.max_depth, self.n_estimators)
        ]
        nearest = sorted(grid, key=lambda pair: (distance(*pair), pair))
        return [
            (
                f"depth_{depth}_trees_{estimators}",
                {"max_depth": depth, "n_estimators": estimators},
            )
            for depth, estimators in nearest
        ]

    @staticmethod
    def _booster(model: Classifier) -> XGBoostClassifier:
        if not isinstance(model, XGBoostClassifier):
            raise TypeError("expected an XGBoostClassifier")
        return model

    def parameter_scale(self, model: Classifier) -> float:
        return float(np.abs(self._booster(model).leaf_values()).mean())

    def perturb_parameters(
        self, model: Classifier, *, radius: float, seed: int
    ) -> Classifier:
        if radius <= 0:
            raise ValueError("radius must be positive")
        booster = self._booster(model)
        (direction,) = _scaled_uniform_direction(
            [torch.Size([len(booster.leaf_values())])], radius, seed
        )
        return booster.with_leaf_offsets(direction.double().numpy())

    def incremental_update(
        self,
        model: Classifier,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> Classifier:
        """Boost new trees on ``(X, y)`` after the deployed ones, early-stopped.

        At most ``n_estimators`` trees are added.
        """

        return self._booster(model).continue_training(
            X, y, max_rounds=self.n_estimators, X_val=X_val, y_val=y_val
        )

    @classmethod
    def selection_grid(cls, base: "XGBoostSpec") -> list["XGBoostSpec"]:
        return [
            base.with_overrides(
                max_depth=depth,
                learning_rate=learning_rate,
                n_estimators=estimators,
                min_child_weight=min_child_weight,
            )
            for depth in XGBOOST_SELECTION_DEPTHS
            for learning_rate in XGBOOST_SELECTION_LEARNING_RATES
            for estimators in XGBOOST_SELECTION_ESTIMATORS
            for min_child_weight in XGBOOST_SELECTION_MIN_CHILD_WEIGHTS
        ]


MODEL_SPECS: dict[str, type[ModelSpec]] = {
    spec.model_type: spec for spec in (MLPSpec, LogisticSpec, XGBoostSpec)
}


def spec_from_dict(values: dict[str, Any]) -> ModelSpec:
    fields = dict(values)
    model_type = str(fields.pop("model_type"))
    if model_type not in MODEL_SPECS:
        raise ValueError(f"Unknown model type {model_type!r}")
    return MODEL_SPECS[model_type](**fields)


def spec_from_manifest(manifest: dict[str, Any]) -> ModelSpec:
    """Rebuild a bank's spec; banks without ``model_spec`` are MLP banks."""

    if "model_spec" in manifest:
        return spec_from_dict(manifest["model_spec"])
    return MLPSpec(
        hidden_dim=tuple(int(width) for width in manifest["selected_architecture"]),
        epochs=int(manifest["epochs"]),
    )
