"""XGBoost: leaf-value perturbation and continued boosting."""

import dataclasses
import math
from typing import Any

import numpy as np
import pandas as pd
import torch

from robustness_benchmark.core.classifiers import Classifier, XGBoostClassifier
from robustness_benchmark.core.model_specs.base import ALL_CHANGE_FAMILIES, ModelSpec
from robustness_benchmark.core.model_specs.torch import scaled_uniform_direction

XGBOOST_ARCHITECTURE_DEPTHS = (1, 2, 3, 4, 5, 6, 8)
XGBOOST_ARCHITECTURE_ESTIMATORS = (25, 50, 100, 200, 300, 500, 800)
XGBOOST_SELECTION_DEPTHS = (3, 4, 6)
XGBOOST_SELECTION_LEARNING_RATES = (0.03, 0.1)
XGBOOST_SELECTION_ESTIMATORS = (100, 200, 400)
XGBOOST_SELECTION_MIN_CHILD_WEIGHTS = (1.0, 5.0)


@dataclasses.dataclass(frozen=True)
class XGBoostSpec(ModelSpec):
    """Gradient-boosted trees; incremental updates continue boosting."""

    model_type = "xgboost"
    change_families = ALL_CHANGE_FAMILIES

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
        (direction,) = scaled_uniform_direction(
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
