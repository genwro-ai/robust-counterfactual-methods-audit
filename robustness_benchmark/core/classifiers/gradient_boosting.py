import json
from types import ModuleType
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from robustness_benchmark.core.classifiers.base import Classifier

if TYPE_CHECKING:
    import xgboost as xgb

_xgboost_module: ModuleType | None = None


def _xgboost() -> ModuleType:
    """Import xgboost on first use, single-threaded together with torch.

    On macOS xgboost and torch load different OpenMP runtimes, and threaded
    regions in either then segfault. Lazy import keeps MLP and LR runs clean.
    """

    global _xgboost_module
    if _xgboost_module is None:
        import torch
        import xgboost

        xgboost.set_config(nthread=1)
        torch.set_num_threads(1)
        _xgboost_module = xgboost
    return _xgboost_module


def _as_matrix(X) -> np.ndarray:
    if isinstance(X, (pd.DataFrame, pd.Series)):
        values = X.to_numpy()
    elif hasattr(X, "detach"):
        values = X.detach().cpu().numpy()
    else:
        values = np.asarray(X)
    values = values.astype(np.float32, copy=False)
    return values.reshape(1, -1) if values.ndim == 1 else values


class XGBoostClassifier(Classifier):
    """Gradient-boosted trees; the seed acts only through row/column subsampling."""

    model_type = "xgboost"

    def __init__(
        self,
        input_dim: int,
        seed: int = 0,
        *,
        n_estimators: int = 100,
        max_depth: int = 3,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_lambda: float = 1.0,
        reg_alpha: float = 0.0,
        min_child_weight: float = 1.0,
        gamma: float = 0.0,
    ):
        self.input_dim = input_dim
        self.seed = seed
        self.hyperparameters: dict[str, Any] = {
            "n_estimators": int(n_estimators),
            "max_depth": int(max_depth),
            "learning_rate": float(learning_rate),
            "subsample": float(subsample),
            "colsample_bytree": float(colsample_bytree),
            "reg_lambda": float(reg_lambda),
            "reg_alpha": float(reg_alpha),
            "min_child_weight": float(min_child_weight),
            "gamma": float(gamma),
        }
        self.booster: xgb.Booster | None = None

    def booster_params(self) -> dict[str, Any]:
        h = self.hyperparameters
        return {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "hist",
            "max_depth": h["max_depth"],
            "eta": h["learning_rate"],
            "subsample": h["subsample"],
            "colsample_bytree": h["colsample_bytree"],
            "lambda": h["reg_lambda"],
            "alpha": h["reg_alpha"],
            "min_child_weight": h["min_child_weight"],
            "gamma": h["gamma"],
            "seed": self.seed,
            "nthread": 1,
        }

    def train(
        self, X: pd.DataFrame, y: pd.DataFrame | pd.Series, **_kwargs: object
    ) -> None:
        self.booster = _xgboost().train(
            self.booster_params(),
            self._matrix(X, y),
            num_boost_round=self.hyperparameters["n_estimators"],
        )

    def continue_training(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        max_rounds: int,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
        early_stopping_rounds: int = 30,
    ) -> "XGBoostClassifier":
        """Return a copy with new trees boosted on ``(X, y)`` after the existing ones.

        With validation data, boosting stops once validation log-loss has not
        improved for ``early_stopping_rounds`` and keeps the best iteration.
        """

        xgb = _xgboost()
        validation = (
            {
                "evals": [(self._matrix(X_val, y_val), "validation")],
                "early_stopping_rounds": early_stopping_rounds,
                "verbose_eval": False,
            }
            if X_val is not None and y_val is not None
            else {}
        )
        updated = self.clone()
        booster = xgb.train(
            self.booster_params(),
            self._matrix(X, y),
            num_boost_round=max_rounds,
            xgb_model=updated._fitted_booster(),
            **validation,
        )
        if validation:
            booster = booster[: booster.best_iteration + 1]
        updated.booster = booster
        return updated

    @staticmethod
    def _matrix(X, y) -> "xgb.DMatrix":
        return _xgboost().DMatrix(
            _as_matrix(X), label=np.asarray(y).astype(np.float32), nthread=1
        )

    def _fitted_booster(self) -> "xgb.Booster":
        if self.booster is None:
            raise RuntimeError("XGBoostClassifier has not been trained")
        return self.booster

    def predict(self, X) -> pd.DataFrame:
        probability = self._fitted_booster().inplace_predict(_as_matrix(X))
        return pd.DataFrame(np.asarray(probability, dtype=np.float64).reshape(-1, 1))

    @property
    def num_trees(self) -> int:
        return self._fitted_booster().num_boosted_rounds()

    def _model_json(self) -> dict[str, Any]:
        return json.loads(self._fitted_booster().save_raw("json"))

    @staticmethod
    def _leaf_positions(model_json: dict[str, Any]) -> list[tuple[int, int]]:
        trees = model_json["learner"]["gradient_booster"]["model"]["trees"]
        return [
            (tree_index, node)
            for tree_index, tree in enumerate(trees)
            for node, left in enumerate(tree["left_children"])
            if left == -1
        ]

    def leaf_values(self) -> np.ndarray:
        """Return every leaf value, ordered by tree and then by node."""

        model_json = self._model_json()
        trees = model_json["learner"]["gradient_booster"]["model"]["trees"]
        return np.array(
            [
                trees[tree]["split_conditions"][node]
                for tree, node in self._leaf_positions(model_json)
            ],
            dtype=np.float64,
        )

    def with_leaf_offsets(self, offsets: np.ndarray) -> "XGBoostClassifier":
        """Return a copy with leaf values shifted by ``offsets``.

        XGBoost stores a leaf value both as its split condition and its base
        weight, so both are shifted.
        """

        model_json = self._model_json()
        positions = self._leaf_positions(model_json)
        if len(offsets) != len(positions):
            raise ValueError(
                f"Expected {len(positions)} leaf offsets, got {len(offsets)}"
            )
        trees = model_json["learner"]["gradient_booster"]["model"]["trees"]
        for (tree, node), offset in zip(positions, offsets, strict=True):
            trees[tree]["split_conditions"][node] += float(offset)
            trees[tree]["base_weights"][node] += float(offset)
        updated = self.clone()
        updated.booster = self._load_booster(json.dumps(model_json))
        return updated

    def clone(self) -> "XGBoostClassifier":
        return self.from_checkpoint_payload(self.checkpoint_payload())

    @staticmethod
    def _load_booster(model_json: str) -> "xgb.Booster":
        booster = _xgboost().Booster(model_file=bytearray(model_json.encode()))
        booster.set_param({"nthread": 1})
        return booster

    def checkpoint_payload(self) -> dict[str, Any]:
        return {
            "input_dim": self.input_dim,
            "seed": self.seed,
            "hyperparameters": dict(self.hyperparameters),
            "booster_json": self._fitted_booster().save_raw("json").decode(),
        }

    @classmethod
    def from_checkpoint_payload(cls, payload: dict[str, Any]) -> "XGBoostClassifier":
        model = cls(
            input_dim=int(payload["input_dim"]),
            seed=int(payload["seed"]),
            **payload["hyperparameters"],
        )
        model.booster = cls._load_booster(str(payload["booster_json"]))
        return model
