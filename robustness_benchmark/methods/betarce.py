"""BetaRCE (Stępka et al.) on top of growing spheres.

The adapter uses the authors' settings: growing spheres with 100 samples,
p = 2, step 0.1 and 100 iterations for both the base counterfactual and the
robust search (``gs`` and ``betarob.fixed_hps.gs`` in BetaRCE's
configs/config_exp.yaml).
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from robustness_benchmark.core.classifiers import predict_crisp
from robustness_benchmark.core.schema import FeatureSchema
from robustness_benchmark.methods.growing_spheres import growing_spheres_search

BETARCE_IMPLEMENTATION = "paper_two_stage_growing_spheres_base100_v4"


def _get_ensemble_predictions_batch(
    instances: np.ndarray,
    estimators_crisp: list[Callable[[np.ndarray], np.ndarray]],
    target_class: int,
) -> np.ndarray:
    instances = np.atleast_2d(instances)
    all_preds = np.array(
        [fn(instances).reshape(-1) for fn in estimators_crisp], dtype=int
    )
    if target_class == 0:
        all_preds = 1 - all_preds
    return all_preds


def beta_left_bound_batch(
    instances: np.ndarray,
    estimators_crisp: list[Callable[[np.ndarray], np.ndarray]],
    target_class: int,
    alpha: float,
) -> np.ndarray:
    preds = _get_ensemble_predictions_batch(instances, estimators_crisp, target_class)
    alpha_prior, beta_prior = 0.5, 0.5
    successes = np.sum(preds == 1, axis=0)
    failures = np.sum(preds == 0, axis=0)
    return stats.beta.ppf(1 - alpha, alpha_prior + successes, beta_prior + failures)


def beta_robustness_mask(
    instances: np.ndarray,
    estimators_crisp: list[Callable[[np.ndarray], np.ndarray]],
    target_class: int,
    alpha: float,
    delta: float,
) -> np.ndarray:
    left_bounds = beta_left_bound_batch(
        instances, estimators_crisp, target_class, alpha
    )
    return left_bounds > delta


@dataclass
class BetaRCEResult:
    counterfactual: np.ndarray | None
    metadata: dict[str, Any]


class BetaRCE:
    def __init__(
        self,
        predict_fn_crisp: Callable[[np.ndarray], np.ndarray],
        estimators_crisp: list[Callable[[np.ndarray], np.ndarray]],
        binary_indices: list[int] | None = None,
        feature_min: np.ndarray | None = None,
        feature_max: np.ndarray | None = None,
        schema: FeatureSchema | None = None,
    ) -> None:
        self.predict_fn_crisp = predict_fn_crisp
        self.schema = schema
        self.estimators_crisp = estimators_crisp
        self.binary_indices = binary_indices
        self.feature_min = feature_min
        self.feature_max = feature_max

    def generate(
        self,
        start_instance: np.ndarray,
        target_class: int,
        delta: float,
        alpha: float,
        gs_n_search_samples: int = 1000,
        gs_p_norm: int = 2,
        gs_step: float = 0.2,
        gs_max_iter: int = 1000,
        rng: np.random.Generator | None = None,
    ) -> BetaRCEResult:
        if rng is None:
            rng = np.random.default_rng()

        start_instance = np.asarray(start_instance).reshape(-1)

        def combined_objective(X: np.ndarray) -> np.ndarray:
            X = np.atleast_2d(X)
            results = np.zeros(X.shape[0], dtype=int)
            valid = self.predict_fn_crisp(X) == target_class
            if np.any(valid):
                robust = beta_robustness_mask(
                    X[valid],
                    self.estimators_crisp,
                    target_class,
                    alpha,
                    delta,
                )
                valid_indices = np.where(valid)[0]
                results[valid_indices[robust]] = 1
            return results

        if combined_objective(start_instance.reshape(1, -1))[0] == 1:
            left_bound = float(
                beta_left_bound_batch(
                    start_instance.reshape(1, -1),
                    self.estimators_crisp,
                    target_class,
                    alpha,
                )[0]
            )
            return BetaRCEResult(
                start_instance, {"status": "start_satisfies", "left_bound": left_bound}
            )

        cf = growing_spheres_search(
            instance=start_instance,
            pred_fn_crisp=combined_objective,
            target_class=1,
            n_search_samples=gs_n_search_samples,
            p_norm=gs_p_norm,
            step=gs_step,
            max_iter=gs_max_iter,
            binary_indices=self.binary_indices,
            feature_min=self.feature_min,
            feature_max=self.feature_max,
            rng=rng,
            schema=self.schema,
        )

        if cf is None:
            return BetaRCEResult(None, {"status": "no_cf_found"})

        left_bound = float(
            beta_left_bound_batch(
                cf.reshape(1, -1),
                self.estimators_crisp,
                target_class,
                alpha,
            )[0]
        )
        return BetaRCEResult(cf, {"status": "ok", "left_bound": left_bound})


class BetaRCEAdapter:
    def __init__(
        self,
        task,
        generation_models: list,
        *,
        seed: int,
        delta: float = 0.9,
        alpha: float = 0.95,
        generation_ensemble_n: int = 32,
        gs_n_search_samples: int = 100,
        gs_p_norm: int = 2,
        gs_step: float = 0.1,
        base_gs_max_iter: int = 100,
        gs_max_iter: int = 100,
    ) -> None:
        if len(generation_models) != generation_ensemble_n:
            raise ValueError(
                "BetaRCE requires exactly "
                f"{generation_ensemble_n} separate generation-ensemble models"
            )
        self.rng = np.random.default_rng(seed)
        self.delta = delta
        self.alpha = alpha
        self.generation_ensemble_n = generation_ensemble_n
        self.gs_n_search_samples = gs_n_search_samples
        self.gs_p_norm = gs_p_norm
        self.gs_step = gs_step
        self.base_gs_max_iter = base_gs_max_iter
        self.gs_max_iter = gs_max_iter
        self.base_predict = lambda values: predict_crisp(task.model, values)
        self.schema = task.schema
        self.generator = BetaRCE(
            self.base_predict,
            [
                lambda values, model=model: predict_crisp(model, values)
                for model in generation_models
            ],
            schema=task.schema,
        )
        self.last_metadata: dict[str, object] = {}

    def generate_for_instance(
        self, factual: pd.Series, *, neg_value: int = 0, **_kwargs: object
    ) -> pd.DataFrame:
        target_class = 1 - neg_value
        base_counterfactual = growing_spheres_search(
            factual.to_numpy(),
            pred_fn_crisp=self.base_predict,
            target_class=target_class,
            n_search_samples=self.gs_n_search_samples,
            p_norm=self.gs_p_norm,
            step=self.gs_step,
            max_iter=self.base_gs_max_iter,
            rng=self.rng,
            schema=self.schema,
        )
        if base_counterfactual is None:
            self.last_metadata = {
                "status": "base_counterfactual_not_found",
                "delta": self.delta,
                "alpha": self.alpha,
                "generation_ensemble_n": self.generation_ensemble_n,
            }
            return pd.DataFrame(columns=factual.index)

        result = self.generator.generate(
            base_counterfactual,
            target_class=target_class,
            delta=self.delta,
            alpha=self.alpha,
            gs_n_search_samples=self.gs_n_search_samples,
            gs_p_norm=self.gs_p_norm,
            gs_step=self.gs_step,
            gs_max_iter=self.gs_max_iter,
            rng=self.rng,
        )
        self.last_metadata = {
            **result.metadata,
            "delta": self.delta,
            "alpha": self.alpha,
            "generation_ensemble_n": self.generation_ensemble_n,
            "base_generator": "growing_spheres",
            "gs_n_search_samples": self.gs_n_search_samples,
            "gs_p_norm": self.gs_p_norm,
            "gs_step": self.gs_step,
            "base_gs_max_iter": self.base_gs_max_iter,
            "gs_max_iter": self.gs_max_iter,
        }
        if result.counterfactual is None:
            return pd.DataFrame(columns=factual.index)
        return pd.DataFrame([result.counterfactual], columns=factual.index)
