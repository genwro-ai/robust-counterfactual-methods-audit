"""RobX (Dutta et al., ICML 2022) on top of growing spheres.

Faithful to the paper and the BetaRCE implementation: interpolation and the
Gaussian stability test treat every encoded column, one-hot and immutable ones
included, as continuous. Only the start counterfactual comes from the
benchmark's shared growing spheres, which respects the feature schema; the
registry projects the returned point before checking validity, like every
method's.

Each of the c chains moves towards its conservative counterfactual until it
passes the stability test, and the passing point closest to the factual is
returned, as in the paper's Algorithm 1; BetaRCE's ``robx_algorithm`` returns
the first point to pass. One fix to the reference code: stability is measured
for a point's own class, so an iterate is accepted only if the model also
assigns it the counterfactual's class.

Its settings (``LOCKED_CONFIGS``) are k = 3 conservative counterfactuals and
100 iterations (Algorithm 1; BetaRCE's ``robx_algorithm`` defaults), and
growing spheres with 100 samples, p = 2, step 0.1 and 100 iterations (``gs``
in BetaRCE's configs/config_exp.yaml), extended to 1,000 iterations where
that limit left RobX without a start (``EXTENDED_GS_SEARCH``).
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from robustness_benchmark.core.classifiers import predict_crisp, predict_proba
from robustness_benchmark.core.schema import FeatureSchema
from robustness_benchmark.methods.configuration import ROBX_IMPLEMENTATION
from robustness_benchmark.methods.growing_spheres import growing_spheres_search
from robustness_benchmark.methods.stability import counterfactual_stability_batch


def get_conservative_counterfactuals(
    counterfactual: np.ndarray,
    data_X: np.ndarray,
    predict_class_proba_fn: Callable[[np.ndarray], np.ndarray],
    variance: np.ndarray | float = 0.1,
    tau: float = 0.5,
    N: int = 100,
    k: int = 3,
    gamma: float = 0.5,
    rng: np.random.Generator | None = None,
    _batch_size: int = 128,
) -> np.ndarray | None:
    cf_prob = predict_class_proba_fn(counterfactual.reshape(1, -1))[0]
    cf_class = 1 if cf_prob > gamma else 0

    data_probs = predict_class_proba_fn(data_X)
    data_classes = (data_probs > gamma).astype(int)
    data = data_X[data_classes == cf_class]
    if data.size == 0:
        return None

    dist = np.sum(np.abs(data - counterfactual), axis=1)
    data = data[np.argsort(dist)]

    conservative = []
    for start in range(0, len(data), _batch_size):
        batch = data[start : start + _batch_size]
        stabilities = counterfactual_stability_batch(
            batch, predict_class_proba_fn, variance, N, gamma, rng
        )
        for x, s in zip(batch, stabilities):
            if s > tau:
                conservative.append(x)
                if len(conservative) == k:
                    return np.array(conservative)

    return np.array(conservative) if conservative else None


def robx_algorithm(
    X_train: np.ndarray,
    predict_class_proba_fn: Callable[[np.ndarray], np.ndarray],
    start_counterfactual: np.ndarray,
    variance: np.ndarray | float = 0.1,
    tau: float = 0.5,
    N: int = 100,
    k: int = 3,
    robx_max_iter: int = 100,
    robx_lambda: float = 0.1,
    gamma: float = 0.5,
    rng: np.random.Generator | None = None,
    *,
    factual: np.ndarray,
    p_norm: int = 2,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Algorithm 1: the stable chain point closest to ``factual`` in ``p_norm``.

    A chain that does not pass within ``robx_max_iter`` steps is dropped.
    """
    if rng is None:
        rng = np.random.default_rng()

    init_stability = counterfactual_stability_batch(
        start_counterfactual.reshape(1, -1),
        predict_class_proba_fn,
        variance,
        N,
        gamma,
        rng,
    )[0]
    if init_stability > tau:
        return start_counterfactual, None

    conservative_cfs = get_conservative_counterfactuals(
        start_counterfactual,
        X_train,
        predict_class_proba_fn,
        variance=variance,
        tau=tau,
        N=N,
        k=k,
        gamma=gamma,
        rng=rng,
    )
    if conservative_cfs is None:
        return None, None

    target_side = predict_class_proba_fn(start_counterfactual.reshape(1, -1))[0] > gamma
    counterfactuals = np.tile(start_counterfactual, (len(conservative_cfs), 1))
    stable = np.zeros(len(conservative_cfs), dtype=bool)
    for _ in range(robx_max_iter):
        moving = ~stable
        counterfactuals[moving] = (
            robx_lambda * conservative_cfs[moving]
            + (1.0 - robx_lambda) * counterfactuals[moving]
        )
        stabilities = counterfactual_stability_batch(
            counterfactuals[moving],
            predict_class_proba_fn,
            variance,
            N,
            gamma,
            rng,
        )
        stable[moving] = (stabilities > tau) & (
            (predict_class_proba_fn(counterfactuals[moving]) > gamma) == target_side
        )
        if stable.all():
            break
    if not stable.any():
        return None, None
    candidates = counterfactuals[stable]
    distances = np.linalg.norm(
        candidates - np.asarray(factual).reshape(1, -1), ord=p_norm, axis=1
    )
    return candidates[np.argmin(distances)], conservative_cfs


@dataclass
class RobXResult:
    counterfactual: np.ndarray | None
    start_counterfactual: np.ndarray | None
    conservative_counterfactuals: np.ndarray | None
    metadata: dict[str, Any]


class RobX:
    def __init__(
        self,
        X_train: np.ndarray,
        predict_fn_crisp: Callable[[np.ndarray], np.ndarray],
        predict_proba_fn: Callable[[np.ndarray], np.ndarray],
        binary_indices: list[int] | None = None,
        feature_min: np.ndarray | None = None,
        feature_max: np.ndarray | None = None,
        schema: FeatureSchema | None = None,
    ) -> None:
        self.X_train = np.asarray(X_train)
        self.schema = schema
        self.predict_fn_crisp = predict_fn_crisp
        self.predict_proba_fn = predict_proba_fn
        self.binary_indices = binary_indices
        self.feature_min = feature_min
        self.feature_max = feature_max

    def _predict_target_proba(self, X: np.ndarray, target_class: int) -> np.ndarray:
        proba = np.asarray(self.predict_proba_fn(X))
        if proba.ndim == 1:
            return proba
        if proba.shape[1] == 1:
            return proba.reshape(-1)
        return proba[:, target_class]

    def generate(
        self,
        start_instance: np.ndarray,
        target_class: int,
        variance=0.1,
        tau=0.5,
        N=100,
        k=3,
        robx_max_iter=100,
        robx_lambda=0.1,
        gamma=0.5,
        gs_n_search_samples=1000,
        gs_p_norm=2,
        gs_step=0.2,
        gs_max_iter=1000,
        rng=None,
    ) -> RobXResult:
        if rng is None:
            rng = np.random.default_rng()

        start_instance = np.asarray(start_instance).reshape(-1)
        start_cf = growing_spheres_search(
            instance=start_instance,
            pred_fn_crisp=self.predict_fn_crisp,
            target_class=target_class,
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

        if start_cf is None:
            return RobXResult(None, None, None, {"status": "no_start_cf"})

        def predict_target_proba(X):
            return self._predict_target_proba(X, target_class)

        robust_cf, conservative_cfs = robx_algorithm(
            self.X_train,
            predict_target_proba,
            start_cf,
            variance=variance,
            tau=tau,
            N=N,
            k=k,
            robx_max_iter=robx_max_iter,
            robx_lambda=robx_lambda,
            gamma=gamma,
            rng=rng,
            factual=start_instance,
            p_norm=gs_p_norm,
        )

        status = "ok" if robust_cf is not None else "no_robust_cf"
        return RobXResult(robust_cf, start_cf, conservative_cfs, {"status": status})


class RobXAdapter:
    def __init__(
        self,
        task,
        *,
        seed: int,
        variance: float,
        tau: float,
        stability_samples: int,
        k: int,
        max_iter: int,
        robx_lambda: float,
        gs_n_search_samples: int,
        gs_p_norm: int,
        gs_step: float,
        gs_max_iter: int,
    ) -> None:
        self.task = task
        self.seed = seed
        self.variance = variance
        self.tau = tau
        self.stability_samples = stability_samples
        self.search = {
            "k": k,
            "robx_max_iter": max_iter,
            "robx_lambda": robx_lambda,
            "gs_n_search_samples": gs_n_search_samples,
            "gs_p_norm": gs_p_norm,
            "gs_step": gs_step,
            "gs_max_iter": gs_max_iter,
        }
        X_train = task.training_data.X.to_numpy()
        self.generator = RobX(
            X_train,
            lambda values: predict_crisp(task.model, values),
            lambda values: predict_proba(task.model, values),
            feature_min=X_train.min(axis=0),
            feature_max=X_train.max(axis=0),
            schema=task.schema,
        )
        self.last_metadata: dict[str, object] = {}

    def generate_for_instance(
        self, factual: pd.Series, *, neg_value: int = 0, **_kwargs: object
    ) -> pd.DataFrame:
        result = self.generator.generate(
            factual.to_numpy(),
            target_class=1 - neg_value,
            variance=self.variance,
            tau=self.tau,
            N=self.stability_samples,
            **self.search,
            rng=np.random.default_rng(self.seed),
        )
        self.last_metadata = {
            f"robx_{key}": value
            for key, value in {
                **result.metadata,
                "implementation": ROBX_IMPLEMENTATION,
                "variance": self.variance,
                "tau": self.tau,
                "stability_samples": self.stability_samples,
                "k": self.search["k"],
            }.items()
        }
        if result.counterfactual is None:
            return pd.DataFrame(columns=factual.index)
        return pd.DataFrame([result.counterfactual], columns=factual.index)
