from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from robustness_benchmark.methods.growing_spheres import growing_spheres_search


def _counterfactual_stability_batch(
    xs: np.ndarray,
    pred_func: Callable[[np.ndarray], np.ndarray],
    variance: np.ndarray | float,
    N: int,
    gamma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    B, d = xs.shape
    results = np.full(B, np.nan)
    valid_mask = np.all(np.isfinite(xs), axis=1)
    if not np.any(valid_mask):
        return results

    valid_xs = xs[valid_mask]
    B_valid = valid_xs.shape[0]

    if isinstance(variance, np.ndarray):
        variance = np.asarray(variance)
        if variance.ndim == 1:
            std = np.sqrt(np.clip(variance, 1e-12, None))
            noise = rng.standard_normal((B_valid, N, d)) * std[None, None, :]
        elif variance.ndim == 2:
            try:
                noise = rng.multivariate_normal(
                    np.zeros(d), variance, size=(B_valid * N,)
                )
                noise = noise.reshape(B_valid, N, d)
            except (ValueError, np.linalg.LinAlgError, FloatingPointError):
                return results
        else:
            raise ValueError("variance must be 1D or 2D")
    else:
        std = np.sqrt(max(float(variance), 1e-12))
        noise = rng.standard_normal((B_valid, N, d)) * std

    X_p = valid_xs[:, None, :] + noise
    finite_per_point = np.all(np.isfinite(X_p.reshape(B_valid, -1)), axis=1)

    orig_preds = pred_func(valid_xs)
    cf_classes = (orig_preds > gamma).astype(int)

    all_preds = pred_func(X_p.reshape(B_valid * N, d)).reshape(B_valid, N)
    X_pred = np.where(cf_classes[:, None] == 1, all_preds, 1.0 - all_preds)
    stabilities = np.mean(X_pred, axis=1) - np.std(X_pred, axis=1)
    stabilities[~finite_per_point] = np.nan
    results[valid_mask] = stabilities
    return results


def counterfactual_stability(
    x: np.ndarray,
    pred_func: Callable[[np.ndarray], np.ndarray],
    variance: np.ndarray | float = 0.1,
    N: int = 100,
    gamma: float = 0.5,
    rng: np.random.Generator | None = None,
) -> float:
    if rng is None:
        rng = np.random.default_rng()
    x = np.asarray(x).reshape(1, -1)
    return float(
        _counterfactual_stability_batch(x, pred_func, variance, N, gamma, rng)[0]
    )


def training_target_stability_scores(
    X_train: np.ndarray,
    predict_target_proba_fn: Callable[[np.ndarray], np.ndarray],
    *,
    variance: np.ndarray | float = 0.01,
    N: int = 1_000,
    gamma: float = 0.5,
    seed: int = 0,
    batch_size: int = 128,
) -> tuple[np.ndarray, int]:
    """Measure local stability for base-predicted target training points."""

    if N < 1:
        raise ValueError("N must be positive")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    values = np.asarray(X_train)
    target_probabilities = np.asarray(predict_target_proba_fn(values)).reshape(-1)
    target_points = values[target_probabilities > gamma]
    if target_points.size == 0:
        raise ValueError("RobX tau calibration found no target-class training points")

    rng = np.random.default_rng(seed)
    score_batches = []
    for start in range(0, len(target_points), batch_size):
        score_batches.append(
            _counterfactual_stability_batch(
                target_points[start : start + batch_size],
                predict_target_proba_fn,
                variance,
                N,
                gamma,
                rng,
            )
        )
    scores = np.concatenate(score_batches)
    finite_scores = scores[np.isfinite(scores)]
    if finite_scores.size == 0:
        raise ValueError("RobX tau calibration produced no finite stability scores")
    return finite_scores, len(target_points)


def tau_from_stability_quantile(
    scores: np.ndarray,
    *,
    quantile: float,
    gamma: float = 0.5,
    target_training_points: int | None = None,
) -> tuple[float, dict[str, int | float]]:
    """Select a RobX threshold from a training-stability quantile."""

    if not 0 < quantile < 1:
        raise ValueError("quantile must lie strictly between zero and one")
    finite_scores = np.asarray(scores)[np.isfinite(scores)]
    if finite_scores.size == 0:
        raise ValueError("RobX tau calibration received no finite stability scores")

    raw_tau = float(np.quantile(finite_scores, quantile))
    tau = max(gamma, raw_tau)
    return tau, {
        "quantile": quantile,
        "target_training_points": int(target_training_points or len(finite_scores)),
        "finite_stability_scores": len(finite_scores),
        "qualifying_training_points": int(np.sum(finite_scores > tau)),
        "raw_tau": raw_tau,
        "tau": tau,
    }


def calibrate_tau_from_training_quantile(
    X_train: np.ndarray,
    predict_target_proba_fn: Callable[[np.ndarray], np.ndarray],
    *,
    quantile: float,
    variance: np.ndarray | float = 0.01,
    N: int = 1_000,
    gamma: float = 0.5,
    seed: int = 0,
    batch_size: int = 128,
) -> tuple[float, dict[str, int | float]]:
    """Calibrate RobX's threshold from favorable training-point stability.

    Dutta et al. recommend choosing a dataset-dependent quantile of the
    stability-score histogram. Calibration is performed only on points that
    the base model assigns to the target class. No changed model is consulted.
    """

    scores, target_training_points = training_target_stability_scores(
        X_train,
        predict_target_proba_fn,
        variance=variance,
        N=N,
        gamma=gamma,
        seed=seed,
        batch_size=batch_size,
    )
    return tau_from_stability_quantile(
        scores,
        quantile=quantile,
        gamma=gamma,
        target_training_points=target_training_points,
    )


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
        stabilities = _counterfactual_stability_batch(
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
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if rng is None:
        rng = np.random.default_rng()

    init_stability = _counterfactual_stability_batch(
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

    counterfactuals = np.tile(start_counterfactual, (len(conservative_cfs), 1))
    for _ in range(robx_max_iter):
        counterfactuals = (
            robx_lambda * conservative_cfs + (1.0 - robx_lambda) * counterfactuals
        )
        stabilities = _counterfactual_stability_batch(
            counterfactuals,
            predict_class_proba_fn,
            variance,
            N,
            gamma,
            rng,
        )
        stable_mask = stabilities > tau
        if np.any(stable_mask):
            return counterfactuals[np.argmax(stable_mask)], conservative_cfs

    return None, None


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
    ) -> None:
        self.X_train = np.asarray(X_train)
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
        )

        status = "ok" if robust_cf is not None else "no_robust_cf"
        return RobXResult(robust_cf, start_cf, conservative_cfs, {"status": status})
