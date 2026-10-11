"""RobX's counterfactual stability and its training-quantile threshold.

Stability of a point is the mean minus the standard deviation of the model's
probability for the point's own class under Gaussian input noise (Dutta et
al., ICML 2022). RobX and T:Rex share it, and both calibrate their threshold
tau as a quantile of the stability of the base model's target-class training
points.
"""

from collections.abc import Callable

import numpy as np

from robustness_benchmark.core.classifiers import Classifier


def counterfactual_stability_batch(
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
        counterfactual_stability_batch(x, pred_func, variance, N, gamma, rng)[0]
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
            counterfactual_stability_batch(
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


def target_stability_scores(
    model: Classifier,
    X_train: np.ndarray,
    *,
    variance: float,
    samples: int,
    seed: int,
) -> tuple[np.ndarray, int]:
    """Stability scores of the training points ``model`` assigns class 1."""

    return training_target_stability_scores(
        X_train,
        lambda values: model.predict_proba(values).iloc[:, 1].to_numpy(),
        variance=variance,
        N=samples,
        seed=seed,
    )
