"""Growing spheres (Laugel et al., 2018) as used by BetaRCE and RobX.

Benchmark extension, not CARLA's or Laugel's: Laugel et al. treat every
feature as continuous, and CARLA's ``growing_spheres_search`` draws each
binary column Bernoulli(0.5) in every candidate (gs_counterfactuals.py:114).
Here, with a constrained FeatureSchema the sphere covers the mutable numeric
columns only, immutable columns keep the instance's values, and each mutable
categorical feature is redrawn as a whole one-hot group
(``resample_categories``). A category change counts toward the candidate's
sampled radius, so its numeric step shrinks to make room and a candidate may
change categories alone. Candidates are projected (integer rounding) before
they are scored.
"""

from collections.abc import Callable

import numpy as np
import pandas as pd

from robustness_benchmark.core.schema import FeatureSchema
from robustness_benchmark.methods.feature_constraints import (
    fit_category_changes_in_radius,
    is_constrained,
    mutable_groups,
    resample_categories,
)


def hyper_sphere_coordinates(
    n_search_samples: int,
    instance: np.ndarray,
    high: float,
    low: float,
    p_norm: int = 2,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if rng is None:
        rng = np.random.default_rng()
    center = np.asarray(instance).reshape(1, -1)
    deltas = rng.standard_normal((n_search_samples, center.shape[1]))
    dist = rng.random(n_search_samples) * (high - low) + low
    norm_p = np.linalg.norm(deltas, ord=p_norm, axis=1)
    scale = dist / (norm_p + 1e-12)
    deltas = deltas * scale.reshape(-1, 1)
    return center + deltas, dist


def growing_spheres_search(
    instance: np.ndarray,
    pred_fn_crisp: Callable[[np.ndarray], np.ndarray],
    target_class: int | None = None,
    n_search_samples: int = 1000,
    p_norm: int = 2,
    step: float = 0.2,
    max_iter: int = 1000,
    binary_indices: list[int] | None = None,
    feature_min: np.ndarray | None = None,
    feature_max: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
    schema: FeatureSchema | None = None,
) -> np.ndarray | None:
    if rng is None:
        rng = np.random.default_rng()
    constrained = is_constrained(schema)

    instance = np.asarray(instance).reshape(1, -1)
    if not np.all(np.isfinite(instance)):
        return None
    instance_label = int(pred_fn_crisp(instance)[0])
    if target_class is None:
        if instance_label in (0, 1):
            target_class = 1 - instance_label
        else:
            raise ValueError("target_class must be provided for multiclass settings.")

    n_features = instance.shape[1]
    binary_indices = sorted(set(binary_indices)) if binary_indices else []
    off_sphere = np.zeros(n_features, dtype=bool)
    if binary_indices:
        off_sphere[binary_indices] = True
    groups = mutable_groups(schema) if constrained else []
    if constrained:
        off_sphere[schema.immutable_columns()] = True
        for group in schema.one_hot_groups().values():
            off_sphere[group] = True
    continuous_indices = np.where(~off_sphere)[0]

    low = 0.0
    high = low + step
    for _ in range(max_iter):
        candidates = np.repeat(instance, n_search_samples, axis=0)

        if continuous_indices.size > 0:
            cont_samples, radii = hyper_sphere_coordinates(
                n_search_samples,
                instance[:, continuous_indices],
                high=high,
                low=low,
                p_norm=p_norm,
                rng=rng,
            )
            candidates[:, continuous_indices] = cont_samples
        else:
            radii = rng.random(n_search_samples) * (high - low) + low

        if binary_indices:
            candidates[:, binary_indices] = rng.binomial(
                n=1,
                p=0.5,
                size=(n_search_samples, len(binary_indices)),
            )

        if groups:
            resample_categories(candidates, groups, rng)
            fit_category_changes_in_radius(
                candidates, instance, groups, continuous_indices, radii, p_norm
            )

        if feature_min is not None and feature_max is not None:
            candidates = np.clip(candidates, feature_min, feature_max)
        if constrained:
            candidates = schema.project(candidates, instance)

        preds = pred_fn_crisp(candidates)
        mask = preds == target_class
        if np.any(mask):
            if p_norm == 1:
                distances = np.abs(candidates - instance).sum(axis=1)
            elif p_norm == 2:
                distances = np.square(candidates - instance).sum(axis=1)
            else:
                raise ValueError("Unsupported p_norm. Use 1 or 2.")
            candidate_indices = np.where(mask)[0]
            best_idx = candidate_indices[np.argmin(distances[mask])]
            return candidates[best_idx]

        low = high
        high = low + step

    return None


class GrowingSpheresAdapter:
    """Non-robust growing spheres with BetaRCE's base-search settings.

    A candidate qualifies once the base model gives the target class at least
    ``target_probability``. At 0.5 this is plain growing spheres; higher values
    are confidence-margin baselines that trace the robustness-distance frontier
    a robust method has to beat.
    """

    def __init__(
        self,
        task,
        *,
        seed: int,
        target_probability: float = 0.5,
        n_search_samples: int = 100,
        p_norm: int = 2,
        step: float = 0.1,
        max_iter: int = 200,
    ) -> None:
        if not 0.5 <= target_probability < 1.0:
            raise ValueError("target_probability must lie in [0.5, 1)")
        self.model = task.model
        self.schema = task.schema
        self.target_probability = target_probability
        self.n_search_samples = n_search_samples
        self.p_norm = p_norm
        self.step = step
        self.max_iter = max_iter
        self.rng = np.random.default_rng(seed)
        self.last_metadata: dict[str, object] = {}

    def generate_for_instance(
        self, factual: pd.Series, *, neg_value: int = 0, **_kwargs: object
    ) -> pd.DataFrame:
        target = 1 - neg_value

        def qualifies(values: np.ndarray) -> np.ndarray:
            probability = self.model.predict(values).iloc[:, 0].to_numpy()
            if target == 0:
                probability = 1.0 - probability
            return (probability >= self.target_probability).astype(int)

        counterfactual = growing_spheres_search(
            factual.to_numpy(dtype=float),
            pred_fn_crisp=qualifies,
            target_class=1,
            n_search_samples=self.n_search_samples,
            p_norm=self.p_norm,
            step=self.step,
            max_iter=self.max_iter,
            rng=self.rng,
            schema=self.schema,
        )
        self.last_metadata = {
            "status": "ok" if counterfactual is not None else "not_found",
            "target_probability": self.target_probability,
        }
        if counterfactual is None:
            return pd.DataFrame(columns=factual.index)
        return pd.DataFrame([counterfactual], columns=factual.index)
