"""A changed model, and the bounded parameter change shared by all classes."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from robustness_benchmark.core.classifiers import Classifier, MLPClassifier
from robustness_benchmark.core.model_specs import ModelSpec, perturb_torch_parameters

FULL_VARIANTS_PER_FAMILY = 25
# Bounded-parameter radii as fractions of the base model's mean |parameter|,
# so the l_inf box means the same relative change for weights and leaves.
RELATIVE_PARAMETER_RADII = (0.05, 0.1, 0.2, 0.3, 0.5)
# A configuration candidate whose validation probabilities differ from the
# base model's by less than this mean absolute amount reproduces the base.
DISTINCT_VALIDATION_MAE = 1e-3


@dataclass(frozen=True)
class ChangedModel:
    change_id: str
    change_family: str
    change_level: str
    model: Classifier
    metadata: dict[str, object]


def parameter_linf_distance(base_model, changed_model) -> float | None:
    """Return aligned parameter L-infinity distance, or None across topologies."""

    base_state = base_model.get_torch_model().state_dict()
    changed_state = changed_model.get_torch_model().state_dict()
    if tuple(base_state) != tuple(changed_state):
        return None
    maximum = 0.0
    for name, base_value in base_state.items():
        changed_value = changed_state[name]
        if base_value.shape != changed_value.shape:
            return None
        maximum = max(
            maximum,
            float(torch.max(torch.abs(base_value - changed_value)).item()),
        )
    return maximum


def make_bounded_parameter_change(
    base_model: Classifier,
    *,
    radius: float,
    perturbation_seed: int,
    replicate: int = 0,
    spec: ModelSpec | None = None,
    relative_radius: float | None = None,
) -> ChangedModel:
    """Perturb aligned parameters within an exact L-infinity radius.

    A single random direction is scaled so that its largest absolute component
    reaches ``radius``. Consequently every parameter change is bounded by the
    declared radius, while the realized L-infinity distance is non-vacuous.
    Torch networks perturb weights and biases; XGBoost perturbs leaf values.
    ``relative_radius`` records ``radius`` as a fraction of the mean |parameter|
    and names the level after it.
    """

    if radius <= 0:
        raise ValueError("radius must be positive")
    if spec is None:
        if not isinstance(base_model, MLPClassifier):
            raise TypeError("a spec is required to perturb a non-torch model")
        model = perturb_torch_parameters(
            base_model, radius=radius, seed=perturbation_seed
        )
    else:
        model = spec.perturb_parameters(
            base_model, radius=radius, seed=perturbation_seed
        )
    level = (
        f"relative_linf_{relative_radius:g}"
        if relative_radius is not None
        else f"linf_{radius:g}"
    )
    return ChangedModel(
        f"bounded_parameter:{level}:r{replicate}",
        "bounded_parameter",
        level,
        model,
        {
            "parameter_norm": "linf",
            "parameter_radius": radius,
            "relative_parameter_radius": relative_radius,
            "perturbed_parameters": (
                "leaf_values" if base_model.model_type == "xgboost" else "weights"
            ),
            "replicate": replicate,
            "perturbation_seed": perturbation_seed,
            "topology_aligned": True,
        },
    )


def behavioral_distance(
    base_model, changed_model, X_reference: pd.DataFrame
) -> dict[str, float]:
    """Prediction disagreement and mean probability gap on reference data."""

    base_probability = base_model.predict(X_reference).iloc[:, 0].to_numpy()
    changed_probability = changed_model.predict(X_reference).iloc[:, 0].to_numpy()
    return {
        "hard_disagreement": float(
            np.mean((base_probability >= 0.5) != (changed_probability >= 0.5))
        ),
        "probability_mae": float(
            np.mean(np.abs(base_probability - changed_probability))
        ),
    }
