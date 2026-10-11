"""How each model class is trained, tuned and changed by the model bank."""

from typing import Any

from robustness_benchmark.core.model_specs.base import ALL_CHANGE_FAMILIES, ModelSpec
from robustness_benchmark.core.model_specs.gradient_boosting import XGBoostSpec
from robustness_benchmark.core.model_specs.logistic import LogisticSpec
from robustness_benchmark.core.model_specs.mlp import MLPSpec
from robustness_benchmark.core.model_specs.torch import (
    perturb_torch_parameters,
    torch_incremental_update,
)

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
    """Rebuild the spec a bank's models were trained with."""

    return spec_from_dict(manifest["model_spec"])


__all__ = [
    "ALL_CHANGE_FAMILIES",
    "MODEL_SPECS",
    "LogisticSpec",
    "MLPSpec",
    "ModelSpec",
    "XGBoostSpec",
    "perturb_torch_parameters",
    "spec_from_dict",
    "spec_from_manifest",
    "torch_incremental_update",
]
