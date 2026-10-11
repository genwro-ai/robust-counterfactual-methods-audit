from typing import Any

from robustness_benchmark.core.classifiers.base import (
    DIFFERENTIABLE,
    LINEAR,
    TORCH_NETWORK,
    Classifier,
    missing_capabilities,
    predict_crisp,
    predict_proba,
)
from robustness_benchmark.core.classifiers.gradient_boosting import XGBoostClassifier
from robustness_benchmark.core.classifiers.logistic import LogisticClassifier
from robustness_benchmark.core.classifiers.mlp import MLPClassifier

MODEL_CLASSES: dict[str, type[Classifier]] = {
    model_class.model_type: model_class
    for model_class in (MLPClassifier, LogisticClassifier, XGBoostClassifier)
}
MODEL_TYPES = tuple(MODEL_CLASSES)


def classifier_from_payload(payload: dict[str, Any]) -> Classifier:
    """Restore a classifier from a checkpoint payload."""

    model_type = str(payload["model_type"])
    if model_type not in MODEL_CLASSES:
        raise ValueError(f"Unknown model type {model_type!r}")
    return MODEL_CLASSES[model_type].from_checkpoint_payload(payload)


__all__ = [
    "DIFFERENTIABLE",
    "LINEAR",
    "MODEL_CLASSES",
    "MODEL_TYPES",
    "TORCH_NETWORK",
    "Classifier",
    "LogisticClassifier",
    "MLPClassifier",
    "XGBoostClassifier",
    "classifier_from_payload",
    "missing_capabilities",
    "predict_crisp",
    "predict_proba",
]
