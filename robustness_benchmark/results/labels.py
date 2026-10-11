"""Display names and orders of methods, model classes, datasets and families."""

from robustness_benchmark.core.classifiers import MODEL_TYPES
from robustness_benchmark.core.datasets import DATASET_LABELS
from robustness_benchmark.core.model_specs import ALL_CHANGE_FAMILIES
from robustness_benchmark.methods.configuration import GROWING_SPHERES_TARGETS
from robustness_benchmark.results.figure_style import METHOD_STYLES, method_style

# Plain growing spheres, then the confidence-margin baselines.
MARGIN_METHODS = tuple(GROWING_SPHERES_TARGETS)
MARGIN_BASELINES = MARGIN_METHODS[1:]
# Tables, heatmaps and legends share the figure style's method order.
METHOD_ORDER = tuple(METHOD_STYLES)
MODEL_TYPE_LABELS = {
    "mlp": "MLP",
    "logistic": "Logistic regression",
    "xgboost": "XGBoost",
}
assert tuple(MODEL_TYPE_LABELS) == MODEL_TYPES

FAMILY_LABELS = {
    "architecture": "Architecture",
    "bootstrap": "Bootstrap",
    "seed": "New seed",
    "training_config": "Training config",
    "label_update": "Label update",
    "deletion": "Deletion",
    "data_addition": "Data addition",
    "bounded_parameter": "Parameter perturb.",
    "incremental_update": "Incremental update",
}


CROSS_CLASS = "cross_class"


SCOPE_LABELS = {**MODEL_TYPE_LABELS, CROSS_CLASS: "Cross-class"}


def ordered(values, order: tuple[str, ...]) -> list[str]:
    """Return ``values`` in ``order``, followed by any unknown values sorted."""

    present = set(values)
    return [value for value in order if value in present] + sorted(
        present.difference(order)
    )


def method_label(method: str) -> str:
    return method_style(method).label


FAMILY_ORDER = tuple(FAMILY_LABELS)
assert set(FAMILY_ORDER) == set(ALL_CHANGE_FAMILIES)

__all__ = [
    "CROSS_CLASS",
    "DATASET_LABELS",
    "FAMILY_LABELS",
    "FAMILY_ORDER",
    "MARGIN_BASELINES",
    "MARGIN_METHODS",
    "METHOD_ORDER",
    "MODEL_TYPE_LABELS",
    "SCOPE_LABELS",
    "method_label",
    "ordered",
]
