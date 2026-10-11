"""Logistic regression fit by scikit-learn, perturbed as a torch network."""

import dataclasses
import itertools
from typing import Any

from robustness_benchmark.core.classifiers import LogisticClassifier
from robustness_benchmark.core.model_specs.torch import TorchSpec

# (name, l1_ratio, class_weight, C values). Candidates are taken round-robin
# across groups, so any prefix mixes penalties and class weighting.
LOGISTIC_CANDIDATE_GROUPS: tuple[
    tuple[str, float, str | None, tuple[float, ...]], ...
] = (
    ("l2", 0.0, None, (1e-3, 3e-3, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100)),
    ("l1", 1.0, None, (0.01, 0.03, 0.1, 0.3, 1, 10)),
    ("elasticnet", 0.5, None, (0.01, 0.03, 0.1, 0.3, 1, 10)),
    ("l2_balanced", 0.0, "balanced", (1e-3, 0.01, 0.1, 1, 10, 100)),
    ("l1_balanced", 1.0, "balanced", (0.01, 0.1, 1, 10)),
    ("elasticnet_balanced", 0.5, "balanced", (0.01, 0.1, 1, 10)),
)
LOGISTIC_SELECTION_C = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)


@dataclasses.dataclass(frozen=True)
class LogisticSpec(TorchSpec):
    """Logistic regression.

    Seed, architecture and incremental-update families do not apply: the fit
    is a convex problem with one optimum, so a new seed or a warm start from
    the deployed coefficients reproduces retraining from scratch.
    """

    model_type = "logistic"
    change_families = (
        "bootstrap",
        "deletion",
        "data_addition",
        "label_update",
        "training_config",
        "bounded_parameter",
    )

    C: float = 1.0
    l1_ratio: float = 0.0
    class_weight: str | None = None
    max_iter: int = 5_000

    def build(self, input_dim: int, seed: int) -> LogisticClassifier:
        return LogisticClassifier(
            input_dim,
            seed=seed,
            C=self.C,
            l1_ratio=self.l1_ratio,
            class_weight=self.class_weight,
            max_iter=self.max_iter,
        )

    def id_fragment(self) -> str:
        return f"logistic_C_{self.C:g}"

    def training_config_variants(self) -> list[tuple[str, dict[str, Any]]]:
        groups = [
            [
                (
                    f"{name}_C_{C:g}",
                    {"C": float(C), "l1_ratio": l1_ratio, "class_weight": class_weight},
                )
                for C in values
            ]
            for name, l1_ratio, class_weight, values in LOGISTIC_CANDIDATE_GROUPS
        ]
        return [
            candidate
            for candidate in itertools.chain.from_iterable(
                itertools.zip_longest(*groups)
            )
            if candidate is not None and self.with_overrides(**candidate[1]) != self
        ]

    @classmethod
    def selection_grid(cls, base: "LogisticSpec") -> list["LogisticSpec"]:
        return [base.with_overrides(C=C) for C in LOGISTIC_SELECTION_C]
