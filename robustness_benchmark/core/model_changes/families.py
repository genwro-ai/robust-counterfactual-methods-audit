"""The evaluation change families, built around one base model."""

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from robustness_benchmark.core.classifiers import Classifier
from robustness_benchmark.core.data import DatasetSplit
from robustness_benchmark.core.fitting import fit_spec
from robustness_benchmark.core.model_changes.base import (
    DISTINCT_VALIDATION_MAE,
    FULL_VARIANTS_PER_FAMILY,
    RELATIVE_PARAMETER_RADII,
    ChangedModel,
    make_bounded_parameter_change,
)
from robustness_benchmark.core.model_specs import ModelSpec

SMOKE_CHANGE_FAMILIES = (
    "seed",
    "bootstrap",
    "deletion",
    "architecture",
    "bounded_parameter",
    "incremental_update",
)
# Offsets added to the base seed, one per family's random draws.
SEED_FAMILY_OFFSET = 1
BOOTSTRAP_SEED_OFFSET = 10_000
DELETION_SEED_OFFSET = 20_000
UPDATE_POOL_SEED_OFFSET = 30_000
LABEL_UPDATE_SEED_OFFSET = 40_000
PERTURBATION_SEED_OFFSET = 80_000


def _replicates_by_level(total: int, levels: int) -> tuple[int, ...]:
    """Distribute a fixed candidate budget as evenly as possible over levels."""

    quotient, remainder = divmod(total, levels)
    return tuple(quotient + (index < remainder) for index in range(levels))


@dataclass
class _Bank:
    """The base model, its training data and the changes built so far."""

    data: DatasetSplit
    base_model: Classifier
    base_seed: int
    spec: ModelSpec
    smoke: bool
    changed: list[ChangedModel] = field(default_factory=list)

    @property
    def replicates(self) -> int:
        return 1 if self.smoke else FULL_VARIANTS_PER_FAMILY

    @property
    def n(self) -> int:
        return len(self.data.X_train)

    def fit_variant(
        self,
        change_id: str,
        family: str,
        level: str,
        X_train: pd.DataFrame | None = None,
        y_train: pd.Series | None = None,
        *,
        seed: int | None = None,
        overrides: dict[str, object] | None = None,
    ) -> ChangedModel:
        X_train = self.data.X_train if X_train is None else X_train
        y_train = self.data.y_train if y_train is None else y_train
        seed = self.base_seed if seed is None else seed
        variant_spec = self.spec.with_overrides(**(overrides or {}))
        model = fit_spec(
            variant_spec,
            X_train.reset_index(drop=True),
            y_train.reset_index(drop=True),
            self.data.X_val,
            self.data.y_val,
            seed=seed,
        )
        return ChangedModel(
            change_id,
            family,
            level,
            model,
            {
                "model_type": self.spec.model_type,
                "training_seed": seed,
                **variant_spec.hyperparameters(),
            },
        )

    def train_variant(self, *args, **kwargs) -> ChangedModel:
        change = self.fit_variant(*args, **kwargs)
        self.changed.append(change)
        return change

    def update_pool_subsets(
        self,
    ) -> Iterator[tuple[str, float, int, int, np.ndarray]]:
        """Yield (level, fraction, replicate, seed, rows) of update-pool subsets.

        A fixed update pool has only one genuinely distinct 100% subset, so
        there are eight independent subsets at each lower level and the full
        pool once. Data addition and incremental update share these subsets.
        """

        design = (
            ((1.0, 1),) if self.smoke else ((0.25, 8), (0.50, 8), (0.75, 8), (1.0, 1))
        )
        pool_n = len(self.data.X_update)
        for fraction, level_replicates in design:
            for replicate in range(level_replicates):
                seed = (
                    self.base_seed
                    + UPDATE_POOL_SEED_OFFSET
                    + round(fraction * 1_000)
                    + replicate
                )
                rows = np.sort(
                    np.random.default_rng(seed).choice(
                        pool_n, size=max(1, round(fraction * pool_n)), replace=False
                    )
                )
                yield f"{fraction:g}_update_pool", fraction, replicate, seed, rows

    def with_update_rows(self, rows: np.ndarray) -> tuple[pd.DataFrame, pd.Series]:
        return (
            pd.concat(
                [self.data.X_train, self.data.X_update.iloc[rows]], ignore_index=True
            ),
            pd.concat(
                [self.data.y_train, self.data.y_update.iloc[rows]], ignore_index=True
            ),
        )


def _configuration_family(
    bank: _Bank, family: str, candidates: list[tuple[str, dict[str, object]]]
) -> None:
    """Keep, in order, the first candidates that change the base model."""

    base_validation = bank.base_model.predict(bank.data.X_val).iloc[:, 0].to_numpy()
    accepted: list[ChangedModel] = []
    skipped: list[str] = []
    for level, overrides in candidates:
        if len(accepted) == bank.replicates:
            break
        change = bank.fit_variant(
            f"{family}:{level}:seed{bank.base_seed}", family, level, overrides=overrides
        )
        validation = change.model.predict(bank.data.X_val).iloc[:, 0].to_numpy()
        mae = float(np.mean(np.abs(validation - base_validation)))
        change.metadata["validation_probability_mae"] = mae
        if mae < DISTINCT_VALIDATION_MAE:
            skipped.append(level)
        else:
            accepted.append(change)
    if len(accepted) < bank.replicates:
        raise AssertionError(
            f"Only {len(accepted)} {family} candidates change the base model"
        )
    for change in accepted:
        change.metadata["skipped_candidates"] = skipped
    bank.changed.extend(accepted)


def _seed(bank: _Bank) -> None:
    for replicate in range(bank.replicates):
        seed = bank.base_seed + SEED_FAMILY_OFFSET + replicate
        bank.train_variant(
            f"seed:new_initialization:r{replicate}:seed{seed}",
            "seed",
            "new_initialization",
            seed=seed,
        )


def _bootstrap(bank: _Bank) -> None:
    n = bank.n
    for replicate in range(bank.replicates):
        sampling_seed = bank.base_seed + BOOTSTRAP_SEED_OFFSET + replicate
        index = np.random.default_rng(sampling_seed).integers(0, n, size=n)
        change = bank.train_variant(
            f"bootstrap:100_percent:r{replicate}:seed{sampling_seed}",
            "bootstrap",
            "100_percent",
            bank.data.X_train.iloc[index],
            bank.data.y_train.iloc[index],
        )
        change.metadata.update(
            {"sampling_seed": sampling_seed, "replicate": replicate, "n": n}
        )


def _deletion(bank: _Bank) -> None:
    n = bank.n
    fractions = (0.10,) if bank.smoke else (0.01, 0.05, 0.10)
    replicates = (
        (1,)
        if bank.smoke
        else _replicates_by_level(FULL_VARIANTS_PER_FAMILY, len(fractions))
    )
    for fraction, level_replicates in zip(fractions, replicates, strict=True):
        for replicate in range(level_replicates):
            sampling_seed = (
                bank.base_seed
                + DELETION_SEED_OFFSET
                + round(fraction * 1_000)
                + replicate
            )
            keep = np.sort(
                np.random.default_rng(sampling_seed).choice(
                    n, size=round((1.0 - fraction) * n), replace=False
                )
            )
            level = f"{fraction:g}"
            change = bank.train_variant(
                f"deletion:{level}:r{replicate}:seed{sampling_seed}",
                "deletion",
                level,
                bank.data.X_train.iloc[keep],
                bank.data.y_train.iloc[keep],
            )
            change.metadata.update(
                {
                    "deleted_fraction": fraction,
                    "sampling_seed": sampling_seed,
                    "replicate": replicate,
                }
            )


def _data_addition(bank: _Bank) -> None:
    for level, fraction, replicate, seed, rows in bank.update_pool_subsets():
        change = bank.train_variant(
            f"data_addition:{level}:r{replicate}:seed{seed}",
            "data_addition",
            level,
            *bank.with_update_rows(rows),
        )
        change.metadata.update(
            {
                "added_n": len(rows),
                "update_fraction": fraction,
                "sampling_seed": seed,
                "replicate": replicate,
            }
        )


def _label_update(bank: _Bank) -> None:
    n = bank.n
    fractions = (0.01, 0.05, 0.10)
    for fraction, level_replicates in zip(
        fractions,
        _replicates_by_level(FULL_VARIANTS_PER_FAMILY, len(fractions)),
        strict=True,
    ):
        for replicate in range(level_replicates):
            sampling_seed = (
                bank.base_seed
                + LABEL_UPDATE_SEED_OFFSET
                + round(fraction * 1_000)
                + replicate
            )
            changed_n = max(1, round(fraction * n))
            positions = np.random.default_rng(sampling_seed).choice(
                n, size=changed_n, replace=False
            )
            updated_labels = bank.data.y_train.reset_index(drop=True).copy()
            updated_labels.iloc[positions] = 1 - updated_labels.iloc[positions]
            level = f"{fraction:g}"
            change = bank.train_variant(
                f"label_update:{level}:r{replicate}:seed{sampling_seed}",
                "label_update",
                level,
                bank.data.X_train,
                updated_labels,
            )
            change.metadata.update(
                {
                    "updated_fraction": fraction,
                    "updated_n": changed_n,
                    "sampling_seed": sampling_seed,
                    "replicate": replicate,
                }
            )


def _training_config(bank: _Bank) -> None:
    _configuration_family(bank, "training_config", bank.spec.training_config_variants())


def _architecture(bank: _Bank) -> None:
    _configuration_family(bank, "architecture", bank.spec.architecture_variants())


def _bounded_parameter(bank: _Bank) -> None:
    scale = bank.spec.parameter_scale(bank.base_model)
    for relative_radius in (0.1,) if bank.smoke else RELATIVE_PARAMETER_RADII:
        for replicate in range(1 if bank.smoke else 5):
            perturbation_seed = (
                bank.base_seed
                + PERTURBATION_SEED_OFFSET
                + round(relative_radius * 1_000)
                + replicate
            )
            change = make_bounded_parameter_change(
                bank.base_model,
                radius=relative_radius * scale,
                perturbation_seed=perturbation_seed,
                replicate=replicate,
                spec=bank.spec,
                relative_radius=relative_radius,
            )
            change.metadata.update(
                {"model_type": bank.spec.model_type, **bank.spec.hyperparameters()}
            )
            bank.changed.append(change)


def _incremental_update(bank: _Bank) -> None:
    # Twin of data addition: the same data, but the deployed model is updated
    # instead of retrained from scratch.
    base_trees = getattr(bank.base_model, "num_trees", None)
    for level, fraction, replicate, seed, rows in bank.update_pool_subsets():
        model = bank.spec.incremental_update(
            bank.base_model,
            *bank.with_update_rows(rows),
            bank.data.X_val,
            bank.data.y_val,
        )
        trees = getattr(model, "num_trees", None)
        bank.changed.append(
            ChangedModel(
                f"incremental_update:{level}:r{replicate}:seed{seed}",
                "incremental_update",
                level,
                model,
                {
                    "model_type": bank.spec.model_type,
                    **bank.spec.hyperparameters(),
                    "initialized_from_base_checkpoint": True,
                    "paired_change_id": f"data_addition:{level}:r{replicate}:seed{seed}",
                    "added_n": len(rows),
                    "update_fraction": fraction,
                    "sampling_seed": seed,
                    "replicate": replicate,
                    "added_trees": None if trees is None else trees - base_trees,
                },
            )
        )


FAMILY_BUILDERS: dict[str, Callable[[_Bank], None]] = {
    "seed": _seed,
    "bootstrap": _bootstrap,
    "deletion": _deletion,
    "data_addition": _data_addition,
    "label_update": _label_update,
    "training_config": _training_config,
    "architecture": _architecture,
    "bounded_parameter": _bounded_parameter,
    "incremental_update": _incremental_update,
}


def make_changed_models(
    data: DatasetSplit,
    base_model: Classifier,
    *,
    base_seed: int,
    spec: ModelSpec,
    profile: str = "full",
) -> list[ChangedModel]:
    """Construct a deterministic model-change bank.

    ``spec`` describes how ``base_model`` was trained. The full profile
    contains 25 candidates in each change family that applies to the model
    class. The smoke profile keeps one cheap representative per applicable
    family in ``SMOKE_CHANGE_FAMILIES`` for fast contract tests.
    """

    if profile not in {"full", "smoke"}:
        raise ValueError(f"Unknown model-bank profile {profile!r}")
    families = tuple(
        family
        for family in spec.change_families
        if profile == "full" or family in SMOKE_CHANGE_FAMILIES
    )
    smoke = profile == "smoke"
    bank = _Bank(data, base_model, base_seed, spec, smoke)
    for family in families:
        FAMILY_BUILDERS[family](bank)
    changed = bank.changed

    expected = len(families) * (1 if smoke else FULL_VARIANTS_PER_FAMILY)
    if len(changed) != expected:
        raise AssertionError(f"Expected {expected} changed models, got {len(changed)}")
    if not smoke:
        family_counts = {
            family: sum(change.change_family == family for change in changed)
            for family in families
        }
        if set(family_counts.values()) != {FULL_VARIANTS_PER_FAMILY}:
            raise AssertionError(f"Unbalanced model-change bank: {family_counts}")
    return changed
