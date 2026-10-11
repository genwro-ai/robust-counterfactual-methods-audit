from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score

from robustness_benchmark.core.classifiers import Classifier, TorchBinaryModel
from robustness_benchmark.core.data import DatasetSplit
from robustness_benchmark.core.model_specs import (
    FULL_VARIANTS_PER_FAMILY,
    MLPSpec,
    ModelSpec,
    perturb_torch_parameters,
    torch_incremental_update,
)

BETARCE_GENERATION_ENSEMBLE_SIZE = 32
BETARCE_GENERATION_ENSEMBLE_PROTOCOL = "bootstrap_fixed_initialization_v2"


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
) -> ChangedModel:
    """Perturb aligned parameters within an exact L-infinity radius.

    A single random direction is scaled so that its largest absolute component
    reaches ``radius``. Consequently every parameter change is bounded by the
    declared radius, while the realized L-infinity distance is non-vacuous.
    Torch networks perturb weights and biases; XGBoost perturbs leaf values.
    """

    if radius <= 0:
        raise ValueError("radius must be positive")
    if spec is None:
        if not isinstance(base_model, TorchBinaryModel):
            raise TypeError("a spec is required to perturb a non-torch model")
        model = perturb_torch_parameters(
            base_model, radius=radius, seed=perturbation_seed
        )
    else:
        model = spec.perturb_parameters(
            base_model, radius=radius, seed=perturbation_seed
        )
    level = f"linf_{radius:g}"
    return ChangedModel(
        f"bounded_parameter:{level}:r{replicate}",
        "bounded_parameter",
        level,
        model,
        {
            "parameter_norm": "linf",
            "parameter_radius": radius,
            "perturbed_parameters": (
                "leaf_values" if base_model.model_type == "xgboost" else "weights"
            ),
            "replicate": replicate,
            "perturbation_seed": perturbation_seed,
            "topology_aligned": True,
        },
    )


def fit_spec(
    spec: ModelSpec,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    *,
    seed: int,
) -> Classifier:
    """Single construction point for every trained model in the bank."""

    return spec.fit(X_train, y_train, X_val, y_val, seed=seed)


def fit_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    *,
    seed: int,
    hidden_dim: tuple[int, ...] = (16, 16),
    epochs: int = 300,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    optimizer_name: str = "adam",
    patience: int = 30,
) -> TorchBinaryModel:
    spec = MLPSpec(
        hidden_dim=hidden_dim,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        optimizer_name=optimizer_name,
        patience=patience,
    )
    model = fit_spec(spec, X_train, y_train, X_val, y_val, seed=seed)
    if not isinstance(model, TorchBinaryModel):
        raise TypeError("an MLP spec must produce a TorchBinaryModel")
    return model


ARCHITECTURE_CANDIDATES: tuple[tuple[int, ...], ...] = (
    (16, 16),
    (32, 16),
    (32, 32),
)
ARCHITECTURE_DEV_SEEDS = (2026, 2027, 2028)
FULL_CHANGE_FAMILIES = MLPSpec.change_families
SMOKE_CHANGE_FAMILIES = (
    "seed",
    "bootstrap",
    "deletion",
    "architecture",
    "bounded_parameter",
    "incremental_update",
)


def _replicates_by_level(total: int, levels: int) -> tuple[int, ...]:
    """Distribute a fixed candidate budget as evenly as possible over levels."""

    quotient, remainder = divmod(total, levels)
    return tuple(quotient + (index < remainder) for index in range(levels))


def architecture_audit(
    data: DatasetSplit,
    *,
    epochs: int,
    candidates: tuple[tuple[int, ...], ...] = ARCHITECTURE_CANDIDATES,
    seeds: tuple[int, ...] = ARCHITECTURE_DEV_SEEDS,
    tolerance: float = 0.01,
) -> tuple[tuple[int, ...], list[dict[str, object]]]:
    """Select the smallest architecture within tolerance of the best mean score."""

    rows: list[dict[str, object]] = []
    means: dict[tuple[int, ...], float] = {}
    for hidden_dim in candidates:
        scores = []
        for seed in seeds:
            model = fit_model(
                data.X_train,
                data.y_train,
                data.X_val,
                data.y_val,
                seed=seed,
                hidden_dim=hidden_dim,
                epochs=epochs,
            )
            prediction = (model.predict(data.X_val).iloc[:, 0] >= 0.5).astype(int)
            score = float(balanced_accuracy_score(data.y_val, prediction))
            scores.append(score)
            rows.append(
                {
                    "hidden_dim": list(hidden_dim),
                    "seed": seed,
                    "validation_balanced_accuracy": score,
                }
            )
        means[hidden_dim] = float(np.mean(scores))

    best_mean = max(means.values())
    eligible = [
        hidden for hidden in candidates if means[hidden] >= best_mean - tolerance
    ]
    selected = min(eligible, key=lambda hidden: (sum(hidden), len(hidden), hidden))
    for row in rows:
        hidden_values = row["hidden_dim"]
        if not isinstance(hidden_values, list):
            raise TypeError("hidden_dim audit field must be a list")
        hidden = tuple(int(width) for width in hidden_values)
        row["mean_validation_balanced_accuracy"] = means[hidden]
        row["best_mean_validation_balanced_accuracy"] = best_mean
        row["within_tolerance"] = hidden in eligible
        row["selected"] = hidden == selected
    return selected, rows


def select_base_spec(
    data: DatasetSplit,
    base: ModelSpec,
    *,
    seeds: tuple[int, ...] = ARCHITECTURE_DEV_SEEDS,
    tolerance: float = 0.01,
) -> tuple[ModelSpec, list[dict[str, object]]]:
    """Select the simplest grid configuration within tolerance of the best.

    Candidates come from ``type(base).selection_grid(base)`` and are scored by
    mean validation balanced accuracy over ``seeds``. Changed models are never
    consulted.
    """

    candidates = type(base).selection_grid(base)
    rows: list[dict[str, object]] = []
    means: list[float] = []
    for index, candidate in enumerate(candidates):
        scores = []
        for seed in seeds:
            model = fit_spec(
                candidate,
                data.X_train,
                data.y_train,
                data.X_val,
                data.y_val,
                seed=seed,
            )
            prediction = (model.predict(data.X_val).iloc[:, 0] >= 0.5).astype(int)
            score = float(balanced_accuracy_score(data.y_val, prediction))
            scores.append(score)
            rows.append(
                {
                    "candidate": index,
                    "model_spec": candidate.to_dict(),
                    "seed": seed,
                    "validation_balanced_accuracy": score,
                }
            )
        means.append(float(np.mean(scores)))

    best_mean = max(means)
    eligible = [
        index for index, mean in enumerate(means) if mean >= best_mean - tolerance
    ]
    selected = min(eligible, key=lambda index: candidates[index].complexity())
    for row in rows:
        index = int(row["candidate"])
        row["mean_validation_balanced_accuracy"] = means[index]
        row["best_mean_validation_balanced_accuracy"] = best_mean
        row["within_tolerance"] = index in eligible
        row["selected"] = index == selected
    return candidates[selected], rows


def make_changed_models(
    data: DatasetSplit,
    base_model: Classifier,
    *,
    base_seed: int,
    epochs: int = 300,
    hidden_dim: tuple[int, ...] = (32, 32),
    profile: str = "full",
    spec: ModelSpec | None = None,
) -> list[ChangedModel]:
    """Construct a deterministic model-change bank.

    ``spec`` describes how ``base_model`` was trained; without one the base is
    an MLP with ``hidden_dim`` trained for ``epochs``. The full profile
    contains 25 candidates in each change family that applies to the model
    class. The smoke profile keeps one cheap representative per applicable
    family in ``SMOKE_CHANGE_FAMILIES`` for fast contract tests.
    """

    if profile not in {"full", "smoke"}:
        raise ValueError(f"Unknown model-bank profile {profile!r}")
    if spec is None:
        spec = MLPSpec(hidden_dim=tuple(hidden_dim), epochs=epochs)
    families = tuple(
        family
        for family in spec.change_families
        if profile == "full" or family in SMOKE_CHANGE_FAMILIES
    )
    smoke = profile == "smoke"

    n = len(data.X_train)
    changed: list[ChangedModel] = []

    def train_variant(
        change_id: str,
        family: str,
        level: str,
        X_train: pd.DataFrame = data.X_train,
        y_train: pd.Series = data.y_train,
        *,
        seed: int = base_seed,
        overrides: dict[str, object] | None = None,
    ) -> None:
        variant_spec = spec.with_overrides(**(overrides or {}))
        model = fit_spec(
            variant_spec,
            X_train.reset_index(drop=True),
            y_train.reset_index(drop=True),
            data.X_val,
            data.y_val,
            seed=seed,
        )
        changed.append(
            ChangedModel(
                change_id,
                family,
                level,
                model,
                {
                    "model_type": spec.model_type,
                    "training_seed": seed,
                    **variant_spec.hyperparameters(),
                },
            )
        )

    def pool_subsets(seed_offset: int):
        # A fixed update pool has only one genuinely distinct 100% subset. Use
        # eight independent subsets at each lower level and the full pool once.
        design = ((1.0, 1),) if smoke else ((0.25, 8), (0.50, 8), (0.75, 8), (1.0, 1))
        for fraction, level_replicates in design:
            for replicate in range(level_replicates):
                sampling_seed = (
                    base_seed + seed_offset + round(fraction * 1_000) + replicate
                )
                update_n = max(1, round(fraction * len(data.X_update)))
                update_rows = np.sort(
                    np.random.default_rng(sampling_seed).choice(
                        len(data.X_update), size=update_n, replace=False
                    )
                )
                yield fraction, replicate, sampling_seed, update_n, update_rows

    replicate_count = 1 if smoke else FULL_VARIANTS_PER_FAMILY

    def build_seed() -> None:
        for replicate in range(replicate_count):
            seed = base_seed + 1 + replicate
            train_variant(
                f"seed:new_initialization:r{replicate}:seed{seed}",
                "seed",
                "new_initialization",
                seed=seed,
            )

    def build_bootstrap() -> None:
        for replicate in range(replicate_count):
            sampling_seed = base_seed + 10_000 + replicate
            index = np.random.default_rng(sampling_seed).integers(0, n, size=n)
            train_variant(
                f"bootstrap:100_percent:r{replicate}:seed{sampling_seed}",
                "bootstrap",
                "100_percent",
                data.X_train.iloc[index],
                data.y_train.iloc[index],
            )
            changed[-1].metadata.update(
                {"sampling_seed": sampling_seed, "replicate": replicate, "n": n}
            )

    def build_deletion() -> None:
        fractions = (0.10,) if smoke else (0.01, 0.05, 0.10)
        replicates = (
            (1,)
            if smoke
            else _replicates_by_level(FULL_VARIANTS_PER_FAMILY, len(fractions))
        )
        for fraction, level_replicates in zip(fractions, replicates, strict=True):
            for replicate in range(level_replicates):
                sampling_seed = base_seed + 20_000 + round(fraction * 1_000) + replicate
                keep = np.sort(
                    np.random.default_rng(sampling_seed).choice(
                        n, size=round((1.0 - fraction) * n), replace=False
                    )
                )
                level = f"{fraction:g}"
                train_variant(
                    f"deletion:{level}:r{replicate}:seed{sampling_seed}",
                    "deletion",
                    level,
                    data.X_train.iloc[keep],
                    data.y_train.iloc[keep],
                )
                changed[-1].metadata.update(
                    {
                        "deleted_fraction": fraction,
                        "sampling_seed": sampling_seed,
                        "replicate": replicate,
                    }
                )

    def build_data_addition() -> None:
        for fraction, replicate, sampling_seed, update_n, update_rows in pool_subsets(
            30_000
        ):
            X_augmented = pd.concat(
                [data.X_train, data.X_update.iloc[update_rows]], ignore_index=True
            )
            y_augmented = pd.concat(
                [data.y_train, data.y_update.iloc[update_rows]], ignore_index=True
            )
            level = f"{fraction:g}_update_pool"
            train_variant(
                f"data_addition:{level}:r{replicate}:seed{sampling_seed}",
                "data_addition",
                level,
                X_augmented,
                y_augmented,
            )
            changed[-1].metadata.update(
                {
                    "added_n": update_n,
                    "update_fraction": fraction,
                    "sampling_seed": sampling_seed,
                    "replicate": replicate,
                }
            )

    def build_label_update() -> None:
        fractions = (0.01, 0.05, 0.10)
        for fraction, level_replicates in zip(
            fractions,
            _replicates_by_level(FULL_VARIANTS_PER_FAMILY, len(fractions)),
            strict=True,
        ):
            for replicate in range(level_replicates):
                sampling_seed = base_seed + 40_000 + round(fraction * 1_000) + replicate
                changed_n = max(1, round(fraction * n))
                positions = np.random.default_rng(sampling_seed).choice(
                    n, size=changed_n, replace=False
                )
                updated_labels = data.y_train.reset_index(drop=True).copy()
                updated_labels.iloc[positions] = 1 - updated_labels.iloc[positions]
                level = f"{fraction:g}"
                train_variant(
                    f"label_update:{level}:r{replicate}:seed{sampling_seed}",
                    "label_update",
                    level,
                    data.X_train,
                    updated_labels,
                )
                changed[-1].metadata.update(
                    {
                        "updated_fraction": fraction,
                        "updated_n": changed_n,
                        "sampling_seed": sampling_seed,
                        "replicate": replicate,
                    }
                )

    def build_training_config() -> None:
        variants = spec.training_config_variants()
        if len(variants) != FULL_VARIANTS_PER_FAMILY:
            raise AssertionError("Training configuration grid must contain 25 models")
        for level, overrides in variants:
            train_variant(
                f"training_config:{level}:seed{base_seed}",
                "training_config",
                level,
                overrides=overrides,
            )
            if spec.model_type == "mlp":
                changed[-1].metadata["grid_version"] = "sgd_2x3_lr_wd_v2"

    def build_architecture() -> None:
        variants = spec.architecture_variants()
        if len(variants) != FULL_VARIANTS_PER_FAMILY:
            raise AssertionError("Architecture pool must contain 25 alternatives")
        if smoke:
            variants = variants[:1]
        for level, overrides in variants:
            train_variant(
                f"architecture:{level}:seed{base_seed}",
                "architecture",
                level,
                overrides=overrides,
            )

    def build_bounded_parameter() -> None:
        radii = (0.01,) if smoke else (0.001, 0.005, 0.01, 0.02, 0.05)
        for radius in radii:
            for replicate in range(1 if smoke else 5):
                perturbation_seed = (
                    base_seed + 80_000 + round(radius * 1_000_000) + replicate
                )
                change = make_bounded_parameter_change(
                    base_model,
                    radius=radius,
                    perturbation_seed=perturbation_seed,
                    replicate=replicate,
                    spec=spec,
                )
                change.metadata.update(
                    {"model_type": spec.model_type, **spec.hyperparameters()}
                )
                changed.append(change)

    def build_incremental_update() -> None:
        # Seeds are disjoint from the APAS calibration replicas (base + 200000).
        for fraction, replicate, sampling_seed, update_n, update_rows in pool_subsets(
            50_000
        ):
            model = spec.incremental_update(
                base_model,
                data.X_update.iloc[update_rows].reset_index(drop=True),
                data.y_update.iloc[update_rows].reset_index(drop=True),
            )
            level = f"{fraction:g}_update_pool"
            changed.append(
                ChangedModel(
                    f"incremental_update:{level}:r{replicate}:seed{sampling_seed}",
                    "incremental_update",
                    level,
                    model,
                    {
                        "model_type": spec.model_type,
                        **spec.hyperparameters(),
                        "initialized_from_base_checkpoint": True,
                        "update_n": update_n,
                        "update_fraction": fraction,
                        "sampling_seed": sampling_seed,
                        "replicate": replicate,
                    },
                )
            )

    builders = {
        "seed": build_seed,
        "bootstrap": build_bootstrap,
        "deletion": build_deletion,
        "data_addition": build_data_addition,
        "label_update": build_label_update,
        "training_config": build_training_config,
        "architecture": build_architecture,
        "bounded_parameter": build_bounded_parameter,
        "incremental_update": build_incremental_update,
    }
    for family in families:
        builders[family]()

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


def _resolve_spec(
    spec: ModelSpec | None,
    hidden_dim: tuple[int, ...] | None,
    epochs: int | None,
) -> ModelSpec:
    if spec is not None:
        return spec
    if hidden_dim is None or epochs is None:
        raise ValueError("pass a spec, or hidden_dim and epochs for an MLP")
    return MLPSpec(hidden_dim=tuple(hidden_dim), epochs=epochs)


def make_betarce_generation_ensemble(
    data: DatasetSplit,
    *,
    base_seed: int,
    hidden_dim: tuple[int, ...] | None = None,
    epochs: int | None = None,
    size: int = BETARCE_GENERATION_ENSEMBLE_SIZE,
    spec: ModelSpec | None = None,
) -> list[Classifier]:
    """Train BetaRCE's separate bootstrap ensemble with fixed initialization."""

    return [
        make_betarce_generation_model(
            data,
            base_seed=base_seed,
            member=member,
            hidden_dim=hidden_dim,
            epochs=epochs,
            spec=spec,
        )
        for member in range(size)
    ]


def make_betarce_generation_model(
    data: DatasetSplit,
    *,
    base_seed: int,
    member: int,
    hidden_dim: tuple[int, ...] | None = None,
    epochs: int | None = None,
    spec: ModelSpec | None = None,
) -> Classifier:
    """Train one resumable member of BetaRCE's bootstrap model space."""

    if member < 0:
        raise ValueError("member must be nonnegative")
    n = len(data.X_train)
    sampling_seed = base_seed + 100_000 + member
    index = np.random.default_rng(sampling_seed).integers(0, n, size=n)
    return fit_spec(
        _resolve_spec(spec, hidden_dim, epochs),
        data.X_train.iloc[index].reset_index(drop=True),
        data.y_train.iloc[index].reset_index(drop=True),
        data.X_val,
        data.y_val,
        seed=base_seed,
    )


def make_apas_calibration_models(
    data: DatasetSplit,
    base_model: TorchBinaryModel,
    *,
    base_seed: int,
    size: int = 10,
    update_epochs: int = 1,
    batch_size: int = 8,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
) -> list[ChangedModel]:
    """Create checkpoint-aligned update replicas used only to calibrate APAS.

    Each replica starts from an exact copy of the final base checkpoint and
    receives one bootstrap-sized incremental update from the reserved update
    pool. Sampling seeds vary the update batch, never network initialization.
    The replicas are calibration data and must not enter the evaluation bank.
    """

    if size < 1:
        raise ValueError("size must be positive")
    update_n = len(data.X_update)
    if update_n < 1:
        raise ValueError("the reserved update pool is empty")

    models: list[ChangedModel] = []
    for replicate in range(size):
        sampling_seed = base_seed + 200_000 + replicate
        sampled_positions = np.random.default_rng(sampling_seed).integers(
            0, update_n, size=update_n
        )
        X_update = data.X_update.iloc[sampled_positions].reset_index(drop=True)
        y_update = data.y_update.iloc[sampled_positions].reset_index(drop=True)
        model = torch_incremental_update(
            base_model,
            X_update,
            y_update,
            epochs=update_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )

        models.append(
            ChangedModel(
                change_id=f"apas_calibration:r{replicate}:seed{sampling_seed}",
                change_family="apas_calibration",
                change_level="bootstrap_update_pool",
                model=model,
                metadata={
                    "role": "apas_delta_calibration",
                    "excluded_from_evaluation_changes": True,
                    "replicate": replicate,
                    "sampling_seed": sampling_seed,
                    "sampling_scheme": "bootstrap_with_replacement",
                    "update_pool_n": update_n,
                    "sampled_n": update_n,
                    "update_epochs": update_epochs,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                    "initialized_from_base_checkpoint": True,
                    "fresh_optimizer": "adam",
                },
            )
        )
    return models
