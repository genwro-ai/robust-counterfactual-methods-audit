import copy
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score

from robustness_benchmark.core.data import DatasetSplit
from robustness_benchmark.core.model import TorchBinaryModel

BETARCE_GENERATION_ENSEMBLE_SIZE = 32
BETARCE_GENERATION_ENSEMBLE_PROTOCOL = "bootstrap_fixed_initialization_v2"


@dataclass(frozen=True)
class ChangedModel:
    change_id: str
    change_family: str
    change_level: str
    model: TorchBinaryModel
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
    base_model: TorchBinaryModel,
    *,
    radius: float,
    perturbation_seed: int,
    replicate: int = 0,
) -> ChangedModel:
    """Perturb aligned parameters within an exact L-infinity radius.

    A single random direction is scaled so that its largest absolute component
    reaches ``radius``. Consequently every parameter change is bounded by the
    declared radius, while the realized L-infinity distance is non-vacuous.
    """

    if radius <= 0:
        raise ValueError("radius must be positive")
    model = copy.deepcopy(base_model)
    generator = torch.Generator().manual_seed(perturbation_seed)
    directions = [
        torch.empty(
            parameter.shape,
            dtype=parameter.dtype,
            device=parameter.device,
        ).uniform_(-1.0, 1.0, generator=generator)
        for parameter in model.get_torch_model().parameters()
    ]
    maximum = max(float(direction.abs().max().item()) for direction in directions)
    scale = radius / maximum
    with torch.no_grad():
        for parameter, direction in zip(
            model.get_torch_model().parameters(), directions, strict=True
        ):
            parameter.add_(direction * scale)
    model.get_torch_model().eval()
    level = f"linf_{radius:g}"
    return ChangedModel(
        f"bounded_parameter:{level}:r{replicate}",
        "bounded_parameter",
        level,
        model,
        {
            "parameter_norm": "linf",
            "parameter_radius": radius,
            "replicate": replicate,
            "perturbation_seed": perturbation_seed,
            "topology_aligned": True,
        },
    )


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
    model = TorchBinaryModel(X_train.shape[1], hidden_dim=hidden_dim, seed=seed)
    model.train(
        X_train,
        y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        optimizer_name=optimizer_name,
        patience=patience,
    )
    return model


ARCHITECTURE_CANDIDATES: tuple[tuple[int, ...], ...] = (
    (16, 16),
    (32, 16),
    (32, 32),
)
ARCHITECTURE_DEV_SEEDS = (2026, 2027, 2028)
FULL_VARIANTS_PER_FAMILY = 25
FULL_CHANGE_FAMILIES = (
    "seed",
    "bootstrap",
    "deletion",
    "data_addition",
    "label_update",
    "training_config",
    "architecture",
    "bounded_parameter",
)
ADAM_TRAINING_CONFIGS = tuple(
    ("adam", learning_rate, weight_decay)
    for learning_rate in (2.5e-4, 5e-4, 1e-3, 2e-3, 4e-3)
    for weight_decay in (0.0, 1e-5, 1e-4, 1e-3)
    if (learning_rate, weight_decay) != (1e-3, 0.0)
)
SGD_TRAINING_CONFIGS = (
    ("sgd", 1e-2, 0.0),
    ("sgd", 1e-2, 1e-4),
    ("sgd", 1e-2, 1e-3),
    ("sgd", 3e-2, 0.0),
    ("sgd", 3e-2, 1e-4),
    ("sgd", 3e-2, 1e-3),
)
TRAINING_CONFIGS = ADAM_TRAINING_CONFIGS + SGD_TRAINING_CONFIGS


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


def make_changed_models(
    data: DatasetSplit,
    base_model: TorchBinaryModel,
    *,
    base_seed: int,
    epochs: int,
    hidden_dim: tuple[int, ...] = (32, 32),
    profile: str = "full",
) -> list[ChangedModel]:
    """Construct a deterministic model-change bank.

    The full profile contains 25 candidates in each of the eight agreed change
    families. The smoke profile keeps one cheap representative per original
    Day-1 family for fast contract tests.
    """

    if profile not in {"full", "smoke"}:
        raise ValueError(f"Unknown model-bank profile {profile!r}")

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
        architecture: tuple[int, ...] = hidden_dim,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        optimizer_name: str = "adam",
        patience: int = 30,
    ) -> None:
        model = fit_model(
            X_train.reset_index(drop=True),
            y_train.reset_index(drop=True),
            data.X_val,
            data.y_val,
            seed=seed,
            hidden_dim=architecture,
            epochs=epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            optimizer_name=optimizer_name,
            patience=patience,
        )
        changed.append(
            ChangedModel(
                change_id,
                family,
                level,
                model,
                {
                    "training_seed": seed,
                    "hidden_dim": list(architecture),
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                    "optimizer_name": optimizer_name,
                    "patience": patience,
                },
            )
        )

    def add_bounded_parameter_change(radius: float, replicate: int = 0) -> None:
        perturbation_seed = base_seed + 80_000 + round(radius * 1_000_000) + replicate
        change = make_bounded_parameter_change(
            base_model,
            radius=radius,
            perturbation_seed=perturbation_seed,
            replicate=replicate,
        )
        change.metadata["hidden_dim"] = list(hidden_dim)
        changed.append(change)

    replicate_count = 1 if profile == "smoke" else FULL_VARIANTS_PER_FAMILY
    for replicate in range(replicate_count):
        seed = base_seed + 1 + replicate
        train_variant(
            f"seed:new_initialization:r{replicate}:seed{seed}",
            "seed",
            "new_initialization",
            seed=seed,
        )

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

    deletion_fractions = (0.10,) if profile == "smoke" else (0.01, 0.05, 0.10)
    deletion_replicates = (
        (1,)
        if profile == "smoke"
        else _replicates_by_level(FULL_VARIANTS_PER_FAMILY, len(deletion_fractions))
    )
    for fraction, level_replicates in zip(
        deletion_fractions, deletion_replicates, strict=True
    ):
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

    if profile == "full":
        # A fixed update pool has only one genuinely distinct 100% subset. Use
        # eight independent subsets at each lower level and the full pool once.
        update_design = ((0.25, 8), (0.50, 8), (0.75, 8), (1.0, 1))
        for fraction, level_replicates in update_design:
            for replicate in range(level_replicates):
                sampling_seed = base_seed + 30_000 + round(fraction * 1_000) + replicate
                update_n = max(1, round(fraction * len(data.X_update)))
                update_rows = np.sort(
                    np.random.default_rng(sampling_seed).choice(
                        len(data.X_update), size=update_n, replace=False
                    )
                )
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

        label_fractions = (0.01, 0.05, 0.10)
        for fraction, level_replicates in zip(
            label_fractions,
            _replicates_by_level(FULL_VARIANTS_PER_FAMILY, len(label_fractions)),
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

        if len(TRAINING_CONFIGS) != FULL_VARIANTS_PER_FAMILY:
            raise AssertionError("Training configuration grid must contain 25 models")
        for optimizer_name, learning_rate, weight_decay in TRAINING_CONFIGS:
            level = f"{optimizer_name}_lr_{learning_rate:g}_wd_{weight_decay:g}"
            train_variant(
                f"training_config:{level}:seed{base_seed}",
                "training_config",
                level,
                optimizer_name=optimizer_name,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
            )
            changed[-1].metadata["grid_version"] = "sgd_2x3_lr_wd_v2"

        architecture_pool = (
            (8,),
            (16,),
            (32,),
            (64,),
            (128,),
            (8, 8),
            (8, 16),
            (8, 32),
            (16, 8),
            (16, 16),
            (16, 32),
            (32, 8),
            (32, 16),
            (32, 32),
            (32, 64),
            (64, 16),
            (64, 32),
            (64, 64),
            (128, 32),
            (128, 64),
            (128, 128),
            (8, 8, 8),
            (16, 16, 8),
            (16, 16, 16),
            (32, 16, 8),
            (32, 32, 16),
            (32, 32, 32),
            (64, 32, 16),
            (64, 64, 32),
            (16, 16, 16, 16),
            (32, 32, 32, 32),
        )
        alternatives = [
            candidate for candidate in architecture_pool if candidate != hidden_dim
        ][:FULL_VARIANTS_PER_FAMILY]
        if len(alternatives) != FULL_VARIANTS_PER_FAMILY:
            raise AssertionError("Architecture pool must contain 25 alternatives")
        for architecture in alternatives:
            level = "_".join(map(str, architecture))
            train_variant(
                f"architecture:{level}:seed{base_seed}",
                "architecture",
                level,
                architecture=architecture,
            )
    else:
        architecture = (32, 16) if hidden_dim != (32, 16) else (32, 32)
        level = "_".join(map(str, architecture))
        train_variant(
            f"architecture:{level}:seed{base_seed}",
            "architecture",
            level,
            architecture=architecture,
        )

    bounded_radii = (0.01,) if profile == "smoke" else (0.001, 0.005, 0.01, 0.02, 0.05)
    bounded_replicates = 1 if profile == "smoke" else 5
    for radius in bounded_radii:
        for replicate in range(bounded_replicates):
            add_bounded_parameter_change(radius, replicate)

    expected = (
        5
        if profile == "smoke"
        else FULL_VARIANTS_PER_FAMILY * len(FULL_CHANGE_FAMILIES)
    )
    if len(changed) != expected:
        raise AssertionError(f"Expected {expected} changed models, got {len(changed)}")

    if profile == "full":
        family_counts = {
            family: sum(change.change_family == family for change in changed)
            for family in FULL_CHANGE_FAMILIES
        }
        if set(family_counts.values()) != {FULL_VARIANTS_PER_FAMILY}:
            raise AssertionError(f"Unbalanced model-change bank: {family_counts}")

    return changed


def make_betarce_generation_ensemble(
    data: DatasetSplit,
    *,
    base_seed: int,
    hidden_dim: tuple[int, ...],
    epochs: int,
    size: int = BETARCE_GENERATION_ENSEMBLE_SIZE,
) -> list[TorchBinaryModel]:
    """Train BetaRCE's separate bootstrap ensemble with fixed initialization."""

    return [
        make_betarce_generation_model(
            data,
            base_seed=base_seed,
            member=member,
            hidden_dim=hidden_dim,
            epochs=epochs,
        )
        for member in range(size)
    ]


def make_betarce_generation_model(
    data: DatasetSplit,
    *,
    base_seed: int,
    member: int,
    hidden_dim: tuple[int, ...],
    epochs: int,
) -> TorchBinaryModel:
    """Train one resumable member of BetaRCE's bootstrap model space."""

    if member < 0:
        raise ValueError("member must be nonnegative")
    n = len(data.X_train)
    sampling_seed = base_seed + 100_000 + member
    index = np.random.default_rng(sampling_seed).integers(0, n, size=n)
    return fit_model(
        data.X_train.iloc[index].reset_index(drop=True),
        data.y_train.iloc[index].reset_index(drop=True),
        data.X_val,
        data.y_val,
        seed=base_seed,
        hidden_dim=hidden_dim,
        epochs=epochs,
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
    if update_epochs < 1:
        raise ValueError("update_epochs must be positive")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if weight_decay < 0:
        raise ValueError("weight_decay must be nonnegative")
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
        X_tensor = torch.as_tensor(X_update.to_numpy(), dtype=torch.float32)
        y_tensor = torch.as_tensor(y_update.to_numpy(), dtype=torch.float32).reshape(
            -1, 1
        )

        model = copy.deepcopy(base_model)
        network = model.get_torch_model()
        optimizer = torch.optim.Adam(
            network.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        criterion = torch.nn.BCELoss()
        for _ in range(update_epochs):
            network.train()
            for start in range(0, update_n, batch_size):
                stop = min(start + batch_size, update_n)
                optimizer.zero_grad()
                loss = criterion(network(X_tensor[start:stop]), y_tensor[start:stop])
                loss.backward()
                optimizer.step()
        network.eval()

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
