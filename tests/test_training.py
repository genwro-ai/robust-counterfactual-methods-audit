from collections import Counter

import pandas as pd
import pytest
import torch

from robustness_benchmark.core.model import TorchBinaryModel
from robustness_benchmark.core.training import (
    FULL_CHANGE_FAMILIES,
    FULL_VARIANTS_PER_FAMILY,
    SGD_TRAINING_CONFIGS,
    TRAINING_CONFIGS,
    make_apas_calibration_models,
    make_bounded_parameter_change,
    make_changed_models,
    parameter_linf_distance,
)


class SmallSplit:
    X_train = pd.DataFrame({"a": range(20), "b": range(20)})
    y_train = pd.Series(([0, 1] * 10), name="target")
    X_update = pd.DataFrame({"a": range(8), "b": range(8)})
    y_update = pd.Series(([0, 1] * 4), name="target")
    X_val = X_train.iloc[:4]
    y_val = y_train.iloc[:4]


def test_training_configuration_grid_has_no_undertrained_sgd_settings():
    assert len(TRAINING_CONFIGS) == FULL_VARIANTS_PER_FAMILY
    assert SGD_TRAINING_CONFIGS == (
        ("sgd", 1e-2, 0.0),
        ("sgd", 1e-2, 1e-4),
        ("sgd", 1e-2, 1e-3),
        ("sgd", 3e-2, 0.0),
        ("sgd", 3e-2, 1e-4),
        ("sgd", 3e-2, 1e-3),
    )


def test_bounded_parameter_change_reaches_declared_linf_radius():
    base = TorchBinaryModel(input_dim=3, hidden_dim=(4,), seed=1)

    change = make_bounded_parameter_change(
        base,
        radius=0.005,
        perturbation_seed=123,
    )

    assert parameter_linf_distance(base, change.model) == pytest.approx(0.005)
    assert change.change_family == "bounded_parameter"
    assert change.metadata["parameter_radius"] == 0.005


def test_bounded_parameter_change_rejects_nonpositive_radius():
    base = TorchBinaryModel(input_dim=2, hidden_dim=(), seed=1)

    with pytest.raises(ValueError, match="positive"):
        make_bounded_parameter_change(base, radius=0.0, perturbation_seed=123)


def test_apas_calibration_updates_copy_checkpoint_and_vary_only_update_sample():
    base = TorchBinaryModel(input_dim=2, hidden_dim=(4,), seed=2026)
    original = {
        name: value.detach().clone()
        for name, value in base.get_torch_model().state_dict().items()
    }

    replicas = make_apas_calibration_models(
        SmallSplit(),
        base,
        base_seed=2026,
        size=3,
        update_epochs=1,
        batch_size=2,
    )

    assert len(replicas) == 3
    assert len({replica.metadata["sampling_seed"] for replica in replicas}) == 3
    assert all(
        replica.metadata["initialized_from_base_checkpoint"] for replica in replicas
    )
    assert all(parameter_linf_distance(base, replica.model) > 0 for replica in replicas)
    assert all(
        torch.equal(value, original[name])
        for name, value in base.get_torch_model().state_dict().items()
    )


def test_full_bank_has_25_unique_candidates_per_family(monkeypatch):
    def untrained_model(X_train, _y_train, _X_val, _y_val, *, seed, hidden_dim, **_):
        return TorchBinaryModel(X_train.shape[1], hidden_dim=hidden_dim, seed=seed)

    monkeypatch.setattr("robustness_benchmark.core.training.fit_model", untrained_model)
    base = TorchBinaryModel(input_dim=2, hidden_dim=(32, 32), seed=2026)

    changes = make_changed_models(
        SmallSplit(),
        base,
        base_seed=2026,
        epochs=1,
        hidden_dim=(32, 32),
        profile="full",
    )

    counts = Counter(change.change_family for change in changes)
    assert counts == {
        family: FULL_VARIANTS_PER_FAMILY for family in FULL_CHANGE_FAMILIES
    }
    assert len({change.change_id for change in changes}) == len(changes) == 200
    full_updates = [
        change
        for change in changes
        if change.change_family == "data_addition"
        and change.metadata["update_fraction"] == 1.0
    ]
    assert len(full_updates) == 1
