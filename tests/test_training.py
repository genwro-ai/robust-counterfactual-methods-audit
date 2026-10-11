import zlib
from collections import Counter

import pandas as pd
import pytest
import torch

from robustness_benchmark.core.model import TorchBinaryModel
from robustness_benchmark.core.model_specs import (
    FULL_VARIANTS_PER_FAMILY,
    SGD_TRAINING_CONFIGS,
    TRAINING_CONFIGS,
)
from robustness_benchmark.core.training import (
    FULL_CHANGE_FAMILIES,
    make_apas_calibration_models,
    make_bounded_parameter_change,
    make_changed_models,
    make_update_replicas,
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
    def untrained_model(spec, X_train, _y_train, _X_val, _y_val, *, seed):
        # Distinct configurations get distinct weights, so none reproduces the base.
        return spec.build(X_train.shape[1], seed + zlib.crc32(repr(spec).encode()))

    monkeypatch.setattr("robustness_benchmark.core.training.fit_spec", untrained_model)
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
    assert len({change.change_id for change in changes}) == len(changes) == 225
    full_updates = [
        change
        for change in changes
        if change.change_family == "data_addition"
        and change.metadata["update_fraction"] == 1.0
    ]
    assert len(full_updates) == 1


def test_rnce_calibration_follows_the_authors_incremental_protocol(tmp_path):
    from robustness_benchmark.methods.calibration import (
        RNCE_UPDATE_CALIBRATION,
        load_or_create_calibration,
    )

    class Split(SmallSplit):
        name = "small"
        X_update = pd.DataFrame({"a": range(40), "b": range(40)})
        y_update = pd.Series(([0, 1] * 20), name="target")
        X_test = SmallSplit.X_train
        y_test = SmallSplit.y_train

    base = TorchBinaryModel(input_dim=2, hidden_dim=(4,), seed=2026)
    replicas = make_update_replicas(
        Split(),
        base,
        base_seed=2026,
        name="rnce",
        size=3,
        seed_offset=210_000,
        sample_fraction=0.1,
        replace=False,
    )
    calibration = load_or_create_calibration(
        RNCE_UPDATE_CALIBRATION,
        output=tmp_path,
        data=Split(),
        base_model=base,
        base_model_id="base",
        base_seed=2026,
        bank_fingerprint="bank",
    )

    assert all(replica.metadata["sampled_n"] == 4 for replica in replicas)
    assert all(
        replica.metadata["sampling_scheme"] == "without_replacement"
        for replica in replicas
    )
    distances = [r["parameter_linf_distance"] for r in calibration["replicas"]]
    assert len(distances) == 10
    assert calibration["selected_delta"] == pytest.approx(sum(distances) / 10)
    assert (
        tmp_path / "calibration" / "rnce" / "seed_2026" / "calibration.json"
    ).is_file()
