import copy
from collections import Counter

import numpy as np
import pandas as pd
import pytest

from robustness_benchmark.core import training
from robustness_benchmark.core.classifiers import (
    LogisticClassifier,
    XGBoostClassifier,
)
from robustness_benchmark.core.model_specs import (
    FULL_VARIANTS_PER_FAMILY,
    LogisticSpec,
    MLPSpec,
    XGBoostSpec,
    spec_from_dict,
    spec_from_manifest,
)
from robustness_benchmark.core.training import (
    make_betarce_generation_model,
    make_bounded_parameter_change,
    make_changed_models,
    parameter_linf_distance,
    select_base_spec,
)


class Split:
    rng = np.random.default_rng(0)
    X_train = pd.DataFrame(rng.normal(size=(60, 3)), columns=["a", "b", "c"])
    y_train = pd.Series((X_train["a"] + X_train["b"] > 0).astype(int), name="target")
    X_update = pd.DataFrame(rng.normal(size=(16, 3)), columns=["a", "b", "c"])
    y_update = pd.Series((X_update["a"] + X_update["b"] > 0).astype(int))
    X_val = pd.DataFrame(rng.normal(size=(20, 3)), columns=["a", "b", "c"])
    y_val = pd.Series((X_val["a"] + X_val["b"] > 0).astype(int))


SMALL_XGBOOST = XGBoostSpec(n_estimators=5, max_depth=2)


@pytest.mark.parametrize(
    ("spec", "families"),
    [
        (LogisticSpec(), 6),
        (SMALL_XGBOOST, 9),
    ],
)
def test_full_bank_per_model_class(spec, families):
    base = spec.fit(Split.X_train, Split.y_train, Split.X_val, Split.y_val, seed=2026)
    before = base.predict(Split.X_val).to_numpy()

    changes = make_changed_models(
        Split(), base, base_seed=2026, profile="full", spec=spec
    )

    counts = Counter(change.change_family for change in changes)
    assert set(counts) == set(spec.change_families)
    assert len(counts) == families
    assert set(counts.values()) == {FULL_VARIANTS_PER_FAMILY}
    assert len({change.change_id for change in changes}) == len(changes)
    assert all(type(change.model) is type(base) for change in changes)
    assert all(change.metadata["model_type"] == spec.model_type for change in changes)
    np.testing.assert_array_equal(base.predict(Split.X_val).to_numpy(), before)


def test_logistic_has_no_seed_or_architecture_family():
    assert "seed" not in LogisticSpec.change_families
    assert "architecture" not in LogisticSpec.change_families
    assert LogisticSpec().architecture_variants() == []


def test_mlp_bank_keeps_its_families_and_adds_incremental_update():
    assert MLPSpec.change_families[:8] == (
        "seed",
        "bootstrap",
        "deletion",
        "data_addition",
        "label_update",
        "training_config",
        "architecture",
        "bounded_parameter",
    )
    assert MLPSpec.change_families[8] == "incremental_update"


def test_candidate_lists_exclude_the_base_and_leave_reserves():
    bases = [
        *LogisticSpec.selection_grid(LogisticSpec()),
        *XGBoostSpec.selection_grid(XGBoostSpec()),
        MLPSpec(),
    ]
    for spec in bases:
        for candidates in (
            spec.training_config_variants(),
            spec.architecture_variants(),
        ):
            if not candidates:
                continue
            levels = [level for level, _ in candidates]
            assert len(candidates) > FULL_VARIANTS_PER_FAMILY
            assert len(set(levels)) == len(levels)
            assert all(
                spec.with_overrides(**changes) != spec for _, changes in candidates
            )


def test_configuration_family_skips_candidates_that_reproduce_the_base(monkeypatch):
    spec = LogisticSpec()
    base = spec.fit(Split.X_train, Split.y_train, Split.X_val, Split.y_val, seed=1)
    real_fit = training.fit_spec

    def fit_or_copy_base(candidate, *args, **kwargs):
        # Every L1 candidate reproduces the base model exactly.
        if candidate.l1_ratio == 1.0:
            return copy.deepcopy(base)
        return real_fit(candidate, *args, **kwargs)

    monkeypatch.setattr(training, "fit_spec", fit_or_copy_base)
    changes = make_changed_models(Split(), base, base_seed=1, spec=spec)

    configs = [c for c in changes if c.change_family == "training_config"]
    assert len(configs) == FULL_VARIANTS_PER_FAMILY
    assert all(c.metadata["l1_ratio"] != 1.0 for c in configs)
    skipped = configs[0].metadata["skipped_candidates"]
    assert skipped and all(level.startswith("l1_") for level in skipped)


def test_logistic_parameter_perturbation_reaches_radius():
    spec = LogisticSpec()
    base = spec.fit(Split.X_train, Split.y_train, Split.X_val, Split.y_val, seed=1)

    change = make_bounded_parameter_change(
        base, radius=0.02, perturbation_seed=5, spec=spec
    )

    assert isinstance(change.model, LogisticClassifier)
    assert parameter_linf_distance(base, change.model) == pytest.approx(0.02)


def test_xgboost_parameter_perturbation_moves_leaves_by_radius():
    base = SMALL_XGBOOST.fit(
        Split.X_train, Split.y_train, Split.X_val, Split.y_val, seed=1
    )

    change = make_bounded_parameter_change(
        base, radius=0.02, perturbation_seed=5, spec=SMALL_XGBOOST
    )

    assert isinstance(change.model, XGBoostClassifier)
    assert change.metadata["perturbed_parameters"] == "leaf_values"
    leaf_shift = np.abs(change.model.leaf_values() - base.leaf_values())
    assert leaf_shift.max() == pytest.approx(0.02, abs=1e-6)


def test_incremental_update_starts_from_the_base_model():
    X = pd.concat([Split.X_train, Split.X_update], ignore_index=True)
    y = pd.concat([Split.y_train, Split.y_update], ignore_index=True)
    spec = MLPSpec(hidden_dim=(4,), epochs=5)
    base = spec.fit(Split.X_train, Split.y_train, Split.X_val, Split.y_val, seed=1)

    updated = spec.incremental_update(base, X, y, Split.X_val, Split.y_val)

    distance = parameter_linf_distance(base, updated)
    assert distance is not None and 0 < distance < 0.1

    xgb_base = SMALL_XGBOOST.fit(
        Split.X_train, Split.y_train, Split.X_val, Split.y_val, seed=1
    )
    xgb_updated = SMALL_XGBOOST.incremental_update(
        xgb_base, X, y, Split.X_val, Split.y_val
    )
    trees = xgb_updated.num_trees - xgb_base.num_trees
    assert 1 <= trees <= SMALL_XGBOOST.n_estimators
    with pytest.raises(NotImplementedError):
        LogisticSpec().incremental_update(base, X, y, Split.X_val, Split.y_val)


def test_betarce_ensemble_member_uses_spec():
    member = make_betarce_generation_model(
        Split(), base_seed=3, member=0, spec=SMALL_XGBOOST
    )

    assert isinstance(member, XGBoostClassifier)
    assert member.seed == 3


def test_select_base_spec_picks_lowest_validation_log_loss():
    selected, rows = select_base_spec(Split(), LogisticSpec(), seeds=(1,))

    assert isinstance(selected, LogisticSpec)
    assert sum(row["selected"] for row in rows) == 1
    best = min(rows, key=lambda row: row["mean_validation_log_loss"])
    assert selected.C == best["model_spec"]["C"]


def test_bank_uses_relative_radii_and_pairs_incremental_with_data_addition():
    spec = SMALL_XGBOOST
    base = spec.fit(Split.X_train, Split.y_train, Split.X_val, Split.y_val, seed=1)

    changes = make_changed_models(Split(), base, base_seed=1, spec=spec)

    scale = spec.parameter_scale(base)
    for change in changes:
        if change.change_family == "bounded_parameter":
            relative = change.metadata["relative_parameter_radius"]
            assert change.change_level == f"relative_linf_{relative:g}"
            shift = np.abs(change.model.leaf_values() - base.leaf_values())
            assert shift.max() == pytest.approx(relative * scale, rel=1e-3)
    ids = {change.change_id for change in changes}
    incremental = [c for c in changes if c.change_family == "incremental_update"]
    assert len(incremental) == FULL_VARIANTS_PER_FAMILY
    assert all(c.metadata["paired_change_id"] in ids for c in incremental)


def test_spec_serialization_round_trip():
    for spec in (MLPSpec(hidden_dim=(8, 4)), LogisticSpec(C=3.0), XGBoostSpec()):
        assert spec_from_dict(spec.to_dict()) == spec
        assert spec_from_manifest({"model_spec": spec.to_dict()}) == spec


def test_legacy_manifest_is_an_mlp_bank():
    spec = spec_from_manifest({"selected_architecture": [32, 32], "epochs": 300})

    assert spec == MLPSpec(hidden_dim=(32, 32), epochs=300)
    assert spec.id_fragment() == "mlp_32_32"
