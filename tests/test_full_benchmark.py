import json

import pandas as pd
import pytest

from robustness_benchmark.cli.full_benchmark import (
    BENCHMARK_PROTOCOL_VERSION,
    configs_from_frozen_tuning,
    read_protocol_runs,
    run_matches_config,
    selection_key,
    tuning_config_is_current,
    tuning_protocol_is_reusable,
)
from robustness_benchmark.methods.apas_calibration import (
    APAS_CALIBRATION_PROTOCOL_VERSION,
    apas_method_config,
)
from robustness_benchmark.methods.configuration import LOCKED_CONFIGS, TUNING_GRIDS


def test_tuning_selects_by_validity_then_cost_without_changed_models():
    candidates = [
        {
            "config": {"lamb": 0.01},
            "base_validity": 0.8,
            "mean_l1_robust_scale": 0.5,
        },
        {
            "config": {"lamb": 0.1},
            "base_validity": 1.0,
            "mean_l1_robust_scale": 0.9,
        },
        {
            "config": {"lamb": 1.0},
            "base_validity": 1.0,
            "mean_l1_robust_scale": 0.7,
        },
        {
            "config": {"lamb": 2.0},
            "base_validity": 0.9,
            "mean_l1_robust_scale": None,
        },
    ]

    selected = max(candidates, key=selection_key)

    assert selected["config"] == {"lamb": 1.0}


def test_missing_cost_ranks_below_any_finite_cost_at_equal_validity():
    with_cost = {
        "config": {},
        "base_validity": 1.0,
        "mean_l1_robust_scale": 5.0,
    }
    without_cost = {
        "config": {},
        "base_validity": 1.0,
        "mean_l1_robust_scale": None,
    }

    assert selection_key(with_cost) > selection_key(without_cost)


def test_robustness_defining_parameters_are_not_grid_tuned():
    assert set(TUNING_GRIDS) == {"wachter", "roar_lime"}
    roar_delta = {candidate["delta"] for candidate in TUNING_GRIDS["roar_lime"]}
    roar_sample_counts = {
        candidate["num_samples"] for candidate in TUNING_GRIDS["roar_lime"]
    }
    assert roar_delta == {0.1}
    assert roar_sample_counts == {20_000}


def test_tuning_cache_rejects_stale_method_implementations():
    current = TUNING_GRIDS["roar_lime"][0]
    stale = {**current, "implementation": "old_roar"}

    assert tuning_config_is_current("roar_lime", current)
    assert not tuning_config_is_current("roar_lime", stale)
    assert tuning_config_is_current("rnce", LOCKED_CONFIGS["rnce"])
    assert not tuning_config_is_current("rnce", {"delta": 0.005, "bias_delta": 0.005})


def test_tuning_cache_rejects_stale_apas_implementation():
    current = LOCKED_CONFIGS["apas"]
    stale = {**current, "implementation": "old_apas"}
    legacy_fixed_delta = {**current, "delta": 0.005}

    assert tuning_config_is_current("apas", current)
    assert not tuning_config_is_current("apas", stale)
    assert not tuning_config_is_current("apas", legacy_fixed_delta)


def test_apas_locked_config_requires_per_seed_calibration():
    assert "delta" not in LOCKED_CONFIGS["apas"]

    calibration = {
        "selected_delta": 0.0123,
        "calibration_fingerprint": "calibration-1",
    }
    config = apas_method_config(LOCKED_CONFIGS["apas"], calibration)

    assert config["delta"] == pytest.approx(0.0123)
    assert config["calibration_fingerprint"] == "calibration-1"
    assert config["calibration_protocol_version"] == APAS_CALIBRATION_PROTOCOL_VERSION
    assert config["delta_selection"].startswith("max_parameter_linf")
    assert "delta" not in LOCKED_CONFIGS["apas"]


def test_tuning_protocol_can_be_reused_across_benchmark_versions():
    tuning = {
        "benchmark_protocol_version": "full_benchmark_v6_validity_tuning",
        "tuning_protocol_version": "validation_v2_separate_validity_roar_paper",
        "report": {"protocol": {"base_seed": 2026}},
    }

    assert tuning_protocol_is_reusable(tuning, 2026)
    assert not tuning_protocol_is_reusable(tuning, 2029)


def test_frozen_tuning_reuses_grids_but_refreshes_fixed_configs(tmp_path):
    path = tmp_path / "tuning.json"
    path.write_text(
        json.dumps(
            {
                "benchmark_protocol_version": "full_benchmark_v6_validity_tuning",
                "tuning_protocol_version": (
                    "validation_v2_separate_validity_roar_paper"
                ),
                "selected_configs": {
                    "wachter": TUNING_GRIDS["wachter"][0],
                    "roar_lime": TUNING_GRIDS["roar_lime"][0],
                    "apas": {**LOCKED_CONFIGS["apas"], "delta": 0.005},
                    "rnce": LOCKED_CONFIGS["rnce"],
                },
                "report": {
                    "protocol": {"base_seed": 2026},
                    "methods": {
                        "wachter": {"selection": "validation grid"},
                        "roar_lime": {"selection": "validation grid"},
                    },
                },
            }
        )
    )

    configs, report = configs_from_frozen_tuning(
        path, ["wachter", "roar_lime", "apas", "rnce"]
    )

    assert configs["wachter"] == TUNING_GRIDS["wachter"][0]
    assert configs["roar_lime"] == TUNING_GRIDS["roar_lime"][0]
    assert configs["apas"] == LOCKED_CONFIGS["apas"]
    assert "delta" not in configs["apas"]
    assert configs["rnce"] == LOCKED_CONFIGS["rnce"]
    assert report["protocol"]["base_seed"] == 2026
    assert report["protocol"]["frozen_source_sha256"]


def test_summary_inputs_keep_only_current_protocol_rows(tmp_path):
    current = tmp_path / "current.parquet"
    stale = tmp_path / "stale.parquet"
    unversioned = tmp_path / "unversioned.parquet"
    pd.DataFrame(
        {"benchmark_protocol_version": [BENCHMARK_PROTOCOL_VERSION], "x": [1]}
    ).to_parquet(current)
    pd.DataFrame(
        {"benchmark_protocol_version": ["full_benchmark_v0"], "x": [2]}
    ).to_parquet(stale)
    pd.DataFrame({"x": [3]}).to_parquet(unversioned)

    frame = read_protocol_runs([current, stale, unversioned])

    assert frame["x"].tolist() == [1]


def test_summary_inputs_require_at_least_one_current_run(tmp_path):
    stale = tmp_path / "stale.parquet"
    pd.DataFrame(
        {"benchmark_protocol_version": ["full_benchmark_v0"], "x": [2]}
    ).to_parquet(stale)

    with pytest.raises(RuntimeError, match="protocol version"):
        read_protocol_runs([stale])


def test_cached_run_must_match_bank_and_base_model(tmp_path):
    generation_path = tmp_path / "generation.parquet"
    survival_path = tmp_path / "survival.parquet"
    config = {"lamb": 0.1}
    pd.DataFrame(
        {
            "method_config": ['{"lamb": 0.1}'],
            "dataset_row_id": [42],
            "benchmark_protocol_version": [BENCHMARK_PROTOCOL_VERSION],
            "base_model_id": ["base-1"],
            "bank_fingerprint": ["bank-1"],
        }
    ).to_parquet(generation_path)
    pd.DataFrame(
        {
            "change_id": ["change-1"],
            "benchmark_protocol_version": [BENCHMARK_PROTOCOL_VERSION],
            "base_model_id": ["base-1"],
            "bank_fingerprint": ["bank-1"],
        }
    ).to_parquet(survival_path)

    assert run_matches_config(
        generation_path,
        survival_path,
        config,
        [42],
        {"change-1"},
        "base-1",
        "bank-1",
    )
    assert not run_matches_config(
        generation_path,
        survival_path,
        config,
        [42],
        {"change-1"},
        "base-1",
        "different-bank",
    )
    assert not run_matches_config(
        generation_path,
        survival_path,
        config,
        [42],
        {"change-1"},
        "different-base",
        "bank-1",
    )


def test_summary_inputs_are_limited_to_current_invocation(tmp_path):
    runs = tmp_path / "runs.parquet"
    pd.DataFrame(
        {
            "benchmark_protocol_version": [BENCHMARK_PROTOCOL_VERSION] * 3,
            "method": ["wachter", "stale_method", "wachter"],
            "base_model_id": ["base-1", "base-1", "base-2"],
            "value": [1, 2, 3],
        }
    ).to_parquet(runs)

    frame = read_protocol_runs([runs], methods={"wachter"}, base_model_ids={"base-1"})

    assert frame["value"].tolist() == [1]
