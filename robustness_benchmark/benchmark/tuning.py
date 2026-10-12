"""Validation tuning of Wachter, ROAR-LIME and ROAR; fixed settings for the rest.

Candidates are ranked by base-model validity on adverse validation factuals,
then by lower Gower distance; changed models are never consulted. Tuning runs
on one base model and is reused for the others.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from robustness_benchmark.benchmark.run import score_generation
from robustness_benchmark.core.data import DatasetSplit
from robustness_benchmark.core.factuals import select_factuals
from robustness_benchmark.core.model_bank import load_base_model, model_bank_fingerprint
from robustness_benchmark.core.protocol import TUNING_FACTUALS_SEED_OFFSET
from robustness_benchmark.core.provenance import file_sha256, write_json
from robustness_benchmark.methods.configuration import TUNING_GRIDS, locked_config
from robustness_benchmark.methods.registry import generate_counterfactuals, make_task

TUNING_PROTOCOL_VERSION = "validation_v3_gower_cost"


def score_configuration(
    method: str,
    config: dict[str, object],
    task,
    factuals: pd.DataFrame,
) -> dict[str, object]:
    counterfactuals, generation = generate_counterfactuals(
        method, task, factuals, method_config=config
    )
    score_generation(task, factuals, counterfactuals, generation)
    base_valid = [bool(row["base_valid"]) for row in generation]
    valid_generation = [row for row in generation if row["base_valid"]]
    return {
        "config": config,
        "base_validity": float(np.mean(base_valid)),
        "mean_gower_distance": (
            float(np.mean([row["gower_distance"] for row in valid_generation]))
            if valid_generation
            else None
        ),
    }


def selection_key(candidate: dict[str, object]) -> tuple[float, float]:
    """Order tuning candidates by base-model validity, then by lower cost.

    This mirrors the original ROAR tuning criterion (validity-driven lambda
    selection); changed models are never consulted, so tuning cannot leak the
    evaluation targets.
    """

    validity = candidate["base_validity"]
    if not isinstance(validity, (int, float)):
        raise TypeError("Tuning base validity must be numeric")
    cost = candidate["mean_gower_distance"]
    numeric_cost = float(cost) if isinstance(cost, (int, float)) else float("inf")
    return (float(validity), -numeric_cost)


def tune_methods(
    bank: Path,
    data,
    methods: list[str],
    tuning_factuals: int,
    base_seed: int,
    model_type: str | None = None,
) -> tuple[dict[str, dict[str, object]], dict[str, object]]:
    configs: dict[str, dict[str, object]] = {}
    for method in methods:
        configs[method] = locked_config(method, data.name, model_type)
    base_id, base_model = load_base_model(bank, base_seed)
    factuals, _, _ = select_factuals(
        base_model,
        data.X_val,
        tuning_factuals,
        seed=base_seed + TUNING_FACTUALS_SEED_OFFSET,
    )
    task = make_task(
        base_model,
        data.X_train,
        data.y_train,
        seed=base_seed,
        schema=data.feature_schema,
    )
    report: dict[str, object] = {
        "protocol": {
            "base_seed": base_seed,
            "split": "validation",
            "factuals_n": len(factuals),
            "objective": (
                "maximize base-model validity, then minimize "
                "Gower distance; changed models are never consulted"
            ),
            "base_model_id": base_id,
            "bank_fingerprint": model_bank_fingerprint(bank),
        },
        "methods": {},
    }
    for method in methods:
        grid = TUNING_GRIDS.get(method)
        if grid is None:
            report["methods"][method] = {
                "selection": "fixed published/upstream setting or parameter-free",
                "selected": configs[method],
            }
            continue
        candidates = [
            score_configuration(method, config, task, factuals)
            for config in tqdm(
                grid,
                desc=f"tuning · {method}",
                unit="config",
                dynamic_ncols=True,
                leave=False,
            )
        ]
        selected = max(candidates, key=selection_key)
        selected_config = selected["config"]
        if not isinstance(selected_config, dict):
            raise TypeError("Selected tuning configuration must be a dictionary")
        configs[method] = dict(selected_config)
        report["methods"][method] = {
            "selection": "validation grid",
            "selected": configs[method],
            "candidates": candidates,
        }
    return configs, report


def tuning_config_is_current(
    method: str,
    config: object,
    dataset: str | None = None,
    model_type: str | None = None,
) -> bool:
    """Return whether a cached method config belongs to the current protocol."""

    if not isinstance(config, dict):
        return False
    grid = TUNING_GRIDS.get(method)
    if grid is not None:
        return config in grid
    return config == locked_config(method, dataset, model_type)


def tuning_protocol_is_reusable(
    tuning: dict[str, object], base_seed: int, bank_fingerprint: str
) -> bool:
    """Reuse tuning across benchmark versions only for the same bank and seed."""

    report = tuning.get("report", {})
    if not isinstance(report, dict):
        return False
    protocol = report.get("protocol", {})
    return (
        isinstance(protocol, dict)
        and tuning.get("tuning_protocol_version") == TUNING_PROTOCOL_VERSION
        and protocol.get("base_seed") == base_seed
        and protocol.get("bank_fingerprint") == bank_fingerprint
    )


def configs_from_frozen_tuning(
    path: Path,
    methods: list[str],
    dataset: str | None = None,
    model_type: str | None = None,
) -> tuple[dict[str, dict[str, object]], dict[str, object]]:
    """Load only genuinely tuned settings and refresh every fixed configuration."""

    source = json.loads(path.read_text())
    if source.get("tuning_protocol_version") != TUNING_PROTOCOL_VERSION:
        raise ValueError(f"Frozen tuning file uses an incompatible protocol: {path}")
    cached_configs = source.get("selected_configs", {})
    if not isinstance(cached_configs, dict):
        raise TypeError(f"Frozen tuning file has no selected_configs mapping: {path}")
    source_report = source.get("report", {})
    if not isinstance(source_report, dict):
        raise TypeError(f"Frozen tuning file has no report mapping: {path}")
    source_method_report = source_report.get("methods", {})
    if not isinstance(source_method_report, dict):
        source_method_report = {}

    configs: dict[str, dict[str, object]] = {}
    method_report: dict[str, object] = {}
    for method in methods:
        if method in TUNING_GRIDS:
            cached = cached_configs.get(method)
            if not tuning_config_is_current(method, cached):
                raise ValueError(
                    f"Frozen tuning file has no current configuration for {method!r}"
                )
            configs[method] = dict(cached)
            method_report[method] = source_method_report.get(method, {})
        else:
            configs[method] = locked_config(method, dataset, model_type)
            method_report[method] = {
                "selection": "fixed published/upstream setting or parameter-free",
                "selected": configs[method],
            }

    source_protocol = source_report.get("protocol", {})
    protocol = dict(source_protocol) if isinstance(source_protocol, dict) else {}
    protocol.update(
        {
            "frozen_source": str(path.resolve()),
            "frozen_source_sha256": file_sha256(path),
        }
    )
    return configs, {"protocol": protocol, "methods": method_report}


def resolve_configs(
    *,
    output: Path,
    bank: Path,
    data: DatasetSplit,
    methods: list[str],
    tuning_factuals: int,
    tuning_base_seed: int,
    model_type: str | None = None,
    frozen_tuning: Path | None = None,
) -> dict[str, dict[str, object]]:
    """Every method's configuration, recorded in output/tuning.json.

    Configurations already tuned for this bank and seed are reused; with
    frozen_tuning the tuned ones come from that file instead.
    """

    tuning_path = output / "tuning.json"
    if frozen_tuning is not None:
        configs, report = configs_from_frozen_tuning(
            frozen_tuning, methods, data.name, model_type
        )
    else:
        cached = json.loads(tuning_path.read_text()) if tuning_path.is_file() else {}
        reusable_protocol = tuning_protocol_is_reusable(
            cached, tuning_base_seed, model_bank_fingerprint(bank)
        )
        cached_configs = cached.get("selected_configs", {}) if reusable_protocol else {}
        reusable = {
            method
            for method in methods
            if tuning_config_is_current(
                method, cached_configs.get(method), data.name, model_type
            )
        }
        to_tune = [method for method in methods if method not in reusable]
        if not to_tune:
            return {method: dict(cached_configs[method]) for method in methods}
        fresh_configs, fresh_report = tune_methods(
            bank, data, to_tune, tuning_factuals, tuning_base_seed, model_type
        )
        configs = {
            method: dict(cached_configs[method])
            if method in reusable
            else fresh_configs[method]
            for method in methods
        }
        cached_methods = cached.get("report", {}).get("methods", {})
        report = {
            "protocol": fresh_report["protocol"],
            "methods": {
                **{method: cached_methods.get(method, {}) for method in reusable},
                **fresh_report["methods"],
            },
        }
    write_json(
        tuning_path,
        {
            "tuning_protocol_version": TUNING_PROTOCOL_VERSION,
            "methods": methods,
            "selected_configs": configs,
            "report": report,
        },
    )
    return configs
