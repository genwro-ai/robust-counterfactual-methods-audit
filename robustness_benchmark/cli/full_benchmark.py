import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from robustness_benchmark.core.data import load_dataset
from robustness_benchmark.core.model_bank import (
    load_base_model,
    load_model_bank,
    model_bank_fingerprint,
    select_factuals,
)
from robustness_benchmark.core.provenance import (
    file_sha256,
    load_model_checkpoint,
    save_model_checkpoint,
    write_json,
)
from robustness_benchmark.core.training import (
    BETARCE_GENERATION_ENSEMBLE_PROTOCOL,
    BETARCE_GENERATION_ENSEMBLE_SIZE,
    make_betarce_generation_model,
)
from robustness_benchmark.evaluation.aggregate import (
    aggregate_survival,
    generation_metrics,
    normalize_survival_schema,
)
from robustness_benchmark.evaluation.metrics import evaluate_survival
from robustness_benchmark.methods.apas_calibration import (
    apas_method_config,
    load_or_create_apas_calibration,
    summarize_apas_calibration,
)
from robustness_benchmark.methods.configuration import (
    LOCKED_CONFIGS,
    METHOD_CHOICES,
    METHODS,
    TUNING_GRIDS,
)
from robustness_benchmark.methods.registry import generate_counterfactuals, make_task
from robustness_benchmark.methods.robx import (
    tau_from_stability_quantile,
    training_target_stability_scores,
)

BENCHMARK_PROTOCOL_VERSION = "full_benchmark_v9_split_50_10_10_30"
TUNING_PROTOCOL_VERSION = "validation_v2_separate_validity_roar_paper"
ADAPTIVE_ROBX_METHODS = ("robx_balanced", "robx_robust")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bank", type=Path, default=Path("artifacts/model_bank_breast_cancer")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/full_breast_cancer")
    )
    parser.add_argument(
        "--methods", nargs="+", choices=METHOD_CHOICES, default=list(METHODS)
    )
    parser.add_argument("--n-factuals", type=int, default=50)
    parser.add_argument("--tuning-factuals", type=int, default=25)
    parser.add_argument("--base-seeds", nargs="+", type=int)
    parser.add_argument("--apas-calibration-replicates", type=int, default=10)
    parser.add_argument("--apas-update-epochs", type=int, default=1)
    parser.add_argument("--apas-update-batch-size", type=int, default=8)
    parser.add_argument("--apas-update-learning-rate", type=float, default=1e-3)
    parser.add_argument("--apas-update-weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--tuning-base-seed",
        type=int,
        help="Bank base seed used for hyperparameter tuning "
        "(default: first seed in the bank manifest)",
    )
    parser.add_argument(
        "--frozen-tuning",
        type=Path,
        help=(
            "Reuse validated Wachter/ROAR-LIME settings from another tuning.json; "
            "fixed-method settings are taken from the current code"
        ),
    )
    return parser.parse_args()


def _score_configuration(
    method: str,
    config: dict[str, object],
    task,
    factuals: pd.DataFrame,
) -> dict[str, object]:
    _, generation = generate_counterfactuals(
        method, task, factuals, method_config=config
    )
    base_valid = [bool(row["base_valid"]) for row in generation]
    valid_generation = [row for row in generation if row["base_valid"]]
    return {
        "config": config,
        "base_validity": float(np.mean(base_valid)),
        "mean_l1_robust_scale": (
            float(np.mean([row["l1_robust_scale_mean"] for row in valid_generation]))
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
    cost = candidate["mean_l1_robust_scale"]
    numeric_cost = float(cost) if isinstance(cost, (int, float)) else float("inf")
    return (float(validity), -numeric_cost)


def tune_methods(
    bank: Path,
    data,
    methods: list[str],
    tuning_factuals: int,
    base_seed: int,
) -> tuple[dict[str, dict[str, object]], dict[str, object]]:
    configs: dict[str, dict[str, object]] = {}
    for method in methods:
        configs[method] = dict(LOCKED_CONFIGS.get(method, {}))
    base_id, base_model = load_base_model(bank, base_seed)
    factuals, _, _ = select_factuals(
        base_model, data.X_val, tuning_factuals, seed=base_seed + 70_000
    )
    task = make_task(base_model, data.X_train, data.y_train, seed=base_seed)
    report: dict[str, object] = {
        "protocol": {
            "base_seed": base_seed,
            "split": "validation",
            "factuals_n": len(factuals),
            "objective": (
                "maximize base-model validity, then minimize "
                "robust-scale L1; changed models are never consulted"
            ),
            "base_model_id": base_id,
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
            _score_configuration(method, config, task, factuals)
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


def tuning_config_is_current(method: str, config: object) -> bool:
    """Return whether a cached method config belongs to the current protocol."""

    if not isinstance(config, dict):
        return False
    grid = TUNING_GRIDS.get(method)
    if grid is not None:
        return config in grid
    return config == LOCKED_CONFIGS.get(method, {})


def resolve_robx_seed_configs(
    configs: dict[str, dict[str, object]],
    methods: list[str],
    base_model,
    X_train: pd.DataFrame,
    base_seed: int,
) -> dict[str, dict[str, object]]:
    """Resolve dataset- and model-dependent RobX stability thresholds."""

    resolved = {method: dict(configs[method]) for method in methods}
    adaptive = [method for method in methods if method in ADAPTIVE_ROBX_METHODS]
    if not adaptive:
        return resolved

    reference = resolved[adaptive[0]]
    variance = float(reference["variance"])
    stability_samples = int(reference["stability_samples"])

    def target_probability(values: np.ndarray) -> np.ndarray:
        return base_model.predict_proba(values).iloc[:, 1].to_numpy()

    scores, target_training_points = training_target_stability_scores(
        X_train.to_numpy(),
        target_probability,
        variance=variance,
        N=stability_samples,
        seed=base_seed + 81_000,
    )
    for method in adaptive:
        config = resolved[method]
        if float(config["variance"]) != variance:
            raise ValueError("Adaptive RobX methods must share one variance")
        if int(config["stability_samples"]) != stability_samples:
            raise ValueError("Adaptive RobX methods must share one sample count")
        tau, metadata = tau_from_stability_quantile(
            scores,
            quantile=float(config["tau_quantile"]),
            target_training_points=target_training_points,
        )
        config["tau"] = tau
        config["tau_calibration_metadata"] = metadata
    return resolved


def tuning_protocol_is_reusable(tuning: dict[str, object], base_seed: int) -> bool:
    """Allow tuning reuse across benchmark versions when its protocol is unchanged."""

    report = tuning.get("report", {})
    if not isinstance(report, dict):
        return False
    protocol = report.get("protocol", {})
    return (
        isinstance(protocol, dict)
        and tuning.get("tuning_protocol_version") == TUNING_PROTOCOL_VERSION
        and protocol.get("base_seed") == base_seed
    )


def configs_from_frozen_tuning(
    path: Path, methods: list[str]
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
            configs[method] = dict(LOCKED_CONFIGS.get(method, {}))
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


def _generation_ensemble(
    output: Path,
    bank: Path,
    bank_manifest: dict,
    base_id: str,
    data,
    base_seed: int,
    hidden_dim: tuple[int, ...],
    epochs: int,
    size: int = BETARCE_GENERATION_ENSEMBLE_SIZE,
    protocol: str = BETARCE_GENERATION_ENSEMBLE_PROTOCOL,
) -> list:
    checkpoints = bank_manifest.get("checkpoints", {})
    bank_member_ids = [
        f"{base_id}/betarce_generation/{protocol}/member_{member:02d}"
        for member in range(size)
    ]
    if all(member_id in checkpoints for member_id in bank_member_ids):
        return [
            load_model_checkpoint(bank / checkpoints[member_id]["path"])[0]
            for member_id in bank_member_ids
        ]
    directory = output / "generation_ensembles" / protocol / f"seed_{base_seed}"
    paths = [directory / f"member_{member:02d}.pt" for member in range(size)]
    models = []
    for member, path in enumerate(
        tqdm(paths, desc=f"seed {base_seed} · BetaRCE ensemble", leave=False)
    ):
        if path.is_file():
            models.append(load_model_checkpoint(path)[0])
            continue
        model = make_betarce_generation_model(
            data,
            base_seed=base_seed,
            member=member,
            hidden_dim=hidden_dim,
            epochs=epochs,
        )
        save_model_checkpoint(
            model,
            f"betarce_generation:{protocol}:seed_{base_seed}:member_{member:02d}",
            path,
            {
                "role": "generation_ensemble",
                "excluded_from_evaluation_changes": True,
                "protocol": protocol,
                "initialization_seed": base_seed,
                "bootstrap_sampling_seed": base_seed + 100_000 + member,
            },
        )
        models.append(model)
    return models


def run_matches_config(
    generation_path: Path,
    survival_path: Path,
    config: dict[str, object],
    expected_dataset_row_ids: list[int | str],
    expected_change_ids: set[str],
    expected_base_model_id: str,
    expected_bank_fingerprint: str,
) -> bool:
    if not generation_path.is_file() or not survival_path.is_file():
        return False
    generation = pd.read_parquet(generation_path)
    survival = pd.read_parquet(survival_path)
    generation_columns = {
        "method_config",
        "dataset_row_id",
        "benchmark_protocol_version",
        "base_model_id",
        "bank_fingerprint",
    }
    survival_columns = {
        "change_id",
        "benchmark_protocol_version",
        "base_model_id",
        "bank_fingerprint",
    }
    if not generation_columns <= set(generation.columns) or not survival_columns <= set(
        survival.columns
    ):
        return False
    actual = generation["method_config"].dropna().unique().tolist()
    actual_row_ids = generation["dataset_row_id"].tolist()
    return (
        actual == [json.dumps(config, sort_keys=True)]
        and actual_row_ids == expected_dataset_row_ids
        and set(survival["change_id"].unique()) == expected_change_ids
        and generation["base_model_id"].eq(expected_base_model_id).all()
        and survival["base_model_id"].eq(expected_base_model_id).all()
        and generation["bank_fingerprint"].eq(expected_bank_fingerprint).all()
        and survival["bank_fingerprint"].eq(expected_bank_fingerprint).all()
        and generation["benchmark_protocol_version"]
        .eq(BENCHMARK_PROTOCOL_VERSION)
        .all()
        and survival["benchmark_protocol_version"].eq(BENCHMARK_PROTOCOL_VERSION).all()
    )


def read_protocol_runs(
    paths: list[Path],
    *,
    methods: set[str] | None = None,
    base_model_ids: set[str] | None = None,
) -> pd.DataFrame:
    """Keep only current-protocol rows belonging to this invocation."""

    frames = []
    for path in paths:
        frame = pd.read_parquet(path)
        if "benchmark_protocol_version" not in frame.columns:
            continue
        frame = frame[frame["benchmark_protocol_version"] == BENCHMARK_PROTOCOL_VERSION]
        if methods is not None:
            frame = frame[frame["method"].isin(methods)]
        if base_model_ids is not None:
            frame = frame[frame["base_model_id"].isin(base_model_ids)]
        if len(frame):
            frames.append(frame)
    if not frames:
        raise RuntimeError(
            f"No runs match protocol version {BENCHMARK_PROTOCOL_VERSION!r}"
        )
    return pd.concat(frames, ignore_index=True)


def write_summary(
    output: Path,
    methods: list[str],
    configs: dict[str, dict[str, object]],
    base_model_ids: list[str],
    apas_calibration: list[dict[str, object]] | None = None,
) -> None:
    invocation_methods = set(methods)
    invocation_base_model_ids = set(base_model_ids)
    generation = read_protocol_runs(
        sorted(output.glob("runs/*/*/generation.parquet")),
        base_model_ids=invocation_base_model_ids,
    )
    survival = read_protocol_runs(
        sorted(output.glob("runs/*/*/survival.parquet")),
        base_model_ids=invocation_base_model_ids,
    )
    survival = normalize_survival_schema(survival)
    generation_runs = set(
        generation[["base_model_id", "method"]].itertuples(index=False, name=None)
    )
    survival_runs = set(
        survival[["base_model_id", "method"]].itertuples(index=False, name=None)
    )
    complete_methods = {
        method
        for method in set(generation["method"]).intersection(survival["method"])
        if all(
            (base_model_id, method) in generation_runs
            and (base_model_id, method) in survival_runs
            for base_model_id in invocation_base_model_ids
        )
    }
    if not invocation_methods.issubset(complete_methods):
        raise RuntimeError(
            "Invoked methods do not have complete current-protocol runs: "
            f"{sorted(invocation_methods - complete_methods)!r}"
        )
    generation = generation[generation["method"].isin(complete_methods)]
    survival = survival[survival["method"].isin(complete_methods)]
    expected_runs = {
        (base_model_id, method)
        for base_model_id in invocation_base_model_ids
        for method in complete_methods
    }
    for name, frame in (("generation", generation), ("survival", survival)):
        actual_runs = set(
            frame[["base_model_id", "method"]].itertuples(index=False, name=None)
        )
        if actual_runs != expected_runs:
            raise RuntimeError(
                f"{name} runs do not match this invocation: "
                f"missing={sorted(expected_runs - actual_runs)!r}, "
                f"unexpected={sorted(actual_runs - expected_runs)!r}"
            )
    generation.to_parquet(output / "generation.parquet", index=False)
    survival.to_parquet(output / "survival.parquet", index=False)
    run_configs = []
    for values, group in generation.groupby(["base_model_id", "method"], sort=True):
        if not isinstance(values, tuple) or len(values) != 2:
            raise RuntimeError(f"Unexpected run configuration key {values!r}")
        serialized = group["method_config"].dropna().unique().tolist()
        if len(serialized) != 1:
            raise RuntimeError(
                f"Expected one configuration for run {values!r}, got {serialized!r}"
            )
        run_configs.append(
            {
                "base_model_id": str(values[0]),
                "method": str(values[1]),
                "config": json.loads(str(serialized[0])),
            }
        )
    family_summary = aggregate_survival(survival)
    overall_survival_summary = []
    eligible_survival = survival[survival["conditional_eligible"]].copy()
    eligible_survival["survived"] = eligible_survival["conditional_survival"].eq(True)
    per_changed_model = (
        eligible_survival.groupby(["method", "base_model_id", "change_id"], sort=True)[
            "survived"
        ]
        .mean()
        .rename("survival")
        .reset_index()
    )
    for method, group in survival.groupby("method", sort=True):
        eligible = group["conditional_eligible"].astype(bool)
        base_valid = group["base_valid"].astype(bool)
        changed_valid = group["ce_valid"].eq(True)
        model_values = per_changed_model.loc[
            per_changed_model["method"] == method, "survival"
        ]
        overall_survival_summary.append(
            {
                "method": str(method),
                "changed_models_n": int(
                    group[["base_model_id", "change_id"]].drop_duplicates().shape[0]
                ),
                "eligible_changed_models_n": len(model_values),
                "eligible_pairs_n": int(eligible.sum()),
                "base_valid_pairs_n": int(base_valid.sum()),
                "base_validity_rate": float(base_valid.mean()),
                "changed_valid_pairs_n": int(changed_valid.sum()),
                "changed_validity_given_base_valid": (
                    float(changed_valid.sum() / base_valid.sum())
                    if base_valid.any()
                    else None
                ),
                "end_to_end_changed_validity": float(changed_valid.mean()),
                "pooled_conditional_survival": (
                    float(group.loc[eligible, "conditional_survival"].eq(True).mean())
                    if eligible.any()
                    else None
                ),
                "mean_changed_model_conditional_survival": (
                    float(model_values.mean()) if len(model_values) else None
                ),
                "median_changed_model_conditional_survival": (
                    float(model_values.median()) if len(model_values) else None
                ),
                "p10_changed_model_conditional_survival": (
                    float(model_values.quantile(0.1)) if len(model_values) else None
                ),
                "min_changed_model_conditional_survival": (
                    float(model_values.min()) if len(model_values) else None
                ),
            }
        )
    severity_summary = []
    for values, group in survival.groupby(["method", "change_severity"], sort=True):
        if not isinstance(values, tuple) or len(values) != 2:
            raise RuntimeError(f"Unexpected severity group key {values!r}")
        method, severity = str(values[0]), str(values[1])
        eligible = group["conditional_eligible"].astype(bool)
        base_valid = group["base_valid"].astype(bool)
        changed_valid = group["ce_valid"].eq(True)
        severity_summary.append(
            {
                "method": method,
                "severity": severity,
                "total_n": len(group),
                "base_valid_pairs_n": int(base_valid.sum()),
                "base_validity_rate": float(base_valid.mean()),
                "eligible_n": int(eligible.sum()),
                "conditional_survival": (
                    float(group.loc[eligible, "conditional_survival"].eq(True).mean())
                    if eligible.any()
                    else None
                ),
                "end_to_end_changed_validity": float(changed_valid.mean()),
            }
        )
    generation_summary = []
    for method, group in generation.groupby("method", sort=True):
        valid_cost = group.loc[group["base_valid"], "l1_robust_scale_mean"]
        certification_reported = (
            "method_certified" in group.columns
            and group["method_certified"].notna().any()
        )
        certified = (
            group["method_certified"].eq(True) if certification_reported else None
        )
        generation_summary.append(
            {
                "method": method,
                **generation_metrics(group),
                "mean_runtime_seconds": float(group["runtime_seconds"].mean()),
                "mean_l1_robust_scale": (
                    float(valid_cost.mean()) if len(valid_cost) else None
                ),
                "certified_n": int(certified.sum()) if certified is not None else None,
                "certified_coverage": (
                    float(certified.mean()) if certified is not None else None
                ),
            }
        )
    write_json(
        output / "summary.json",
        {
            "methods": sorted(str(method) for method in generation["method"].unique()),
            "invocation_methods": methods,
            "invocation_base_configs": configs,
            "run_configs": run_configs,
            "apas_calibration": apas_calibration or [],
            "generation": generation_summary,
            "survival_all_changes": overall_survival_summary,
            "survival_by_family": family_summary,
            "survival_by_severity": severity_summary,
            "benchmark_protocol_version": BENCHMARK_PROTOCOL_VERSION,
        },
    )


def main() -> None:
    args = parse_args()
    methods = list(args.methods)
    if "apas" in methods:
        if args.apas_calibration_replicates < 1:
            raise ValueError("APAS calibration replicates must be positive")
        if args.apas_update_epochs < 1:
            raise ValueError("APAS update epochs must be positive")
        if args.apas_update_batch_size < 1:
            raise ValueError("APAS update batch size must be positive")
        if args.apas_update_learning_rate <= 0:
            raise ValueError("APAS update learning rate must be positive")
        if args.apas_update_weight_decay < 0:
            raise ValueError("APAS update weight decay cannot be negative")
    args.output.mkdir(parents=True, exist_ok=True)
    bank_manifest = json.loads((args.bank / "manifest.json").read_text())
    bank_fingerprint = model_bank_fingerprint(args.bank)
    available_base_seeds = [int(seed) for seed in bank_manifest["base_seeds"]]
    # Banks built before split_seed was recorded all used the 2026 split.
    split_seed = int(bank_manifest.get("split_seed", 2026))
    data = load_dataset(
        str(bank_manifest["dataset"]),
        seed=split_seed,
        split_version=str(bank_manifest["split_version"]),
    )
    tuning_path = args.output / "tuning.json"
    expected_tuning_methods = methods
    if args.frozen_tuning is not None:
        configs, report = configs_from_frozen_tuning(args.frozen_tuning, methods)
        tuning = {
            "benchmark_protocol_version": BENCHMARK_PROTOCOL_VERSION,
            "tuning_protocol_version": TUNING_PROTOCOL_VERSION,
            "methods": expected_tuning_methods,
            "selected_configs": configs,
            "report": report,
        }
        write_json(tuning_path, tuning)
    else:
        tuning_base_seed = (
            int(args.tuning_base_seed)
            if args.tuning_base_seed is not None
            else available_base_seeds[0]
        )
        if tuning_base_seed not in available_base_seeds:
            raise ValueError(
                f"Tuning base seed {tuning_base_seed} is not in the model bank"
            )
        if tuning_path.is_file():
            tuning = json.loads(tuning_path.read_text())
            protocol_reusable = tuning_protocol_is_reusable(tuning, tuning_base_seed)
        else:
            tuning = {}
            protocol_reusable = False
        cached_configs = tuning.get("selected_configs", {})
        reusable_methods = {
            method
            for method in methods
            if protocol_reusable
            and tuning_config_is_current(method, cached_configs.get(method))
        }
        methods_to_tune = [
            method for method in methods if method not in reusable_methods
        ]
        if not methods_to_tune:
            configs = {method: dict(cached_configs[method]) for method in methods}
        else:
            fresh_configs, fresh_report = tune_methods(
                args.bank, data, methods_to_tune, args.tuning_factuals, tuning_base_seed
            )
            configs = {
                method: (
                    dict(cached_configs[method])
                    if method in reusable_methods
                    else fresh_configs[method]
                )
                for method in methods
            }
            cached_report = tuning.get("report", {}) if protocol_reusable else {}
            report = {
                "protocol": fresh_report["protocol"],
                "methods": {
                    **{
                        method: cached_report.get("methods", {}).get(method, {})
                        for method in reusable_methods
                    },
                    **fresh_report["methods"],
                },
            }
            tuning = {
                "benchmark_protocol_version": BENCHMARK_PROTOCOL_VERSION,
                "tuning_protocol_version": TUNING_PROTOCOL_VERSION,
                "methods": expected_tuning_methods,
                "selected_configs": configs,
                "report": report,
            }
            write_json(tuning_path, tuning)

    architecture = tuple(int(width) for width in bank_manifest["selected_architecture"])
    epochs = int(bank_manifest["epochs"])
    selected_base_seeds = args.base_seeds or available_base_seeds
    unknown_seeds = sorted(set(selected_base_seeds) - set(available_base_seeds))
    if unknown_seeds:
        raise ValueError(f"Base seeds are not in the model bank: {unknown_seeds}")
    selected_base_model_ids: list[str] = []
    apas_calibration_summaries: list[dict[str, object]] = []
    for base_seed in selected_base_seeds:
        base_id, base_model, changes = load_model_bank(args.bank, int(base_seed))
        selected_base_model_ids.append(base_id)
        factuals, factual_ids, row_ids = select_factuals(
            base_model,
            data.X_test,
            args.n_factuals,
            seed=int(base_seed) + 30_000,
        )
        task = make_task(base_model, data.X_train, data.y_train, seed=int(base_seed))
        seed_configs = resolve_robx_seed_configs(
            configs,
            methods,
            base_model,
            data.X_train,
            int(base_seed),
        )
        if "apas" in methods:
            calibration = load_or_create_apas_calibration(
                output=args.output,
                data=data,
                base_model=base_model,
                base_model_id=base_id,
                base_seed=int(base_seed),
                bank_fingerprint=bank_fingerprint,
                calibration_replicates=args.apas_calibration_replicates,
                update_epochs=args.apas_update_epochs,
                update_batch_size=args.apas_update_batch_size,
                update_learning_rate=args.apas_update_learning_rate,
                update_weight_decay=args.apas_update_weight_decay,
            )
            seed_configs["apas"] = apas_method_config(configs["apas"], calibration)
            apas_calibration_summaries.append(
                summarize_apas_calibration(
                    calibration, base_model_id=base_id, base_seed=int(base_seed)
                )
            )
        ensemble = (
            _generation_ensemble(
                args.output,
                args.bank,
                bank_manifest,
                base_id,
                data,
                int(base_seed),
                architecture,
                epochs,
                size=int(seed_configs["betarce"]["generation_ensemble_n"]),
                protocol=str(seed_configs["betarce"]["generation_model_space"]),
            )
            if "betarce" in methods
            else None
        )
        for method in methods:
            run_dir = args.output / "runs" / f"seed_{base_seed}" / method
            generation_path = run_dir / "generation.parquet"
            survival_path = run_dir / "survival.parquet"
            method_config = seed_configs[method]
            if run_matches_config(
                generation_path,
                survival_path,
                method_config,
                row_ids,
                {change.change_id for change in changes},
                base_id,
                bank_fingerprint,
            ):
                continue
            run_dir.mkdir(parents=True, exist_ok=True)
            counterfactuals, generation = generate_counterfactuals(
                method,
                task,
                factuals,
                factual_ids=factual_ids,
                dataset_row_ids=row_ids,
                generation_models=ensemble if method == "betarce" else None,
                method_config=method_config,
                progress_description=f"seed {base_seed} · {method}",
            )
            for row in generation:
                row["base_model_id"] = base_id
                row["method_config"] = json.dumps(method_config, sort_keys=True)
                row["benchmark_protocol_version"] = BENCHMARK_PROTOCOL_VERSION
                row["bank_fingerprint"] = bank_fingerprint
                if method == "apas":
                    row["apas_calibration_fingerprint"] = method_config[
                        "calibration_fingerprint"
                    ]
            survival = evaluate_survival(
                method,
                factuals,
                counterfactuals,
                changes,
                base_model,
                data.X_test,
                data.y_test,
                [bool(row["base_valid"]) for row in generation],
                factual_ids,
                row_ids,
                base_id,
            )
            for row in survival:
                row["benchmark_protocol_version"] = BENCHMARK_PROTOCOL_VERSION
                row["method_config"] = json.dumps(method_config, sort_keys=True)
                row["bank_fingerprint"] = bank_fingerprint
                if method == "apas":
                    row["apas_calibration_fingerprint"] = method_config[
                        "calibration_fingerprint"
                    ]
            pd.DataFrame(generation).to_parquet(generation_path, index=False)
            pd.DataFrame(survival).to_parquet(survival_path, index=False)
            counterfactuals.to_parquet(run_dir / "counterfactuals.parquet", index=False)
            factuals.to_parquet(run_dir / "factuals.parquet", index=False)
            print(
                json.dumps(
                    {
                        "seed": base_seed,
                        "method": method,
                        **generation_metrics(pd.DataFrame(generation)),
                    }
                ),
                flush=True,
            )
    write_summary(
        args.output,
        methods,
        configs,
        selected_base_model_ids,
        apas_calibration=apas_calibration_summaries,
    )


if __name__ == "__main__":
    main()
