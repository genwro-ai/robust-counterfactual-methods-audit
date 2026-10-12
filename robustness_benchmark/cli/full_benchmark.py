"""Run every method on a model bank and evaluate it on the changed models."""

import argparse
import json
from pathlib import Path

import pandas as pd

from robustness_benchmark.benchmark.run import run_method
from robustness_benchmark.benchmark.store import (
    BENCHMARK_PROTOCOL_VERSION,
    run_directory,
    run_is_current,
    write_run,
)
from robustness_benchmark.benchmark.summary import write_summary
from robustness_benchmark.benchmark.tuning import resolve_configs
from robustness_benchmark.core.classifiers import MODEL_CLASSES
from robustness_benchmark.core.data import load_dataset
from robustness_benchmark.core.factuals import select_factuals
from robustness_benchmark.core.model_bank import (
    load_generation_ensemble,
    load_model_bank,
    model_bank_fingerprint,
)
from robustness_benchmark.core.model_specs import spec_from_manifest
from robustness_benchmark.core.protocol import EVALUATION_FACTUALS_SEED_OFFSET
from robustness_benchmark.evaluation.aggregate import generation_metrics
from robustness_benchmark.evaluation.plausibility import PlausibilityReference
from robustness_benchmark.methods.calibration import (
    RNCE_UPDATE_CALIBRATION,
    apas_update_calibration,
    calibrate_seed_configs,
)
from robustness_benchmark.methods.configuration import METHODS, unsupported_methods
from robustness_benchmark.methods.registry import make_task


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bank", type=Path, default=Path("artifacts/model_bank_breast_cancer_mlp")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/full_breast_cancer_mlp")
    )
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
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
            "Reuse validated Wachter/ROAR-LIME/ROAR settings from another "
            "tuning.json; fixed-method settings are taken from the current code"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads((args.bank / "manifest.json").read_text())
    spec = spec_from_manifest(manifest)
    skipped_methods = unsupported_methods(
        args.methods, MODEL_CLASSES[spec.model_type].capabilities
    )
    for method, reason in skipped_methods.items():
        print(f"Skipping {method} on a {spec.model_type} bank: it {reason}", flush=True)
    methods = [method for method in args.methods if method not in skipped_methods]
    if not methods:
        raise ValueError(f"No requested method supports a {spec.model_type} model")
    calibration_protocols = {
        "apas": apas_update_calibration(
            args.apas_calibration_replicates,
            args.apas_update_epochs,
            args.apas_update_batch_size,
            args.apas_update_learning_rate,
            args.apas_update_weight_decay,
        ),
        "rnce": RNCE_UPDATE_CALIBRATION,
    }
    bank_seeds = [int(seed) for seed in manifest["base_seeds"]]
    base_seeds = args.base_seeds or bank_seeds
    tuning_base_seed = args.tuning_base_seed or bank_seeds[0]
    unknown = sorted({*base_seeds, tuning_base_seed} - set(bank_seeds))
    if unknown:
        raise ValueError(f"Base seeds are not in the model bank: {unknown}")

    args.output.mkdir(parents=True, exist_ok=True)
    bank_fingerprint = model_bank_fingerprint(args.bank)
    data = load_dataset(
        str(manifest["dataset"]),
        seed=int(manifest["split_seed"]),
        split_version=str(manifest["split_version"]),
    )
    configs = resolve_configs(
        output=args.output,
        bank=args.bank,
        data=data,
        methods=methods,
        tuning_factuals=args.tuning_factuals,
        tuning_base_seed=tuning_base_seed,
        model_type=spec.model_type,
        frozen_tuning=args.frozen_tuning,
    )

    base_model_ids: list[str] = []
    calibrations: list[dict[str, object]] = []
    for base_seed in base_seeds:
        base_id, base_model, changes = load_model_bank(args.bank, base_seed)
        base_model_ids.append(base_id)
        factuals, factual_ids, row_ids = select_factuals(
            base_model,
            data.X_test,
            args.n_factuals,
            seed=base_seed + EVALUATION_FACTUALS_SEED_OFFSET,
        )
        task = make_task(
            base_model,
            data.X_train,
            data.y_train,
            seed=base_seed,
            schema=data.feature_schema,
        )
        plausibility = PlausibilityReference(
            data.X_train, base_model, schema=data.feature_schema
        )
        # Radii and thresholds calibrated on this base model only.
        seed_configs, seed_calibrations = calibrate_seed_configs(
            configs,
            methods,
            protocols=calibration_protocols,
            output=args.output,
            data=data,
            base_model=base_model,
            base_model_id=base_id,
            base_seed=base_seed,
            bank_fingerprint=bank_fingerprint,
        )
        calibrations.extend(seed_calibrations)
        ensemble = (
            load_generation_ensemble(
                bank=args.bank,
                base_id=base_id,
                data=data,
                base_seed=base_seed,
                spec=spec,
                cache=args.output / "generation_ensembles",
                size=int(seed_configs["betarce"]["generation_ensemble_n"]),
                protocol=str(seed_configs["betarce"]["generation_model_space"]),
            )
            if "betarce" in methods
            else None
        )
        change_ids = {change.change_id for change in changes}
        for method in methods:
            directory = run_directory(args.output, base_seed, method)
            method_config = seed_configs[method]
            if run_is_current(
                directory,
                method_config,
                row_ids,
                change_ids,
                base_id,
                bank_fingerprint,
            ):
                continue
            provenance = {
                "benchmark_protocol_version": BENCHMARK_PROTOCOL_VERSION,
                "bank_fingerprint": bank_fingerprint,
            }
            if "calibration_fingerprint" in method_config:
                provenance["calibration_fingerprint"] = method_config[
                    "calibration_fingerprint"
                ]
            run = run_method(
                method,
                task=task,
                factuals=factuals,
                factual_ids=factual_ids,
                dataset_row_ids=row_ids,
                changes=changes,
                base_model_id=base_id,
                X_reference=data.X_test,
                y_reference=data.y_test,
                method_config=method_config,
                generation_models=ensemble if method == "betarce" else None,
                plausibility=plausibility,
                provenance=provenance,
                progress_description=f"seed {base_seed} · {method}",
            )
            write_run(directory, run, factuals)
            metrics = generation_metrics(pd.DataFrame(run.generation))
            print(
                json.dumps({"seed": base_seed, "method": method, **metrics}),
                flush=True,
            )
    write_summary(
        args.output,
        methods,
        configs,
        base_model_ids,
        calibrations=calibrations,
        skipped_methods=skipped_methods,
        model_type=spec.model_type,
    )


if __name__ == "__main__":
    main()
