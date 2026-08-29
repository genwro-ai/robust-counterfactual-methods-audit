import argparse
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import torch

from robustness_benchmark.core.data import DatasetSplit, load_dataset
from robustness_benchmark.core.provenance import (
    file_sha256,
    save_model_checkpoint,
    write_json,
)
from robustness_benchmark.core.training import (
    ARCHITECTURE_CANDIDATES,
    ARCHITECTURE_DEV_SEEDS,
    BETARCE_GENERATION_ENSEMBLE_PROTOCOL,
    BETARCE_GENERATION_ENSEMBLE_SIZE,
    FULL_CHANGE_FAMILIES,
    FULL_VARIANTS_PER_FAMILY,
    TRAINING_CONFIGS,
    architecture_audit,
    fit_model,
    make_betarce_generation_ensemble,
    make_changed_models,
)
from robustness_benchmark.evaluation.metrics import assess_model_changes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="breast_cancer")
    parser.add_argument("--split-version", default="behavior_v3")
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--base-seeds", type=int, nargs="+", default=[2026, 2027, 2028])
    parser.add_argument("--base-architecture", type=int, nargs="+", default=[32, 32])
    parser.add_argument("--maximum-accuracy-drop", type=float, default=0.03)
    parser.add_argument("--include-betarce-ensemble", action="store_true")
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Base-model banks to train concurrently (default: 2)",
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/model_bank"))
    return parser.parse_args()


def _configure_training_worker() -> None:
    """Prevent each training process from spawning a large CPU thread pool."""

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # PyTorch permits setting this only once in a process.
        pass


def _prepare_base_seed(
    *,
    data: DatasetSplit,
    dataset: str,
    split_version: str,
    epochs: int,
    base_seed: int,
    selected_architecture: tuple[int, ...],
    maximum_accuracy_drop: float,
    include_betarce_ensemble: bool,
    output: Path,
) -> tuple[list[dict[str, object]], dict[str, dict[str, object]]]:
    """Train and save one base model and its complete, independent change bank."""

    base_model = fit_model(
        data.X_train,
        data.y_train,
        data.X_val,
        data.y_val,
        seed=base_seed,
        hidden_dim=selected_architecture,
        epochs=epochs,
    )
    base_id = (
        f"{dataset}:mlp_{'_'.join(map(str, selected_architecture))}:seed_{base_seed}"
    )
    base_path = output / "models" / f"seed_{base_seed}" / "base.pt"
    save_model_checkpoint(
        base_model,
        base_id,
        base_path,
        {
            "role": "base",
            "split_version": split_version,
            "training_seed": base_seed,
            "hidden_dim": list(selected_architecture),
        },
    )
    checkpoints = {
        base_id: {
            "path": str(base_path.relative_to(output)),
            "sha256": file_sha256(base_path),
        }
    }

    changes = make_changed_models(
        data,
        base_model,
        base_seed=base_seed,
        epochs=epochs,
        hidden_dim=selected_architecture,
        profile="full",
    )
    assessments = assess_model_changes(
        changes,
        base_model,
        data.X_test,
        data.y_test,
        maximum_accuracy_drop=maximum_accuracy_drop,
    )
    catalog_rows: list[dict[str, object]] = []
    for index, (change, assessment) in enumerate(
        zip(changes, assessments, strict=True)
    ):
        checkpoint_path = (
            output
            / "models"
            / f"seed_{base_seed}"
            / f"change_{index:03d}_{change.change_family}.pt"
        )
        save_model_checkpoint(
            change.model,
            change.change_id,
            checkpoint_path,
            {
                "role": "changed",
                "base_model_id": base_id,
                **assessment,
            },
        )
        checkpoints[f"{base_id}/{change.change_id}"] = {
            "path": str(checkpoint_path.relative_to(output)),
            "sha256": file_sha256(checkpoint_path),
        }
        catalog_rows.append({"base_model_id": base_id, **assessment})

    if include_betarce_ensemble:
        generation_models = make_betarce_generation_ensemble(
            data,
            base_seed=base_seed,
            hidden_dim=selected_architecture,
            epochs=epochs,
        )
        for member, generation_model in enumerate(generation_models):
            ensemble_id = (
                f"{base_id}/betarce_generation/"
                f"{BETARCE_GENERATION_ENSEMBLE_PROTOCOL}/member_{member:02d}"
            )
            ensemble_path = (
                output
                / "models"
                / f"seed_{base_seed}"
                / "betarce_generation"
                / BETARCE_GENERATION_ENSEMBLE_PROTOCOL
                / f"member_{member:02d}.pt"
            )
            save_model_checkpoint(
                generation_model,
                ensemble_id,
                ensemble_path,
                {
                    "role": "generation_ensemble",
                    "method": "betarce",
                    "excluded_from_evaluation_changes": True,
                    "member": member,
                    "protocol": BETARCE_GENERATION_ENSEMBLE_PROTOCOL,
                    "initialization_seed": base_seed,
                },
            )
            checkpoints[ensemble_id] = {
                "path": str(ensemble_path.relative_to(output)),
                "sha256": file_sha256(ensemble_path),
            }
    return catalog_rows, checkpoints


def prepare_model_bank(
    *,
    dataset: str,
    split_version: str,
    split_seed: int = 2026,
    epochs: int,
    base_seeds: list[int],
    base_architecture: tuple[int, ...] = (32, 32),
    maximum_accuracy_drop: float,
    include_betarce_ensemble: bool = False,
    workers: int = 2,
    output: Path,
) -> dict[str, object]:
    if workers < 1:
        raise ValueError("workers must be positive")
    data = load_dataset(dataset, seed=split_seed, split_version=split_version)
    audit_recommendation, audit_rows = architecture_audit(data, epochs=epochs)
    if not base_architecture or any(width <= 0 for width in base_architecture):
        raise ValueError("base_architecture must contain positive layer widths")
    selected_architecture = tuple(base_architecture)
    for row in audit_rows:
        row["audit_recommended"] = row["selected"]
        hidden_values = row["hidden_dim"]
        if not isinstance(hidden_values, list):
            raise TypeError("hidden_dim audit field must be a list")
        row["selected"] = tuple(hidden_values) == selected_architecture
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(audit_rows).to_parquet(
        output / "architecture_audit.parquet", index=False
    )
    write_json(
        output / "architecture_audit.json",
        {
            "candidate_architectures": [
                list(candidate) for candidate in ARCHITECTURE_CANDIDATES
            ],
            "dev_seeds": list(ARCHITECTURE_DEV_SEEDS),
            "audit_rule": "smallest model within 0.01 of best mean validation balanced accuracy",
            "audit_recommendation": list(audit_recommendation),
            "selection_rule": "prespecified architecture fixed before the benchmark",
            "selected_architecture": list(selected_architecture),
            "runs": audit_rows,
        },
    )

    worker_arguments = [
        {
            "data": data,
            "dataset": dataset,
            "split_version": split_version,
            "epochs": epochs,
            "base_seed": base_seed,
            "selected_architecture": selected_architecture,
            "maximum_accuracy_drop": maximum_accuracy_drop,
            "include_betarce_ensemble": include_betarce_ensemble,
            "output": output,
        }
        for base_seed in base_seeds
    ]
    if workers == 1 or len(worker_arguments) == 1:
        base_results = [
            _prepare_base_seed(**arguments) for arguments in worker_arguments
        ]
    else:
        maximum_workers = min(workers, len(worker_arguments))
        with ProcessPoolExecutor(
            max_workers=maximum_workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_configure_training_worker,
        ) as executor:
            futures = [
                executor.submit(_prepare_base_seed, **arguments)
                for arguments in worker_arguments
            ]
            base_results = [future.result() for future in futures]

    catalog_rows = [row for rows, _ in base_results for row in rows]
    checkpoints = {
        identifier: checkpoint
        for _, base_checkpoints in base_results
        for identifier, checkpoint in base_checkpoints.items()
    }

    catalog = pd.DataFrame(catalog_rows)
    catalog["metadata"] = catalog["metadata"].map(
        lambda value: json.dumps(value, sort_keys=True)
    )
    catalog.to_parquet(output / "model_change_catalog.parquet", index=False)
    summary: dict[str, object] = {
        "created_at": datetime.now(UTC).isoformat(),
        "dataset": dataset,
        "split_version": split_version,
        "split_seed": split_seed,
        "epochs": epochs,
        "base_seeds": base_seeds,
        "selected_architecture": list(selected_architecture),
        "change_families": list(FULL_CHANGE_FAMILIES),
        "candidate_models_per_family": FULL_VARIANTS_PER_FAMILY,
        "candidate_models_per_base": (
            FULL_VARIANTS_PER_FAMILY * len(FULL_CHANGE_FAMILIES)
        ),
        "training_config_grid_version": "sgd_2x3_lr_wd_v2",
        "training_config_grid": [
            {
                "optimizer_name": optimizer_name,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
            }
            for optimizer_name, learning_rate, weight_decay in TRAINING_CONFIGS
        ],
        "workers": min(workers, len(base_seeds)),
        "generation_ensemble_included": include_betarce_ensemble,
        "betarce_generation_models_per_base": (
            BETARCE_GENERATION_ENSEMBLE_SIZE if include_betarce_ensemble else 0
        ),
        "quality_filter": {"maximum_accuracy_drop": maximum_accuracy_drop},
        "quality_pass_n": int(catalog["quality_pass"].sum()),
        "candidate_n": len(catalog),
        "checkpoints": checkpoints,
    }
    write_json(output / "manifest.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    summary = prepare_model_bank(
        dataset=args.dataset,
        split_version=args.split_version,
        split_seed=args.split_seed,
        epochs=args.epochs,
        base_seeds=args.base_seeds,
        base_architecture=tuple(args.base_architecture),
        maximum_accuracy_drop=args.maximum_accuracy_drop,
        include_betarce_ensemble=args.include_betarce_ensemble,
        workers=args.workers,
        output=args.output,
    )
    console_summary = {
        key: summary[key]
        for key in (
            "dataset",
            "split_version",
            "selected_architecture",
            "base_seeds",
            "candidate_models_per_base",
            "candidate_n",
            "quality_pass_n",
            "generation_ensemble_included",
            "betarce_generation_models_per_base",
        )
    }
    console_summary["output"] = str(args.output.resolve())
    print(json.dumps(console_summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
