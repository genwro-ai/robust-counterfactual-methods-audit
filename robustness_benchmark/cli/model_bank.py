import argparse
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import torch

from robustness_benchmark.core.classifiers import MODEL_TYPES
from robustness_benchmark.core.data import DatasetSplit, load_dataset
from robustness_benchmark.core.model_specs import MODEL_SPECS, MLPSpec, ModelSpec
from robustness_benchmark.core.provenance import (
    file_sha256,
    save_model_checkpoint,
    write_json,
)
from robustness_benchmark.core.training import (
    BETARCE_GENERATION_ENSEMBLE_PROTOCOL,
    BETARCE_GENERATION_ENSEMBLE_SIZE,
    DISTINCT_VALIDATION_MAE,
    FULL_VARIANTS_PER_FAMILY,
    SELECTION_DEV_SEEDS,
    fit_spec,
    make_betarce_generation_ensemble,
    make_changed_models,
    select_base_spec,
)
from robustness_benchmark.evaluation.metrics import assess_model_changes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="breast_cancer")
    parser.add_argument("--model-type", choices=MODEL_TYPES, default="mlp")
    parser.add_argument("--split-version", default="behavior_v3")
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument(
        "--epochs", type=int, default=300, help="MLP training epochs (MLP only)"
    )
    parser.add_argument("--base-seeds", type=int, nargs="+", default=[2026, 2027, 2028])
    parser.add_argument(
        "--base-architecture",
        type=int,
        nargs="+",
        default=[32, 32],
        help="Prespecified MLP hidden layers (MLP only)",
    )
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
    base_seed: int,
    spec: ModelSpec,
    maximum_accuracy_drop: float,
    include_betarce_ensemble: bool,
    output: Path,
) -> tuple[list[dict[str, object]], dict[str, dict[str, object]]]:
    """Train and save one base model and its complete, independent change bank."""

    base_model = fit_spec(
        spec, data.X_train, data.y_train, data.X_val, data.y_val, seed=base_seed
    )
    base_id = f"{dataset}:{spec.id_fragment()}:seed_{base_seed}"
    base_path = output / "models" / f"seed_{base_seed}" / "base.pt"
    save_model_checkpoint(
        base_model,
        base_id,
        base_path,
        {
            "role": "base",
            "split_version": split_version,
            "training_seed": base_seed,
            "model_spec": spec.to_dict(),
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
        profile="full",
        spec=spec,
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
            data, base_seed=base_seed, spec=spec
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


def _select_spec(
    data: DatasetSplit,
    base: ModelSpec,
    *,
    output: Path,
    prespecified: ModelSpec | None = None,
) -> ModelSpec:
    """Run the a priori base-configuration selection and record it.

    LR and XGBoost use the selected configuration. The MLP keeps its
    prespecified architecture; its selection run is reported as an audit.
    """

    recommended, rows = select_base_spec(data, base)
    selected = prespecified or recommended
    pd.DataFrame(
        [{**row, "model_spec": json.dumps(row["model_spec"])} for row in rows]
    ).to_parquet(output / "base_config_selection.parquet", index=False)
    write_json(
        output / "base_config_selection.json",
        {
            "model_type": base.model_type,
            "dev_seeds": list(SELECTION_DEV_SEEDS),
            "selection_rule": "lowest mean validation log-loss over the dev seeds",
            "recommended": recommended.to_dict(),
            "selected": selected.to_dict(),
            "prespecified": prespecified is not None,
            "runs": rows,
        },
    )
    return selected


def prepare_model_bank(
    *,
    dataset: str,
    split_version: str,
    split_seed: int = 2026,
    model_type: str = "mlp",
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
    output.mkdir(parents=True, exist_ok=True)
    if model_type == "mlp":
        if not base_architecture or any(width <= 0 for width in base_architecture):
            raise ValueError("base_architecture must contain positive layer widths")
        base: ModelSpec = MLPSpec(hidden_dim=tuple(base_architecture), epochs=epochs)
        spec = _select_spec(data, base, output=output, prespecified=base)
    else:
        spec = _select_spec(data, MODEL_SPECS[model_type](), output=output)

    worker_arguments = [
        {
            "data": data,
            "dataset": dataset,
            "split_version": split_version,
            "base_seed": base_seed,
            "spec": spec,
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
        "model_type": spec.model_type,
        "model_spec": spec.to_dict(),
        "selected_architecture": (
            list(spec.hidden_dim) if isinstance(spec, MLPSpec) else None
        ),
        "change_families": list(spec.change_families),
        "candidate_models_per_family": FULL_VARIANTS_PER_FAMILY,
        "candidate_models_per_base": (
            FULL_VARIANTS_PER_FAMILY * len(spec.change_families)
        ),
        "configuration_candidates_version": "ordered_distinct_v1",
        "distinct_validation_mae": DISTINCT_VALIDATION_MAE,
        "training_config_candidates": [
            {"level": level, **overrides}
            for level, overrides in spec.training_config_variants()
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
        model_type=args.model_type,
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
            "model_type",
            "model_spec",
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
