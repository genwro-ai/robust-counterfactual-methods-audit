import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from robustness_benchmark.core.data import load_dataset, select_adverse_indices
from robustness_benchmark.core.provenance import (
    environment_info,
    file_sha256,
    git_info,
    row_id_value,
    save_model_checkpoint,
    write_json,
)
from robustness_benchmark.core.training import (
    BETARCE_GENERATION_ENSEMBLE_PROTOCOL,
    fit_model,
    make_betarce_generation_ensemble,
    make_changed_models,
)
from robustness_benchmark.evaluation.metrics import evaluate_survival
from robustness_benchmark.evaluation.summaries import generation_summary, survival_summary
from robustness_benchmark.methods.registry import generate_counterfactuals, make_task


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="breast_cancer")
    parser.add_argument("--methods", nargs="+", default=["kdtree", "wachter"])
    parser.add_argument("--n-factuals", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=Path("artifacts/smoke"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_dataset(args.dataset, seed=args.seed)
    model = fit_model(
        data.X_train,
        data.y_train,
        data.X_val,
        data.y_val,
        seed=args.seed,
        epochs=args.epochs,
    )
    test_probability = model.predict(data.X_test).iloc[:, 0]
    test_probability.index = data.X_test.index
    adverse_indices = select_adverse_indices(
        test_probability,
        requested=args.n_factuals,
        seed=args.seed + 30_000,
    )
    factuals = data.X_test.loc[adverse_indices].reset_index(drop=True)
    factual_ids = list(range(len(factuals)))
    dataset_row_ids: list[int | str] = [
        row_id_value(value) for value in adverse_indices.tolist()
    ]

    changes = make_changed_models(
        data,
        model,
        base_seed=args.seed,
        epochs=args.epochs,
        hidden_dim=(16, 16),
        profile="smoke",
    )
    generation_models = (
        make_betarce_generation_ensemble(
            data,
            base_seed=args.seed,
            hidden_dim=(16, 16),
            epochs=args.epochs,
        )
        if "betarce" in args.methods
        else None
    )
    base_model_id = f"{args.dataset}:mlp_16_16:seed_{args.seed}"
    generation_rows: list[dict[str, object]] = []
    survival_rows: list[dict[str, object]] = []
    ce_tables: list[pd.DataFrame] = []
    task = make_task(model, data.X_train, data.y_train, seed=args.seed)

    for method in args.methods:
        counterfactuals, method_generation = generate_counterfactuals(
            method,
            task,
            factuals,
            factual_ids=factual_ids,
            dataset_row_ids=dataset_row_ids,
            generation_models=generation_models,
        )
        for row in method_generation:
            row["base_model_id"] = base_model_id
        generation_rows.extend(method_generation)
        base_valid = [bool(row["base_valid"]) for row in method_generation]
        survival_rows.extend(
            evaluate_survival(
                method,
                factuals,
                counterfactuals,
                changes,
                model,
                data.X_test,
                data.y_test,
                base_valid,
                factual_ids,
                dataset_row_ids,
                base_model_id,
            )
        )
        table = counterfactuals.copy()
        table.insert(0, "base_model_id", base_model_id)
        table.insert(0, "dataset_row_id", np.asarray(dataset_row_ids, dtype=object))
        table.insert(0, "factual_id", np.arange(len(table)))
        table.insert(0, "method", method)
        ce_tables.append(table)

    args.output.mkdir(parents=True, exist_ok=True)
    model_directory = args.output / "models"
    base_checkpoint = model_directory / "base.pt"
    save_model_checkpoint(
        model,
        base_model_id,
        base_checkpoint,
        {"role": "base", "dataset": args.dataset, "training_seed": args.seed},
    )
    checkpoint_paths: dict[str, dict[str, object]] = {
        base_model_id: {
            "path": "models/base.pt",
            "sha256": file_sha256(base_checkpoint),
            "bytes": base_checkpoint.stat().st_size,
        }
    }
    for position, change in enumerate(changes):
        relative_path = (
            Path("models") / f"change_{position:02d}_{change.change_family}.pt"
        )
        checkpoint_path = args.output / relative_path
        save_model_checkpoint(
            change.model,
            change.change_id,
            checkpoint_path,
            {
                "role": "changed",
                "base_model_id": base_model_id,
                "family": change.change_family,
                "level": change.change_level,
                **change.metadata,
            },
        )
        checkpoint_paths[change.change_id] = {
            "path": str(relative_path),
            "sha256": file_sha256(checkpoint_path),
            "bytes": checkpoint_path.stat().st_size,
        }
    if generation_models is not None:
        for member, generation_model in enumerate(generation_models):
            ensemble_id = (
                f"{base_model_id}/betarce_generation/"
                f"{BETARCE_GENERATION_ENSEMBLE_PROTOCOL}/member_{member:02d}"
            )
            relative_path = (
                Path("models")
                / "betarce_generation"
                / BETARCE_GENERATION_ENSEMBLE_PROTOCOL
                / f"member_{member:02d}.pt"
            )
            checkpoint_path = args.output / relative_path
            save_model_checkpoint(
                generation_model,
                ensemble_id,
                checkpoint_path,
                {
                    "role": "generation_ensemble",
                    "method": "betarce",
                    "excluded_from_evaluation_changes": True,
                    "member": member,
                    "protocol": BETARCE_GENERATION_ENSEMBLE_PROTOCOL,
                    "initialization_seed": args.seed,
                },
            )
            checkpoint_paths[ensemble_id] = {
                "path": str(relative_path),
                "sha256": file_sha256(checkpoint_path),
                "bytes": checkpoint_path.stat().st_size,
            }

    pd.DataFrame(generation_rows).to_parquet(
        args.output / "generation.parquet", index=False
    )
    pd.DataFrame(survival_rows).to_parquet(
        args.output / "survival.parquet", index=False
    )
    pd.concat(ce_tables, ignore_index=True).to_parquet(
        args.output / "counterfactuals.parquet", index=False
    )
    factual_table = factuals.copy()
    factual_table.insert(0, "base_model_id", base_model_id)
    factual_table.insert(0, "dataset_row_id", np.asarray(dataset_row_ids, dtype=object))
    factual_table.insert(0, "factual_id", np.arange(len(factual_table)))
    factual_table.to_parquet(args.output / "factuals.parquet", index=False)

    generation = pd.DataFrame(generation_rows)
    survival = pd.DataFrame(survival_rows)
    summary = {
        "created_at": datetime.now(UTC).isoformat(),
        "dataset": args.dataset,
        "seed": args.seed,
        "epochs": args.epochs,
        "requested_n_factuals": args.n_factuals,
        "actual_n_factuals": len(factuals),
        "available_adverse_n": int((test_probability < 0.5).sum()),
        "methods": args.methods,
        "base_test_accuracy": model.evaluate(data.X_test, data.y_test),
        "behavior_reference_split": "test",
        "generation": generation_summary(generation),
        "survival": survival_summary(survival),
    }
    write_json(args.output / "summary.json", summary)

    project_root = Path(__file__).resolve().parents[1]
    split_ids = {
        name: [row_id_value(value) for value in frame.index.tolist()]
        for name, frame in {
            "train": data.X_train,
            "update": data.X_update,
            "validation": data.X_val,
            "test": data.X_test,
        }.items()
    }
    manifest = {
        "created_at": summary["created_at"],
        "command": [sys.executable, "-m", "robustness_benchmark.cli.smoke", *sys.argv[1:]],
        "configuration": {
            "dataset": args.dataset,
            "dataset_source": data.source,
            "split_version": data.split_version,
            "seed": args.seed,
            "epochs": args.epochs,
            "methods": args.methods,
            "requested_n_factuals": args.n_factuals,
            "adverse_label": data.adverse_label,
            "favorable_label": data.favorable_label,
            "favorable_source_label": data.favorable_source_label,
            "behavior_reference_split": "test",
            "behavior_metrics": ["hard_disagreement", "probability_mae"],
            "betarce_generation_ensemble_n": (
                len(generation_models) if generation_models is not None else 0
            ),
        },
        "base_model_id": base_model_id,
        "changes": [
            {
                "change_id": change.change_id,
                "family": change.change_family,
                "level": change.change_level,
                "metadata": change.metadata,
            }
            for change in changes
        ],
        "model_checkpoints": checkpoint_paths,
        "selected_factuals": [
            {"factual_id": factual_id, "dataset_row_id": dataset_row_id}
            for factual_id, dataset_row_id in zip(
                factual_ids, dataset_row_ids, strict=True
            )
        ],
        "split_row_ids": split_ids,
        "environment": environment_info(),
        "project_git": git_info(project_root),
    }
    write_json(args.output / "manifest.json", manifest)
    console_summary = {
        "output": str(args.output.resolve()),
        "base_test_accuracy": summary["base_test_accuracy"],
        "actual_n_factuals": summary["actual_n_factuals"],
        "available_adverse_n": summary["available_adverse_n"],
        "generation_coverage": {
            row["method"]: row["generation_coverage"] for row in summary["generation"]
        },
    }
    print(json.dumps(console_summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
