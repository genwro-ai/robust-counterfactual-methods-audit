"""Installation check: a small MLP, one change per family, a few factuals."""

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from robustness_benchmark.benchmark.run import run_method
from robustness_benchmark.core.classifiers import MODEL_CLASSES
from robustness_benchmark.core.data import load_dataset
from robustness_benchmark.core.factuals import select_factuals
from robustness_benchmark.core.fitting import fit_spec
from robustness_benchmark.core.model_bank import (
    base_model_id,
    change_checkpoint_id,
    ensemble_member_id,
    register_checkpoint,
)
from robustness_benchmark.core.model_changes import (
    BETARCE_GENERATION_ENSEMBLE_PROTOCOL,
    make_betarce_generation_ensemble,
    make_changed_models,
)
from robustness_benchmark.core.model_specs import MLPSpec
from robustness_benchmark.core.protocol import EVALUATION_FACTUALS_SEED_OFFSET
from robustness_benchmark.core.provenance import row_id_value, write_json
from robustness_benchmark.evaluation.aggregate import (
    change_survival_summary,
    generation_summary,
)
from robustness_benchmark.methods.configuration import (
    METHODS,
    ROBX_METHODS,
    default_config,
    unsupported_methods,
)
from robustness_benchmark.methods.registry import make_task

# APΔS, RNCE and RobX need per-base-model calibration, which only
# full_benchmark performs; ROAR needs a linear model. Tuned methods use their
# grid's first setting.
CALIBRATED_METHODS = {"apas", "rnce", *ROBX_METHODS}
SMOKE_METHODS = tuple(
    method
    for method in METHODS
    if method not in CALIBRATED_METHODS
    and not unsupported_methods([method], MODEL_CLASSES["mlp"].capabilities)
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="breast_cancer")
    parser.add_argument(
        "--methods", nargs="+", choices=SMOKE_METHODS, default=["kdtree", "wachter"]
    )
    parser.add_argument("--n-factuals", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=Path("artifacts/smoke"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output
    data = load_dataset(args.dataset, seed=args.seed)
    spec = MLPSpec(hidden_dim=(16, 16), epochs=args.epochs)
    model = fit_spec(
        spec, data.X_train, data.y_train, data.X_val, data.y_val, seed=args.seed
    )
    base_id = base_model_id(args.dataset, spec, args.seed)
    factuals, factual_ids, row_ids = select_factuals(
        model,
        data.X_test,
        args.n_factuals,
        seed=args.seed + EVALUATION_FACTUALS_SEED_OFFSET,
    )
    changes = make_changed_models(
        data, model, base_seed=args.seed, spec=spec, profile="smoke"
    )
    ensemble = (
        make_betarce_generation_ensemble(data, base_seed=args.seed, spec=spec)
        if "betarce" in args.methods
        else []
    )
    task = make_task(
        model, data.X_train, data.y_train, seed=args.seed, schema=data.feature_schema
    )

    generation, survival, counterfactuals = [], [], []
    for method in args.methods:
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
            method_config=default_config(method),
            generation_models=ensemble or None,
        )
        generation.extend(run.generation)
        survival.extend(run.survival)
        counterfactuals.append(
            run.counterfactuals.assign(
                method=method,
                factual_id=factual_ids,
                dataset_row_id=np.asarray(row_ids, dtype=object),
                base_model_id=base_id,
            )
        )

    checkpoints: dict[str, dict[str, object]] = {}
    models = output / "models"
    register_checkpoint(
        checkpoints,
        base_id,
        model,
        base_id,
        models / "base.pt",
        output,
        {"role": "base", "dataset": args.dataset, "training_seed": args.seed},
    )
    for position, change in enumerate(changes):
        register_checkpoint(
            checkpoints,
            change_checkpoint_id(base_id, change.change_id),
            change.model,
            change.change_id,
            models / f"change_{position:02d}_{change.change_family}.pt",
            output,
            {"role": "changed", "base_model_id": base_id, **change.metadata},
        )
    protocol = BETARCE_GENERATION_ENSEMBLE_PROTOCOL
    for member, generation_model in enumerate(ensemble):
        member_id = ensemble_member_id(base_id, protocol, member)
        register_checkpoint(
            checkpoints,
            member_id,
            generation_model,
            member_id,
            models / "betarce_generation" / protocol / f"member_{member:02d}.pt",
            output,
            {"role": "generation_ensemble", "member": member, "protocol": protocol},
        )

    generation_frame = pd.DataFrame(generation)
    generation_frame.to_parquet(output / "generation.parquet", index=False)
    survival_frame = pd.DataFrame(survival)
    survival_frame.to_parquet(output / "survival.parquet", index=False)
    pd.concat(counterfactuals, ignore_index=True).to_parquet(
        output / "counterfactuals.parquet", index=False
    )
    factuals.assign(factual_id=factual_ids, dataset_row_id=row_ids).to_parquet(
        output / "factuals.parquet", index=False
    )
    created_at = datetime.now(UTC).isoformat()
    summary = {
        "created_at": created_at,
        "dataset": args.dataset,
        "seed": args.seed,
        "epochs": args.epochs,
        "methods": args.methods,
        "requested_n_factuals": args.n_factuals,
        "actual_n_factuals": len(factuals),
        "base_test_accuracy": model.evaluate(data.X_test, data.y_test),
        "generation": generation_summary(generation_frame),
        "survival": change_survival_summary(survival_frame),
    }
    write_json(output / "summary.json", summary)
    write_json(
        output / "manifest.json",
        {
            "created_at": created_at,
            "command": [sys.executable, "-m", "robustness_benchmark.cli.smoke"]
            + sys.argv[1:],
            "dataset": args.dataset,
            "dataset_source": data.source,
            "split_version": data.split_version,
            "favorable_source_label": data.favorable_source_label,
            "base_model_id": base_id,
            "changes": [
                {
                    "change_id": change.change_id,
                    "family": change.change_family,
                    "level": change.change_level,
                    "metadata": change.metadata,
                }
                for change in changes
            ],
            "model_checkpoints": checkpoints,
            "split_row_ids": {
                name: [row_id_value(value) for value in frame.index]
                for name, frame in {
                    "train": data.X_train,
                    "update": data.X_update,
                    "validation": data.X_val,
                    "test": data.X_test,
                }.items()
            },
        },
    )
    print(
        json.dumps(
            {
                "output": str(output.resolve()),
                "base_test_accuracy": summary["base_test_accuracy"],
                "generation_coverage": {
                    row["method"]: row["generation_coverage"]
                    for row in summary["generation"]
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
