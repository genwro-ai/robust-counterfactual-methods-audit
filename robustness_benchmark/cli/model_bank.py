"""Build the model bank of one dataset and model class."""

import argparse
import json
from pathlib import Path

from robustness_benchmark.core.classifiers import MODEL_TYPES
from robustness_benchmark.core.model_bank import build_model_bank
from robustness_benchmark.core.model_specs import MODEL_SPECS, MLPSpec, ModelSpec
from robustness_benchmark.core.protocol import (
    BASE_SEEDS,
    SPLIT_SEED,
    SPLIT_VERSION,
    bank_dir,
)

MLP_DEFAULTS = MLPSpec()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="breast_cancer")
    parser.add_argument("--model-type", choices=MODEL_TYPES, default="mlp")
    parser.add_argument("--split-version", default=SPLIT_VERSION)
    parser.add_argument("--split-seed", type=int, default=SPLIT_SEED)
    parser.add_argument(
        "--epochs",
        type=int,
        help=f"MLP training epochs (MLP only; default {MLP_DEFAULTS.epochs})",
    )
    parser.add_argument(
        "--base-architecture",
        type=int,
        nargs="+",
        help="Prespecified MLP hidden layers (MLP only; default "
        f"{' '.join(map(str, MLP_DEFAULTS.hidden_dim))})",
    )
    parser.add_argument("--base-seeds", type=int, nargs="+", default=list(BASE_SEEDS))
    parser.add_argument("--maximum-accuracy-drop", type=float, default=0.03)
    parser.add_argument("--include-betarce-ensemble", action="store_true")
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Base-model banks to train concurrently (default: 2)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Default: artifacts/model_bank_<dataset>_<model_type>",
    )
    return parser.parse_args()


def base_spec(args: argparse.Namespace) -> ModelSpec:
    """The prespecified MLP, or the grid seed of another model class."""

    if args.model_type != "mlp":
        if args.epochs is not None or args.base_architecture is not None:
            raise ValueError("--epochs and --base-architecture apply to the MLP only")
        return MODEL_SPECS[args.model_type]()
    overrides = {}
    if args.epochs is not None:
        overrides["epochs"] = args.epochs
    if args.base_architecture is not None:
        if any(width <= 0 for width in args.base_architecture):
            raise ValueError("--base-architecture needs positive layer widths")
        overrides["hidden_dim"] = tuple(args.base_architecture)
    return MLP_DEFAULTS.with_overrides(**overrides)


def main() -> None:
    args = parse_args()
    output = args.output or bank_dir(Path("artifacts"), args.dataset, args.model_type)
    manifest = build_model_bank(
        dataset=args.dataset,
        split_version=args.split_version,
        split_seed=args.split_seed,
        base_spec=base_spec(args),
        base_seeds=args.base_seeds,
        maximum_accuracy_drop=args.maximum_accuracy_drop,
        include_betarce_ensemble=args.include_betarce_ensemble,
        workers=args.workers,
        output=output,
    )
    summary = {
        key: manifest[key]
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
    summary["output"] = str(output.resolve())
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
