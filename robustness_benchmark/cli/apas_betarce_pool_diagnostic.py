import argparse
import json
from pathlib import Path

import pandas as pd

from robustness_benchmark.core.data import load_dataset
from robustness_benchmark.core.model_bank import (
    load_model_bank,
    model_bank_fingerprint,
    select_factuals,
)
from robustness_benchmark.core.provenance import load_model_checkpoint, write_json
from robustness_benchmark.core.training import parameter_linf_distance
from robustness_benchmark.evaluation.aggregate import aggregate_survival, generation_metrics
from robustness_benchmark.evaluation.metrics import behavioral_distance, evaluate_survival
from robustness_benchmark.evaluation.summaries import (
    apas_generation_summary,
    seed_family_survival_summary,
)
from robustness_benchmark.methods.apas import APAS_IMPLEMENTATION
from robustness_benchmark.methods.apas_calibration import stable_hash
from robustness_benchmark.methods.configuration import LOCKED_CONFIGS
from robustness_benchmark.methods.registry import generate_counterfactuals, make_task

PROTOCOL_VERSION = "apas_betarce_pool_diagnostic_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", required=True, type=Path)
    parser.add_argument("--betarce-pool", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--n-factuals", type=int, default=5)
    parser.add_argument("--pool-members", type=int, default=10)
    parser.add_argument("--base-seeds", nargs="+", type=int)
    return parser.parse_args()


def _pool_calibration(
    *,
    pool: Path,
    data,
    base_model,
    base_model_id: str,
    base_seed: int,
    pool_members: int,
) -> dict[str, object]:
    paths = sorted((pool / f"seed_{base_seed}").glob("member_*.pt"))
    if len(paths) < pool_members:
        raise ValueError(
            f"Expected at least {pool_members} BetaRCE models for seed {base_seed}, "
            f"found {len(paths)}"
        )
    records: list[dict[str, object]] = []
    for member, path in enumerate(paths[:pool_members]):
        model, payload = load_model_checkpoint(path)
        distance = parameter_linf_distance(base_model, model)
        if distance is None:
            raise ValueError(f"BetaRCE member {path} has incompatible topology")
        records.append(
            {
                "member": member,
                "path": str(path.resolve()),
                "model_id": payload["model_id"],
                "parameter_linf_distance": distance,
                **behavioral_distance(base_model, model, data.X_test),
            }
        )
    selected_delta = max(float(row["parameter_linf_distance"]) for row in records)
    fingerprint = stable_hash(
        {
            "protocol_version": PROTOCOL_VERSION,
            "base_model_id": base_model_id,
            "selected_delta": selected_delta,
            "models": records,
        }
    )
    return {
        "base_model_id": base_model_id,
        "base_seed": base_seed,
        "pool_members": pool_members,
        "member_selection": "first members in numeric filename order",
        "selected_delta": selected_delta,
        "models": records,
        "calibration_fingerprint": fingerprint,
        "interpretation": (
            "diagnostic only: independently initialized BetaRCE bootstrap models "
            "are not checkpoint-aligned local updates"
        ),
    }


def _write_summary(output: Path, calibration: list[dict[str, object]]) -> None:
    generation = pd.concat(
        [
            pd.read_parquet(path)
            for path in sorted(output.glob("runs/*/generation.parquet"))
        ],
        ignore_index=True,
    )
    survival = pd.concat(
        [
            pd.read_parquet(path)
            for path in sorted(output.glob("runs/*/survival.parquet"))
        ],
        ignore_index=True,
    )
    generation.to_parquet(output / "generation.parquet", index=False)
    survival.to_parquet(output / "survival.parquet", index=False)
    write_json(
        output / "summary.json",
        {
            "protocol_version": PROTOCOL_VERSION,
            "method": "apas_betarce_pool",
            "implementation": APAS_IMPLEMENTATION,
            "diagnostic_only": True,
            "calibration": calibration,
            "generation_by_seed": apas_generation_summary(generation),
            "survival_by_seed_and_family": seed_family_survival_summary(survival),
            "survival_by_family": aggregate_survival(survival),
        },
    )


def main() -> None:
    args = parse_args()
    if args.pool_members < 1:
        raise ValueError("pool-members must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.bank / "manifest.json").read_text())
    bank_fingerprint = model_bank_fingerprint(args.bank)
    available_seeds = [int(seed) for seed in manifest["base_seeds"]]
    selected_seeds = args.base_seeds or available_seeds
    unknown_seeds = sorted(set(selected_seeds) - set(available_seeds))
    if unknown_seeds:
        raise ValueError(f"Base seeds are not in the model bank: {unknown_seeds}")
    data = load_dataset(
        str(manifest["dataset"]),
        seed=int(manifest.get("split_seed", 2026)),
        split_version=str(manifest["split_version"]),
    )

    calibration_summaries: list[dict[str, object]] = []
    for base_seed in selected_seeds:
        base_model_id, base_model, changes = load_model_bank(args.bank, base_seed)
        calibration = _pool_calibration(
            pool=args.betarce_pool,
            data=data,
            base_model=base_model,
            base_model_id=base_model_id,
            base_seed=base_seed,
            pool_members=args.pool_members,
        )
        calibration_summaries.append(calibration)
        write_json(args.output / f"calibration_seed_{base_seed}.json", calibration)
        selected_delta = float(calibration["selected_delta"])
        method_config = {
            **LOCKED_CONFIGS["apas"],
            "delta": selected_delta,
            "delta_selection": "max_linf_over_10_betarce_bootstrap_models",
            "calibration_fingerprint": calibration["calibration_fingerprint"],
            "calibration_protocol_version": PROTOCOL_VERSION,
        }
        factuals, factual_ids, row_ids = select_factuals(
            base_model,
            data.X_test,
            args.n_factuals,
            seed=base_seed + 30_000,
        )
        task = make_task(base_model, data.X_train, data.y_train, seed=base_seed)
        counterfactuals, generation = generate_counterfactuals(
            "apas",
            task,
            factuals,
            factual_ids=factual_ids,
            dataset_row_ids=row_ids,
            method_config=method_config,
            progress_description=(
                f"{data.name} · seed {base_seed} · APAS-BetaPool δ={selected_delta:.4g}"
            ),
        )
        serialized_config = json.dumps(method_config, sort_keys=True)
        for row in generation:
            row["method"] = "apas_betarce_pool"
            row["base_model_id"] = base_model_id
            row["method_config"] = serialized_config
            row["benchmark_protocol_version"] = PROTOCOL_VERSION
            row["bank_fingerprint"] = bank_fingerprint
        survival = evaluate_survival(
            "apas_betarce_pool",
            factuals,
            counterfactuals,
            changes,
            base_model,
            data.X_test,
            data.y_test,
            [bool(row["base_valid"]) for row in generation],
            factual_ids,
            row_ids,
            base_model_id,
        )
        for row in survival:
            row["method_config"] = serialized_config
            row["benchmark_protocol_version"] = PROTOCOL_VERSION
            row["bank_fingerprint"] = bank_fingerprint
        run_dir = args.output / "runs" / f"seed_{base_seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(generation).to_parquet(run_dir / "generation.parquet", index=False)
        pd.DataFrame(survival).to_parquet(run_dir / "survival.parquet", index=False)
        counterfactuals.to_parquet(run_dir / "counterfactuals.parquet", index=False)
        factuals.to_parquet(run_dir / "factuals.parquet", index=False)
        print(
            json.dumps(
                {
                    "dataset": data.name,
                    "base_seed": base_seed,
                    "apas_delta": selected_delta,
                    **generation_metrics(pd.DataFrame(generation)),
                }
            ),
            flush=True,
        )

    _write_summary(args.output, calibration_summaries)


if __name__ == "__main__":
    main()
