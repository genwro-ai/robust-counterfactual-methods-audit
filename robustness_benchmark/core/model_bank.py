"""The model bank: base models and their changed variants, built and stored once.

A bank directory holds ``manifest.json`` (spec, seeds, checkpoint paths and
hashes), ``model_change_catalog.parquet`` (one row per changed model with its
behavioural distance from the base model) and the checkpoints under
``models/``.
"""

import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score
from tqdm.auto import tqdm

from robustness_benchmark.core.classifiers import Classifier, predict_crisp
from robustness_benchmark.core.data import DatasetSplit, load_dataset
from robustness_benchmark.core.fitting import fit_spec, select_base_spec
from robustness_benchmark.core.model_changes import (
    BETARCE_GENERATION_ENSEMBLE_PROTOCOL,
    BETARCE_GENERATION_ENSEMBLE_SIZE,
    BETARCE_SAMPLING_SEED_OFFSET,
    DISTINCT_VALIDATION_MAE,
    FULL_VARIANTS_PER_FAMILY,
    ChangedModel,
    behavioral_distance,
    make_betarce_generation_ensemble,
    make_betarce_generation_model,
    make_changed_models,
)
from robustness_benchmark.core.model_specs import MLPSpec, ModelSpec
from robustness_benchmark.core.protocol import SELECTION_DEV_SEEDS
from robustness_benchmark.core.provenance import (
    file_sha256,
    load_model_checkpoint,
    save_model_checkpoint,
    stable_hash,
    write_json,
)

CONFIGURATION_CANDIDATES_VERSION = "ordered_distinct_v1"


def base_model_id(dataset: str, spec: ModelSpec, base_seed: int) -> str:
    return f"{dataset}:{spec.id_fragment()}:seed_{base_seed}"


def change_checkpoint_id(base_id: str, change_id: str) -> str:
    return f"{base_id}/{change_id}"


def ensemble_member_id(base_id: str, protocol: str, member: int) -> str:
    return f"{base_id}/betarce_generation/{protocol}/member_{member:02d}"


def register_checkpoint(
    checkpoints: dict[str, dict[str, object]],
    key: str,
    model: Classifier,
    model_id: str,
    path: Path,
    root: Path,
    metadata: dict[str, object],
) -> None:
    """Save a checkpoint and record its path (relative to ``root``) and hash."""

    save_model_checkpoint(model, model_id, path, metadata)
    checkpoints[key] = {
        "path": str(path.relative_to(root)),
        "sha256": file_sha256(path),
    }


def stratify_severity(rows: list[dict[str, object]]) -> None:
    """Relabel every candidate with relative severity from behavioral distance.

    This is the single stratification rule shared by bank construction and
    bank loading; every candidate participates, without quality exclusions.
    """

    ordered = sorted(
        rows,
        key=lambda row: (row["probability_mae"], row["hard_disagreement"]),
    )
    denominator = max(len(ordered), 1)
    for rank, row in enumerate(ordered, start=1):
        percentile = (rank - 0.5) / denominator
        row["severity_percentile"] = percentile
        row["severity"] = (
            "low"
            if percentile <= 1 / 3
            else "medium"
            if percentile <= 2 / 3
            else "high"
        )


def assess_model_changes(
    changes: list[ChangedModel],
    base_model: Classifier,
    X_reference: pd.DataFrame,
    y_reference: pd.Series,
    *,
    maximum_accuracy_drop: float = 0.03,
) -> list[dict[str, object]]:
    """Describe every changed model and stratify the bank by severity.

    The accuracy drop and its ``quality_pass`` flag are descriptive only: no
    changed model is excluded.
    """

    labels = y_reference.to_numpy()
    base_accuracy = float(
        balanced_accuracy_score(labels, predict_crisp(base_model, X_reference))
    )
    rows: list[dict[str, object]] = []
    for change in changes:
        changed_accuracy = float(
            balanced_accuracy_score(labels, predict_crisp(change.model, X_reference))
        )
        accuracy_drop = base_accuracy - changed_accuracy
        rows.append(
            {
                "change_id": change.change_id,
                "change_family": change.change_family,
                "change_level": change.change_level,
                **behavioral_distance(base_model, change.model, X_reference),
                "base_reference_balanced_accuracy": base_accuracy,
                "changed_reference_balanced_accuracy": changed_accuracy,
                "balanced_accuracy_drop": accuracy_drop,
                "quality_pass": accuracy_drop <= maximum_accuracy_drop,
                "maximum_accuracy_drop": maximum_accuracy_drop,
                "metadata": change.metadata,
            }
        )

    stratify_severity(rows)
    return rows


def _configure_training_worker() -> None:
    """Prevent each training process from spawning a large CPU thread pool."""

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # PyTorch permits setting this only once in a process.
        pass


def build_seed_bank(
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
    base_id = base_model_id(dataset, spec, base_seed)
    directory = output / "models" / f"seed_{base_seed}"
    checkpoints: dict[str, dict[str, object]] = {}
    register_checkpoint(
        checkpoints,
        base_id,
        base_model,
        base_id,
        directory / "base.pt",
        output,
        {
            "role": "base",
            "split_version": split_version,
            "training_seed": base_seed,
            "model_spec": spec.to_dict(),
        },
    )

    changes = make_changed_models(
        data, base_model, base_seed=base_seed, spec=spec, profile="full"
    )
    assessments = assess_model_changes(
        changes,
        base_model,
        data.X_test,
        data.y_test,
        maximum_accuracy_drop=maximum_accuracy_drop,
    )
    for index, (change, assessment) in enumerate(
        zip(changes, assessments, strict=True)
    ):
        register_checkpoint(
            checkpoints,
            change_checkpoint_id(base_id, change.change_id),
            change.model,
            change.change_id,
            directory / f"change_{index:03d}_{change.change_family}.pt",
            output,
            {"role": "changed", "base_model_id": base_id, **assessment},
        )
    catalog_rows = [
        {"base_model_id": base_id, **assessment} for assessment in assessments
    ]

    if include_betarce_ensemble:
        protocol = BETARCE_GENERATION_ENSEMBLE_PROTOCOL
        generation_models = make_betarce_generation_ensemble(
            data, base_seed=base_seed, spec=spec
        )
        for member, generation_model in enumerate(generation_models):
            member_id = ensemble_member_id(base_id, protocol, member)
            register_checkpoint(
                checkpoints,
                member_id,
                generation_model,
                member_id,
                directory / "betarce_generation" / protocol / f"member_{member:02d}.pt",
                output,
                {
                    "role": "generation_ensemble",
                    "method": "betarce",
                    "excluded_from_evaluation_changes": True,
                    "member": member,
                    "protocol": protocol,
                    "initialization_seed": base_seed,
                },
            )
    return catalog_rows, checkpoints


def select_and_record_spec(
    data: DatasetSplit,
    base: ModelSpec,
    *,
    output: Path,
    prespecified: bool,
) -> ModelSpec:
    """Run the a priori base-configuration selection and record it.

    LR and XGBoost use the selected configuration. The MLP keeps its
    prespecified architecture; its selection run is reported as an audit.
    """

    recommended, rows = select_base_spec(data, base)
    selected = base if prespecified else recommended
    pd.DataFrame(
        [
            {
                **row,
                "model_spec": json.dumps(row["model_spec"]),
                "used": row["model_spec"] == selected.to_dict(),
            }
            for row in rows
        ]
    ).to_parquet(output / "base_config_selection.parquet", index=False)
    write_json(
        output / "base_config_selection.json",
        {
            "model_type": base.model_type,
            "dev_seeds": list(SELECTION_DEV_SEEDS),
            "selection_rule": "lowest mean validation log-loss over the dev seeds",
            "recommended": recommended.to_dict(),
            "selected": selected.to_dict(),
            "prespecified": prespecified,
            "runs": rows,
        },
    )
    return selected


def build_model_bank(
    *,
    dataset: str,
    split_version: str,
    split_seed: int,
    base_spec: ModelSpec,
    base_seeds: list[int],
    maximum_accuracy_drop: float,
    include_betarce_ensemble: bool,
    workers: int,
    output: Path,
) -> dict[str, object]:
    """Build the bank of every base seed and write its manifest and catalog.

    An MLP ``base_spec`` is prespecified; for the other model classes it seeds
    the a priori configuration grid.
    """

    if workers < 1:
        raise ValueError("workers must be positive")
    manifest_path = output / "manifest.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text())
        if (existing["dataset"], existing["model_type"]) != (
            dataset,
            base_spec.model_type,
        ):
            raise ValueError(
                f"{output} holds a {existing['model_type']} bank for "
                f"{existing['dataset']}; choose another output directory"
            )
    data = load_dataset(dataset, seed=split_seed, split_version=split_version)
    output.mkdir(parents=True, exist_ok=True)
    spec = select_and_record_spec(
        data, base_spec, output=output, prespecified=isinstance(base_spec, MLPSpec)
    )

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
        seed_banks = [build_seed_bank(**arguments) for arguments in worker_arguments]
    else:
        with ProcessPoolExecutor(
            max_workers=min(workers, len(worker_arguments)),
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_configure_training_worker,
        ) as executor:
            futures = [
                executor.submit(build_seed_bank, **arguments)
                for arguments in worker_arguments
            ]
            seed_banks = [future.result() for future in futures]

    catalog = pd.DataFrame([row for rows, _ in seed_banks for row in rows])
    catalog["metadata"] = catalog["metadata"].map(
        lambda value: json.dumps(value, sort_keys=True)
    )
    catalog.to_parquet(output / "model_change_catalog.parquet", index=False)
    manifest: dict[str, object] = {
        "created_at": datetime.now(UTC).isoformat(),
        "dataset": dataset,
        "split_version": split_version,
        "split_seed": split_seed,
        "base_seeds": base_seeds,
        "model_type": spec.model_type,
        "model_spec": spec.to_dict(),
        "change_families": list(spec.change_families),
        "candidate_models_per_family": FULL_VARIANTS_PER_FAMILY,
        "candidate_models_per_base": (
            FULL_VARIANTS_PER_FAMILY * len(spec.change_families)
        ),
        "configuration_candidates_version": CONFIGURATION_CANDIDATES_VERSION,
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
        "checkpoints": {
            key: value
            for _, checkpoints in seed_banks
            for key, value in checkpoints.items()
        },
    }
    write_json(manifest_path, manifest)
    return manifest


def load_base_model(bank: Path, base_seed: int) -> tuple[str, Classifier]:
    """Load one base model identified by its experiment seed."""

    manifest = json.loads((bank / "manifest.json").read_text())
    catalog = pd.read_parquet(bank / "model_change_catalog.parquet")
    base_rows = catalog[catalog["base_model_id"].str.endswith(f"seed_{base_seed}")]
    if base_rows.empty:
        raise ValueError(f"No model bank found for base seed {base_seed}")
    base_id = str(base_rows["base_model_id"].iloc[0])
    base_model, _ = load_model_checkpoint(
        bank / manifest["checkpoints"][base_id]["path"]
    )
    return base_id, base_model


def load_model_bank(
    bank: Path, base_seed: int
) -> tuple[str, Classifier, list[ChangedModel]]:
    """Load a base model and every changed-model variant for one seed."""

    manifest = json.loads((bank / "manifest.json").read_text())
    catalog = pd.read_parquet(bank / "model_change_catalog.parquet")
    base_rows = catalog[catalog["base_model_id"].str.endswith(f"seed_{base_seed}")]
    base_id, base_model = load_base_model(bank, base_seed)

    rows = base_rows.to_dict("records")
    stratify_severity(rows)
    changes: list[ChangedModel] = []
    for row in rows:
        change_id = str(row["change_id"])
        key = change_checkpoint_id(base_id, change_id)
        model, _ = load_model_checkpoint(bank / manifest["checkpoints"][key]["path"])
        metadata = json.loads(str(row["metadata"]))
        metadata.update(
            {
                "quality_pass": bool(row["quality_pass"]),
                "severity": str(row["severity"]),
                "severity_percentile": float(row["severity_percentile"]),
                "balanced_accuracy_drop": float(row["balanced_accuracy_drop"]),
            }
        )
        changes.append(
            ChangedModel(
                change_id,
                str(row["change_family"]),
                str(row["change_level"]),
                model,
                metadata,
            )
        )
    return base_id, base_model, changes


def model_bank_fingerprint(bank: Path) -> str:
    """Identify the exact model-bank manifest and catalog used by a run."""

    return stable_hash(
        {
            "manifest_sha256": file_sha256(bank / "manifest.json"),
            "model_change_catalog_sha256": file_sha256(
                bank / "model_change_catalog.parquet"
            ),
        }
    )


def load_generation_ensemble(
    *,
    bank: Path,
    base_id: str,
    data: DatasetSplit,
    base_seed: int,
    spec: ModelSpec,
    cache: Path,
    size: int = BETARCE_GENERATION_ENSEMBLE_SIZE,
    protocol: str = BETARCE_GENERATION_ENSEMBLE_PROTOCOL,
) -> list[Classifier]:
    """BetaRCE's generation ensemble: from the bank, else trained and cached.

    Cached members are reused only when they were trained for this bank.
    """

    checkpoints = json.loads((bank / "manifest.json").read_text())["checkpoints"]
    member_ids = [
        ensemble_member_id(base_id, protocol, member) for member in range(size)
    ]
    if all(member_id in checkpoints for member_id in member_ids):
        return [
            load_model_checkpoint(bank / checkpoints[member_id]["path"])[0]
            for member_id in member_ids
        ]
    directory = cache / protocol / f"seed_{base_seed}"
    fingerprint = model_bank_fingerprint(bank)
    models = []
    for member in tqdm(
        range(size), desc=f"seed {base_seed} · BetaRCE ensemble", leave=False
    ):
        path = directory / f"member_{member:02d}.pt"
        if path.is_file():
            model, payload = load_model_checkpoint(path)
            if payload["metadata"].get("bank_fingerprint") == fingerprint:
                models.append(model)
                continue
        model = make_betarce_generation_model(
            data, base_seed=base_seed, member=member, spec=spec
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
                "bootstrap_sampling_seed": base_seed
                + BETARCE_SAMPLING_SEED_OFFSET
                + member,
                "bank_fingerprint": fingerprint,
            },
        )
        models.append(model)
    return models
