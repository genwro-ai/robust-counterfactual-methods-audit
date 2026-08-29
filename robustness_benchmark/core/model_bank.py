import json
from pathlib import Path

import pandas as pd

from robustness_benchmark.core.data import select_adverse_indices
from robustness_benchmark.core.provenance import (
    file_sha256,
    load_model_checkpoint,
    row_id_value,
)
from robustness_benchmark.core.training import ChangedModel
from robustness_benchmark.evaluation.metrics import stratify_severity


def load_base_model(bank: Path, base_seed: int) -> tuple[str, object]:
    """Load one base model identified by its experiment seed."""

    manifest = json.loads((bank / "manifest.json").read_text())
    catalog = pd.read_parquet(bank / "model_change_catalog.parquet")
    base_rows = catalog[catalog["base_model_id"].str.endswith(f"seed_{base_seed}")]
    if base_rows.empty:
        raise ValueError(f"No model bank found for base seed {base_seed}")
    base_id = str(base_rows["base_model_id"].iloc[0])
    base_path = bank / manifest["checkpoints"][base_id]["path"]
    base_model, _ = load_model_checkpoint(base_path)
    return base_id, base_model


def load_model_bank(
    bank: Path, base_seed: int
) -> tuple[str, object, list[ChangedModel]]:
    """Load a base model and every changed-model variant for one seed."""

    manifest = json.loads((bank / "manifest.json").read_text())
    catalog = pd.read_parquet(bank / "model_change_catalog.parquet")
    base_rows = catalog[catalog["base_model_id"].str.endswith(f"seed_{base_seed}")]
    base_id, base_model = load_base_model(bank, base_seed)

    # Older catalogs can contain obsolete "excluded" labels. Reapply the
    # current relative severity stratification to every model in the bank.
    rows = base_rows.to_dict("records")
    stratify_severity(rows)
    changes: list[ChangedModel] = []
    for row in rows:
        change_id = str(row["change_id"])
        path = bank / manifest["checkpoints"][f"{base_id}/{change_id}"]["path"]
        model, _ = load_model_checkpoint(path)
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


def select_factuals(
    model, X: pd.DataFrame, requested: int, seed: int
) -> tuple[pd.DataFrame, list[int], list[int | str]]:
    """Select a reproducible sample of adverse factual instances."""

    probability = model.predict(X).iloc[:, 0]
    probability.index = X.index
    indices = select_adverse_indices(probability, requested=requested, seed=seed)
    frame = X.loc[indices].reset_index(drop=True)
    factual_ids = list(range(len(frame)))
    dataset_row_ids = [row_id_value(value) for value in indices]
    return frame, factual_ids, dataset_row_ids


def model_bank_fingerprint(bank: Path) -> str:
    """Identify the exact model-bank manifest and catalog used by a run."""

    return json.dumps(
        {
            "manifest_sha256": file_sha256(bank / "manifest.json"),
            "model_change_catalog_sha256": file_sha256(
                bank / "model_change_catalog.parquet"
            ),
        },
        sort_keys=True,
    )
