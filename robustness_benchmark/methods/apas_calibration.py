import hashlib
import json
from pathlib import Path

import numpy as np

from robustness_benchmark.core.provenance import file_sha256, save_model_checkpoint, write_json
from robustness_benchmark.core.training import (
    make_apas_calibration_models,
    parameter_linf_distance,
)
from robustness_benchmark.evaluation.metrics import behavioral_distance

APAS_CALIBRATION_PROTOCOL_VERSION = "apas_update_calibration_v1"


def stable_hash(value: object) -> str:
    serialized = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode()).hexdigest()


def apas_calibration_config(
    *,
    dataset: str,
    base_model_id: str,
    bank_fingerprint: str,
    calibration_replicates: int,
    update_epochs: int,
    update_batch_size: int,
    update_learning_rate: float,
    update_weight_decay: float,
) -> dict[str, object]:
    return {
        "protocol_version": APAS_CALIBRATION_PROTOCOL_VERSION,
        "dataset": dataset,
        "base_model_id": base_model_id,
        "bank_fingerprint": bank_fingerprint,
        "calibration_replicates": calibration_replicates,
        "update_epochs": update_epochs,
        "update_batch_size": update_batch_size,
        "update_learning_rate": update_learning_rate,
        "update_weight_decay": update_weight_decay,
        "sampling_scheme": "bootstrap_with_replacement",
        "sampling_seed_rule": "base_seed + 200000 + replicate",
        "initialized_from_base_checkpoint": True,
        "optimizer": "fresh_adam",
        "delta_estimator": "maximum_parameter_linf_distance",
        "excluded_from_evaluation_changes": True,
    }


def load_or_create_apas_calibration(
    *,
    output: Path,
    data,
    base_model,
    base_model_id: str,
    base_seed: int,
    bank_fingerprint: str,
    calibration_replicates: int,
    update_epochs: int,
    update_batch_size: int,
    update_learning_rate: float,
    update_weight_decay: float,
) -> dict[str, object]:
    directory = output / "calibration" / f"seed_{base_seed}"
    metadata_path = directory / "calibration.json"
    config = apas_calibration_config(
        dataset=data.name,
        base_model_id=base_model_id,
        bank_fingerprint=bank_fingerprint,
        calibration_replicates=calibration_replicates,
        update_epochs=update_epochs,
        update_batch_size=update_batch_size,
        update_learning_rate=update_learning_rate,
        update_weight_decay=update_weight_decay,
    )
    if metadata_path.is_file():
        existing = json.loads(metadata_path.read_text())
        checkpoints = existing.get("checkpoints", [])
        reusable = (
            existing.get("config") == config
            and len(checkpoints) == calibration_replicates
        )
        if reusable:
            for checkpoint in checkpoints:
                path = directory / str(checkpoint["path"])
                if not path.is_file() or file_sha256(path) != checkpoint["sha256"]:
                    reusable = False
                    break
        if reusable:
            return existing

    directory.mkdir(parents=True, exist_ok=True)
    calibration_models = make_apas_calibration_models(
        data,
        base_model,
        base_seed=base_seed,
        size=calibration_replicates,
        update_epochs=update_epochs,
        batch_size=update_batch_size,
        learning_rate=update_learning_rate,
        weight_decay=update_weight_decay,
    )
    base_probability = base_model.predict(data.X_test).iloc[:, 0].to_numpy()
    base_accuracy = float(np.mean((base_probability >= 0.5) == data.y_test.to_numpy()))
    records: list[dict[str, object]] = []
    checkpoints: list[dict[str, object]] = []
    for replicate, calibration in enumerate(calibration_models):
        distance = parameter_linf_distance(base_model, calibration.model)
        if distance is None:
            raise RuntimeError("APAS calibration model is not parameter-aligned")
        changed_probability = (
            calibration.model.predict(data.X_test).iloc[:, 0].to_numpy()
        )
        changed_accuracy = float(
            np.mean((changed_probability >= 0.5) == data.y_test.to_numpy())
        )
        path = directory / f"replica_{replicate:02d}.pt"
        save_model_checkpoint(
            calibration.model,
            f"{base_model_id}/{calibration.change_id}",
            path,
            calibration.metadata,
        )
        checkpoint = {
            "path": path.name,
            "sha256": file_sha256(path),
            "change_id": calibration.change_id,
        }
        checkpoints.append(checkpoint)
        records.append(
            {
                "replicate": replicate,
                "parameter_linf_distance": distance,
                **behavioral_distance(base_model, calibration.model, data.X_test),
                "base_calibration_accuracy": base_accuracy,
                "changed_calibration_accuracy": changed_accuracy,
                "calibration_accuracy_drop": base_accuracy - changed_accuracy,
                **calibration.metadata,
            }
        )

    selected_delta = max(float(record["parameter_linf_distance"]) for record in records)
    fingerprint = stable_hash(
        {
            "config": config,
            "checkpoints": checkpoints,
            "selected_delta": selected_delta,
        }
    )
    result = {
        "config": config,
        "selected_delta": selected_delta,
        "replicas": records,
        "checkpoints": checkpoints,
        "calibration_fingerprint": fingerprint,
    }
    write_json(metadata_path, result)
    return result


def apas_method_config(
    base_config: dict[str, object], calibration: dict[str, object]
) -> dict[str, object]:
    return {
        **base_config,
        "delta": float(calibration["selected_delta"]),
        "delta_selection": (
            "max_parameter_linf_over_checkpoint_aligned_update_replicas"
        ),
        "calibration_fingerprint": calibration["calibration_fingerprint"],
        "calibration_protocol_version": APAS_CALIBRATION_PROTOCOL_VERSION,
    }


def summarize_apas_calibration(
    calibration: dict[str, object], *, base_model_id: str, base_seed: int
) -> dict[str, object]:
    replicas = calibration["replicas"]
    if not isinstance(replicas, list):
        raise TypeError("APAS calibration replicas must be a list")
    return {
        "base_model_id": base_model_id,
        "base_seed": base_seed,
        "selected_delta": float(calibration["selected_delta"]),
        "calibration_fingerprint": calibration["calibration_fingerprint"],
        "parameter_linf_distances": [
            replica["parameter_linf_distance"] for replica in replicas
        ],
        "mean_hard_disagreement": float(
            np.mean([replica["hard_disagreement"] for replica in replicas])
        ),
        "mean_probability_mae": float(
            np.mean([replica["probability_mae"] for replica in replicas])
        ),
    }
