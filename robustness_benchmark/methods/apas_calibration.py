import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from robustness_benchmark.core.provenance import (
    file_sha256,
    save_model_checkpoint,
    write_json,
)
from robustness_benchmark.core.training import (
    make_update_replicas,
    parameter_linf_distance,
)
from robustness_benchmark.evaluation.metrics import behavioral_distance

APAS_CALIBRATION_PROTOCOL_VERSION = "apas_update_calibration_v1"
RNCE_CALIBRATION_PROTOCOL_VERSION = "rnce_incremental_delta_v1"


@dataclass(frozen=True)
class UpdateCalibration:
    """How a method's parameter radius is calibrated from update replicas."""

    method: str
    protocol_version: str
    replicates: int
    update_epochs: int
    update_batch_size: int
    update_learning_rate: float
    update_weight_decay: float
    sample_fraction: float
    replace: bool
    seed_offset: int
    delta_estimator: str  # "max" (APAS) or "mean" (RNCE)


def apas_update_calibration(
    replicates: int,
    update_epochs: int,
    update_batch_size: int,
    update_learning_rate: float,
    update_weight_decay: float,
) -> UpdateCalibration:
    """APAS: the largest distance over bootstrap updates on the whole pool."""

    return UpdateCalibration(
        method="apas",
        protocol_version=APAS_CALIBRATION_PROTOCOL_VERSION,
        replicates=replicates,
        update_epochs=update_epochs,
        update_batch_size=update_batch_size,
        update_learning_rate=update_learning_rate,
        update_weight_decay=update_weight_decay,
        sample_fraction=1.0,
        replace=True,
        seed_offset=200_000,
        delta_estimator="max",
    )


# Jiang et al. (AIJ 2024, §8.2) set delta_inc to the mean L-infinity distance
# over all weights and biases between the model and ten copies trained for one
# more epoch (Adam, lr 0.01, batch 256, shuffled) on 10% of the incoming data,
# sampled without replacement (intabs/evaluation.py incremental_training_get_delta
# and classifier_binary.py partial_fit). The update pool is the incoming data.
RNCE_UPDATE_CALIBRATION = UpdateCalibration(
    method="rnce",
    protocol_version=RNCE_CALIBRATION_PROTOCOL_VERSION,
    replicates=10,
    update_epochs=1,
    update_batch_size=256,
    update_learning_rate=0.01,
    update_weight_decay=0.0,
    sample_fraction=0.1,
    replace=False,
    seed_offset=210_000,
    delta_estimator="mean",
)


def stable_hash(value: object) -> str:
    serialized = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode()).hexdigest()


def calibration_config(
    protocol: UpdateCalibration,
    *,
    dataset: str,
    base_model_id: str,
    bank_fingerprint: str,
) -> dict[str, object]:
    return {
        **asdict(protocol),
        "dataset": dataset,
        "base_model_id": base_model_id,
        "bank_fingerprint": bank_fingerprint,
        "initialized_from_base_checkpoint": True,
        "optimizer": "fresh_adam",
        "excluded_from_evaluation_changes": True,
    }


def load_or_create_calibration(
    protocol: UpdateCalibration,
    *,
    output: Path,
    data,
    base_model,
    base_model_id: str,
    base_seed: int,
    bank_fingerprint: str,
) -> dict[str, object]:
    """Train (or reuse) one protocol's replicas and select its radius."""

    directory = output / "calibration" / protocol.method / f"seed_{base_seed}"
    metadata_path = directory / "calibration.json"
    config = calibration_config(
        protocol,
        dataset=data.name,
        base_model_id=base_model_id,
        bank_fingerprint=bank_fingerprint,
    )
    if metadata_path.is_file():
        existing = json.loads(metadata_path.read_text())
        checkpoints = existing.get("checkpoints", [])
        reusable = (
            existing.get("config") == config and len(checkpoints) == protocol.replicates
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
    calibration_models = make_update_replicas(
        data,
        base_model,
        base_seed=base_seed,
        name=protocol.method,
        size=protocol.replicates,
        seed_offset=protocol.seed_offset,
        sample_fraction=protocol.sample_fraction,
        replace=protocol.replace,
        update_epochs=protocol.update_epochs,
        batch_size=protocol.update_batch_size,
        learning_rate=protocol.update_learning_rate,
        weight_decay=protocol.update_weight_decay,
    )
    base_probability = base_model.predict(data.X_test).iloc[:, 0].to_numpy()
    base_accuracy = float(np.mean((base_probability >= 0.5) == data.y_test.to_numpy()))
    records: list[dict[str, object]] = []
    checkpoints: list[dict[str, object]] = []
    for replicate, calibration in enumerate(calibration_models):
        distance = parameter_linf_distance(base_model, calibration.model)
        if distance is None:
            raise RuntimeError("calibration model is not parameter-aligned")
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

    distances = [float(record["parameter_linf_distance"]) for record in records]
    estimate = {"max": max, "mean": lambda values: float(np.mean(values))}
    selected_delta = estimate[protocol.delta_estimator](distances)
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


def rnce_method_config(
    base_config: dict[str, object], calibration: dict[str, object]
) -> dict[str, object]:
    """RNCE's delta_inc from its own update replicas (Jiang et al. §8.2)."""

    delta = float(calibration["selected_delta"])
    return {
        **base_config,
        "delta": delta,
        "bias_delta": delta,
        "delta_selection": "mean_parameter_linf_over_incremental_update_replicas",
        "calibration_fingerprint": calibration["calibration_fingerprint"],
        "calibration_protocol_version": RNCE_CALIBRATION_PROTOCOL_VERSION,
    }


def summarize_calibration(
    calibration: dict[str, object], *, base_model_id: str, base_seed: int
) -> dict[str, object]:
    replicas = calibration["replicas"]
    if not isinstance(replicas, list):
        raise TypeError("calibration replicas must be a list")
    return {
        "method": calibration["config"]["method"],
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
