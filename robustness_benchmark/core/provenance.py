import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from robustness_benchmark.core.classifiers import Classifier, classifier_from_payload


def row_id_value(value: object) -> int | str:
    if isinstance(value, (int, np.integer)):
        return int(value)
    return str(value)


def save_model_checkpoint(
    model: Classifier,
    model_id: str,
    path: Path,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_type": model.model_type,
            "model_id": model_id,
            "metadata": dict(metadata or {}),
            **model.checkpoint_payload(),
        },
        path,
    )


def load_model_checkpoint(path: Path) -> tuple[Classifier, dict[str, Any]]:
    """Restore a saved model variant and return its complete checkpoint payload."""

    payload: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=True)
    return classifier_from_payload(payload), payload


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def stable_hash(value: object) -> str:
    """SHA-256 of a value's canonical JSON."""

    serialized = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode()).hexdigest()
