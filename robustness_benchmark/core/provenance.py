import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from robustness_benchmark.core.model import TorchBinaryModel


def json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def row_id_value(value: object) -> int | str:
    if isinstance(value, (int, np.integer)):
        return int(value)
    return str(value)


def git_info(path: Path) -> dict[str, object]:
    def run(*arguments: str) -> str | None:
        completed = subprocess.run(
            ["git", "-C", str(path), *arguments],
            capture_output=True,
            check=False,
            text=True,
        )
        return completed.stdout.strip() if completed.returncode == 0 else None

    initialized = run("rev-parse", "--git-dir") is not None
    commit = run("rev-parse", "HEAD") if initialized else None
    status = run("status", "--porcelain=v1") if initialized else None
    return {
        "path": str(path.resolve()),
        "initialized": initialized,
        "commit": commit,
        "dirty": bool(status) if status is not None else None,
        "remote": run("remote", "get-url", "origin") if initialized else None,
    }


def environment_info() -> dict[str, object]:
    packages = [
        "gurobipy",
        "lime",
        "numpy",
        "pandas",
        "pyarrow",
        "scikit-learn",
        "torch",
    ]
    versions = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": versions,
    }


def save_model_checkpoint(
    model: TorchBinaryModel,
    model_id: str,
    path: Path,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_id": model_id,
            "input_dim": model.input_dim,
            "hidden_dim": model.hidden_dim,
            "output_dim": model.output_dim,
            "seed": model.seed,
            "metadata": dict(metadata or {}),
            "state_dict": model.get_torch_model().state_dict(),
        },
        path,
    )


def load_model_checkpoint(path: Path) -> tuple[TorchBinaryModel, dict[str, Any]]:
    """Restore a saved model variant and return its complete checkpoint payload."""

    payload: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=True)
    model = TorchBinaryModel(
        input_dim=int(payload["input_dim"]),
        hidden_dim=tuple(int(width) for width in payload["hidden_dim"]),
        seed=int(payload["seed"]),
    )
    model.get_torch_model().load_state_dict(payload["state_dict"])
    model.get_torch_model().eval()
    return model, payload


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
