"""Training and selection of base models."""

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, log_loss

from robustness_benchmark.core.classifiers import Classifier, TorchBinaryModel
from robustness_benchmark.core.data import DatasetSplit
from robustness_benchmark.core.model_specs import MLPSpec, ModelSpec

SELECTION_DEV_SEEDS = (2026, 2027, 2028)


def fit_spec(
    spec: ModelSpec,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    *,
    seed: int,
) -> Classifier:
    """Single construction point for every trained model in the bank."""

    return spec.fit(X_train, y_train, X_val, y_val, seed=seed)


def fit_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    *,
    seed: int,
    hidden_dim: tuple[int, ...] = (16, 16),
    epochs: int = 300,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    optimizer_name: str = "adam",
    patience: int = 30,
) -> TorchBinaryModel:
    spec = MLPSpec(
        hidden_dim=hidden_dim,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        optimizer_name=optimizer_name,
        patience=patience,
    )
    model = fit_spec(spec, X_train, y_train, X_val, y_val, seed=seed)
    if not isinstance(model, TorchBinaryModel):
        raise TypeError("an MLP spec must produce a TorchBinaryModel")
    return model


def select_base_spec(
    data: DatasetSplit,
    base: ModelSpec,
    *,
    seeds: tuple[int, ...] = SELECTION_DEV_SEEDS,
) -> tuple[ModelSpec, list[dict[str, object]]]:
    """Select the grid configuration with the lowest validation log-loss.

    The grid (``type(base).selection_grid(base)``) and the rule are fixed in
    advance; each candidate is scored by its mean validation log-loss over
    ``seeds``. Changed models are never consulted.
    """

    candidates = type(base).selection_grid(base)
    rows: list[dict[str, object]] = []
    means: list[float] = []
    for index, candidate in enumerate(candidates):
        losses = []
        for seed in seeds:
            model = fit_spec(
                candidate, data.X_train, data.y_train, data.X_val, data.y_val, seed=seed
            )
            probability = model.predict(data.X_val).iloc[:, 0].to_numpy()
            loss = float(log_loss(data.y_val, probability, labels=[0, 1]))
            losses.append(loss)
            rows.append(
                {
                    "candidate": index,
                    "model_spec": candidate.to_dict(),
                    "seed": seed,
                    "validation_log_loss": loss,
                    "validation_balanced_accuracy": float(
                        balanced_accuracy_score(data.y_val, probability >= 0.5)
                    ),
                }
            )
        means.append(float(np.mean(losses)))

    selected = int(np.argmin(means))
    for row in rows:
        row["mean_validation_log_loss"] = means[int(row["candidate"])]
        row["recommended"] = row["candidate"] == selected
    return candidates[selected], rows
