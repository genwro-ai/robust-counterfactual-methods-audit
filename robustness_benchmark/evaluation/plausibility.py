"""Plausibility: how well the training data supports a counterfactual."""

import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors

from robustness_benchmark.core.schema import FeatureSchema
from robustness_benchmark.evaluation.proximity import GowerDistance

METRIC_COLUMNS = (
    "knn_distance",
    "knn_distance_relative",
    "lof_inlier",
    "lof_score",
    "ynn",
)


class PlausibilityReference:
    """Data-support metrics against a base model's target region.

    Built once per base model because the target-class set, scales and neighbour
    indices are shared by every method's counterfactuals. kNN distance, LOF and
    yNN all use the Gower distance: Manhattan distance on range-scaled inputs.
    """

    def __init__(
        self,
        X_train: pd.DataFrame,
        base_model,
        *,
        schema: FeatureSchema | None = None,
        k: int = 5,
        lof_neighbors: int = 20,
        target_class: int = 1,
    ) -> None:
        self.k = k
        self.distance = GowerDistance(X_train, schema)
        self.scale = self.distance.scale
        train = X_train.to_numpy(dtype=float)
        probability = base_model.predict(X_train).iloc[:, 0].to_numpy()
        self.train_is_target = (probability >= 0.5) == bool(target_class)
        self.target = train[self.train_is_target]

        self.all_neighbors = NearestNeighbors(
            n_neighbors=min(k, len(train)), metric="manhattan"
        ).fit(train / self.scale)
        n_target = len(self.target)
        self.target_neighbors = None
        self.lof = None
        self.typical_knn_distance = float("nan")
        if n_target >= 2:
            self.k_target = min(k, n_target - 1)
            scaled_target = self.target / self.scale
            self.target_neighbors = NearestNeighbors(metric="manhattan").fit(
                scaled_target
            )
            self.lof = LocalOutlierFactor(
                n_neighbors=max(1, min(lof_neighbors, n_target - 1)),
                metric="manhattan",
                novelty=True,
            ).fit(scaled_target)
            # One extra neighbour drops each point's match with itself.
            distances, _ = self.target_neighbors.kneighbors(
                scaled_target, n_neighbors=self.k_target + 1
            )
            self.typical_knn_distance = float(
                np.median(distances[:, 1:].mean(axis=1) / self.distance.n_features)
            )

    def evaluate(
        self, counterfactuals: pd.DataFrame, base_valid: list[bool]
    ) -> list[dict[str, object]]:
        """One metric row per counterfactual; invalid rows are NaN, never scored."""

        valid = np.asarray(base_valid, dtype=bool)
        empty = {column: float("nan") for column in METRIC_COLUMNS}
        empty["lof_inlier"] = None
        rows: list[dict[str, object]] = [dict(empty) for _ in valid]
        if not valid.any():
            return rows
        scaled_cf = counterfactuals.to_numpy(dtype=float)[valid] / self.scale
        results: dict[str, np.ndarray] = {
            "ynn": self.train_is_target[
                self.all_neighbors.kneighbors(scaled_cf, return_distance=False)
            ].mean(axis=1),
        }
        if self.target_neighbors is not None:
            distances, _ = self.target_neighbors.kneighbors(
                scaled_cf, n_neighbors=self.k_target
            )
            knn = distances.mean(axis=1) / self.distance.n_features
            results["knn_distance"] = knn
            results["knn_distance_relative"] = knn / self.typical_knn_distance
            results["lof_inlier"] = self.lof.predict(scaled_cf) == 1
            results["lof_score"] = -self.lof.score_samples(scaled_cf)
        for position, row_index in enumerate(np.flatnonzero(valid)):
            for column, values in results.items():
                value = values[position]
                rows[row_index][column] = (
                    bool(value) if column == "lof_inlier" else float(value)
                )
        return rows


def plausibility_summary(valid: pd.DataFrame) -> dict[str, float | None]:
    """Means over base-valid rows; None for metrics a run did not record."""

    def mean(column: str, values: pd.Series | None = None) -> float | None:
        if column not in valid.columns:
            return None
        values = valid[column] if values is None else values
        values = pd.to_numeric(values, errors="coerce").dropna()
        return float(values.mean()) if len(values) else None

    inlier = (
        valid["lof_inlier"].map({True: 1.0, False: 0.0})
        if "lof_inlier" in valid.columns
        else None
    )
    return {
        "mean_knn_distance": mean("knn_distance"),
        "mean_knn_distance_relative": mean("knn_distance_relative"),
        "lof_inlier_rate": mean("lof_inlier", inlier),
        "mean_lof_score": mean("lof_score"),
        "mean_ynn": mean("ynn"),
    }
