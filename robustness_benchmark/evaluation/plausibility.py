import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors

from robustness_benchmark.core.schema import FeatureSchema

SPARSITY_TOLERANCE = 1e-6
METRIC_COLUMNS = (
    "n_changed_features",
    "fraction_changed_features",
    "knn_distance",
    "knn_distance_relative",
    "lof_inlier",
    "lof_score",
    "ynn",
)


def robust_feature_scale(X: pd.DataFrame) -> pd.Series:
    """Per-feature MAD, falling back to std then 1, so L1 costs are unit-free."""

    mad = (X - X.median(axis=0)).abs().median(axis=0)
    std = X.std(axis=0, ddof=0)
    return mad.where(mad > 1e-8, std).where(lambda value: value > 1e-8, 1.0)


class MixedDistance:
    """Gower-style distance: the mean over raw features of a per-feature cost.

    A numeric feature costs |delta| / s with s its robust training scale; a
    categorical feature costs 1 when its category changes. One-hot columns are
    divided by 2, so on valid encodings a changed category (two flipped
    columns) costs exactly 1 and the distance stays a scaled L1 that
    neighbour indices can use. On all-numeric data it is the mean robust-scaled
    L1 over columns.
    """

    def __init__(self, X_train: pd.DataFrame, schema: FeatureSchema | None = None):
        schema = schema or FeatureSchema.numeric_only(X_train.columns)
        self.scale = robust_feature_scale(X_train).to_numpy(copy=True)
        self.n_features = len(schema.raw_features)
        for group in schema.one_hot_groups().values():
            self.scale[group] = 2.0
        owner = {
            column: feature
            for feature, group in schema.categorical.items()
            for column in group
        }
        feature_index = {name: i for i, name in enumerate(schema.raw_features)}
        # membership[column, feature] marks the raw feature each column encodes.
        self.membership = np.zeros((len(schema.columns), self.n_features), dtype=int)
        for column, name in enumerate(schema.columns):
            self.membership[column, feature_index[owner.get(name, name)]] = 1

    def __call__(self, a, b) -> np.ndarray:
        difference = np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))
        return (difference / self.scale).sum(axis=-1) / self.n_features

    def n_changed(self, a, b) -> np.ndarray:
        """Raw features changed beyond tolerance; a one-hot group counts once."""

        changed = np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))
        return ((changed > SPARSITY_TOLERANCE) @ self.membership > 0).sum(axis=-1)


class PlausibilityReference:
    """Sparsity and data-support metrics against a base model's target region.

    Built once per base model because the target-class set, scales and neighbour
    indices are shared by every method's counterfactuals.
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
        self.distance = MixedDistance(X_train, schema)
        self.scale = self.distance.scale
        train = X_train.to_numpy(dtype=float)
        probability = base_model.predict(X_train).iloc[:, 0].to_numpy()
        self.train_is_target = (probability >= 0.5) == bool(target_class)
        self.target = train[self.train_is_target]

        self.all_neighbors = NearestNeighbors(n_neighbors=min(k, len(train))).fit(train)
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
                n_neighbors=max(1, min(lof_neighbors, n_target - 1)), novelty=True
            ).fit(self.target)
            # One extra neighbour drops each point's match with itself.
            distances, _ = self.target_neighbors.kneighbors(
                scaled_target, n_neighbors=self.k_target + 1
            )
            self.typical_knn_distance = float(
                np.median(distances[:, 1:].mean(axis=1) / self.distance.n_features)
            )

    def evaluate(
        self,
        factuals: pd.DataFrame,
        counterfactuals: pd.DataFrame,
        base_valid: list[bool],
    ) -> list[dict[str, object]]:
        """One metric row per counterfactual; invalid rows are NaN, never scored."""

        valid = np.asarray(base_valid, dtype=bool)
        empty = {column: float("nan") for column in METRIC_COLUMNS}
        empty["lof_inlier"] = None
        rows: list[dict[str, object]] = [dict(empty) for _ in valid]
        if not valid.any():
            return rows
        x = factuals.to_numpy(dtype=float)[valid]
        cf = counterfactuals.to_numpy(dtype=float)[valid]

        n_changed = self.distance.n_changed(cf, x)
        results: dict[str, np.ndarray] = {
            "n_changed_features": n_changed.astype(float),
            "fraction_changed_features": n_changed / self.distance.n_features,
            "ynn": self.train_is_target[
                self.all_neighbors.kneighbors(cf, return_distance=False)
            ].mean(axis=1),
        }
        if self.target_neighbors is not None:
            distances, _ = self.target_neighbors.kneighbors(
                cf / self.scale, n_neighbors=self.k_target
            )
            knn = distances.mean(axis=1) / self.distance.n_features
            results["knn_distance"] = knn
            results["knn_distance_relative"] = knn / self.typical_knn_distance
            results["lof_inlier"] = self.lof.predict(cf) == 1
            results["lof_score"] = -self.lof.score_samples(cf)
        for position, row_index in enumerate(np.flatnonzero(valid)):
            for column, values in results.items():
                value = values[position]
                rows[row_index][column] = (
                    bool(value) if column == "lof_inlier" else float(value)
                )
        return rows


def plausibility_summary(valid: pd.DataFrame) -> dict[str, float | None]:
    """Means over base-valid rows; None when absent (runs predating the metrics)."""

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
        "mean_n_changed_features": mean("n_changed_features"),
        "mean_knn_distance": mean("knn_distance"),
        "mean_knn_distance_relative": mean("knn_distance_relative"),
        "lof_inlier_rate": mean("lof_inlier", inlier),
        "mean_lof_score": mean("lof_score"),
        "mean_ynn": mean("ynn"),
    }
