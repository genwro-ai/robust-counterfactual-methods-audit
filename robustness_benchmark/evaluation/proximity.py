"""Proximity and sparsity: how far a counterfactual moves from its factual."""

import numpy as np
import pandas as pd

from robustness_benchmark.core.schema import CHANGE_TOLERANCE, FeatureSchema


def feature_range(X: pd.DataFrame) -> pd.Series:
    """Per-feature training range, Gower's normaliser; 1 for constant features."""

    span = X.max(axis=0) - X.min(axis=0)
    return span.where(span > 1e-8, 1.0)


class GowerDistance:
    """Gower's (1971) distance: the mean over raw features of a per-feature cost.

    A numeric feature costs |delta| / R with R its training range; a
    categorical feature costs 1 when its category changes. One-hot columns are
    divided by 2, so on valid encodings a changed category (two flipped
    columns) costs exactly 1 and the distance is a scaled L1 that neighbour
    indices can use. Points outside the training range may cost more than 1
    per feature; the cost is not clipped.
    """

    def __init__(self, X_train: pd.DataFrame, schema: FeatureSchema | None = None):
        schema = schema or FeatureSchema.numeric_only(X_train.columns)
        self.scale = feature_range(X_train).to_numpy(copy=True)
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
        return ((changed > CHANGE_TOLERANCE) @ self.membership > 0).sum(axis=-1)
