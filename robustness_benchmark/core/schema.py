from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureSchema:
    """How model-input columns map back to raw features.

    Numeric columns are standardized with the training ``numeric_scale``
    (mean, std) pairs. A categorical feature is a group of one-hot columns
    holding 0/1, exactly one of which is 1. ``integer`` lists numeric features
    whose raw values must be whole numbers, and ``immutable`` lists raw
    features (numeric or categorical) a counterfactual must not change.
    """

    columns: tuple[str, ...]
    categorical: dict[str, tuple[str, ...]] = field(default_factory=dict)
    numeric_scale: dict[str, tuple[float, float]] = field(default_factory=dict)
    integer: tuple[str, ...] = ()
    immutable: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        one_hot = [column for group in self.categorical.values() for column in group]
        unknown = set(one_hot) - set(self.columns)
        if unknown or len(one_hot) != len(set(one_hot)):
            raise ValueError(f"Invalid one-hot groups: {sorted(unknown)}")
        features = set(self.numeric) | set(self.categorical)
        for name in (*self.integer, *self.immutable):
            if name not in features:
                raise ValueError(f"Unknown raw feature {name!r}")
        if set(self.integer) - set(self.numeric_scale):
            raise ValueError("integer features need a numeric_scale entry")

    @classmethod
    def numeric_only(cls, columns) -> "FeatureSchema":
        return cls(columns=tuple(columns))

    @property
    def numeric(self) -> tuple[str, ...]:
        one_hot = {column for group in self.categorical.values() for column in group}
        return tuple(column for column in self.columns if column not in one_hot)

    @property
    def raw_features(self) -> tuple[str, ...]:
        """Numeric columns, then categorical features, in column order."""

        seen: list[str] = []
        owner = {
            column: feature
            for feature, group in self.categorical.items()
            for column in group
        }
        for column in self.columns:
            feature = owner.get(column, column)
            if feature not in seen:
                seen.append(feature)
        return tuple(seen)

    def indices(self, columns) -> list[int]:
        position = {column: index for index, column in enumerate(self.columns)}
        return [position[column] for column in columns]

    def one_hot_groups(self) -> dict[str, list[int]]:
        """Column indices of each categorical feature's one-hot group."""

        return {
            feature: self.indices(group) for feature, group in self.categorical.items()
        }

    def immutable_columns(self) -> list[int]:
        """Indices of every column belonging to an immutable feature."""

        columns = [
            column
            for feature in self.immutable
            for column in self.categorical.get(feature, (feature,))
        ]
        return self.indices(columns)

    def project(self, candidate, factual) -> np.ndarray:
        """Return the nearest valid encoding of ``candidate`` for ``factual``.

        Immutable columns are copied from the factual, each one-hot group is
        set to its largest entry, and integer features are rounded in raw
        units. Works on one row or a 2-D batch of rows.
        """

        values = np.array(candidate, dtype=float, copy=True)
        rows = values.reshape(-1, len(self.columns))
        reference = np.broadcast_to(
            np.asarray(factual, dtype=float).reshape(-1, len(self.columns)),
            rows.shape,
        )
        immutable = self.immutable_columns()
        rows[:, immutable] = reference[:, immutable]
        for group in self.one_hot_groups().values():
            chosen = rows[:, group].argmax(axis=1)
            rows[:, group] = 0.0
            rows[np.arange(len(rows)), np.asarray(group)[chosen]] = 1.0
        for feature in self.integer:
            (index,) = self.indices([feature])
            mean, scale = self.numeric_scale[feature]
            rows[:, index] = np.round(rows[:, index] * scale + mean) - mean
            rows[:, index] /= scale
        return rows.reshape(values.shape)

    def project_frame(
        self, candidate: pd.DataFrame, factual: pd.Series
    ) -> pd.DataFrame:
        return pd.DataFrame(
            self.project(candidate.to_numpy(), factual.to_numpy()),
            columns=candidate.columns,
            index=candidate.index,
        )
