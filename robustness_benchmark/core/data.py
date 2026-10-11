"""The frozen train/update/validation/test split of an encoded dataset."""

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from robustness_benchmark.core.datasets import load_raw_dataset
from robustness_benchmark.core.protocol import SPLIT_SEED, SPLIT_VERSION
from robustness_benchmark.core.schema import FeatureSchema


@dataclass(frozen=True)
class DatasetSplit:
    name: str
    source: str
    split_version: str
    favorable_source_label: str
    X_train: pd.DataFrame
    y_train: pd.Series
    X_update: pd.DataFrame
    y_update: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    adverse_label: int = 0
    favorable_label: int = 1
    schema: FeatureSchema | None = None

    @property
    def feature_schema(self) -> FeatureSchema:
        return self.schema or FeatureSchema.numeric_only(self.X_train.columns)

    def __post_init__(self) -> None:
        if (self.adverse_label, self.favorable_label) != (0, 1):
            raise ValueError(
                "The label contract is fixed at 0 = adverse, 1 = favorable; "
                "loaders must remap raw labels via datasets.encode_favorable"
            )
        for split_name, labels in (
            ("train", self.y_train),
            ("update", self.y_update),
            ("validation", self.y_val),
            ("test", self.y_test),
        ):
            values = set(pd.unique(labels))
            if not values <= {0, 1}:
                raise ValueError(
                    f"{split_name} labels violate the 0/1 contract: {sorted(values)}"
                )


def load_dataset(
    name: str, seed: int = SPLIT_SEED, split_version: str = SPLIT_VERSION
) -> DatasetSplit:
    """Load, encode and freeze one binary-classification benchmark dataset.

    Nominal features become one-hot 0/1 groups over every category in the
    loaded dataset, so all splits share columns. Only numeric columns are
    standardized, with statistics from the training split.
    """

    if split_version != SPLIT_VERSION:
        raise ValueError(f"Unknown split version {split_version!r}")

    raw = load_raw_dataset(name)
    X, categorical = _one_hot(raw.X)
    y = raw.y

    # Frozen protocol: 50% base training, 10% reserved update pool, 10%
    # validation, and 30% untouched test data.
    X_train, X_remainder, y_train, y_remainder = train_test_split(
        X, y, test_size=0.50, random_state=seed, stratify=y
    )
    X_update, X_val_test, y_update, y_val_test = train_test_split(
        X_remainder,
        y_remainder,
        test_size=0.80,
        random_state=seed,
        stratify=y_remainder,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_val_test,
        y_val_test,
        test_size=0.75,
        random_state=seed,
        stratify=y_val_test,
    )

    one_hot = {column for group in categorical.values() for column in group}
    numeric = [column for column in X.columns if column not in one_hot]
    scaler = StandardScaler().fit(X_train[numeric])

    def standardize(frame: pd.DataFrame) -> pd.DataFrame:
        scaled = frame.astype(float)
        scaled[numeric] = scaler.transform(frame[numeric])
        scaled.index.name = "dataset_row_id"
        return scaled

    return DatasetSplit(
        name=name,
        source=raw.source,
        split_version=split_version,
        favorable_source_label=raw.favorable_source_label,
        X_train=standardize(X_train),
        y_train=y_train.rename("target"),
        X_update=standardize(X_update),
        y_update=y_update.rename("target"),
        X_val=standardize(X_val),
        y_val=y_val.rename("target"),
        X_test=standardize(X_test),
        y_test=y_test.rename("target"),
        schema=FeatureSchema(
            columns=tuple(X.columns),
            categorical=categorical,
            numeric_scale={
                column: (float(mean), float(scale))
                for column, mean, scale in zip(
                    numeric, scaler.mean_, scaler.scale_, strict=True
                )
            },
            integer=raw.integer,
            immutable=raw.immutable,
        ),
    )


def _one_hot(X: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, tuple[str, ...]]]:
    """Replace each non-numeric column by ``<feature>=<category>`` 0/1 columns."""

    categorical = X.select_dtypes(exclude="number").columns
    parts = [X.drop(columns=categorical)]
    groups = {}
    for feature in categorical:
        dummies = pd.get_dummies(
            X[feature], prefix=feature, prefix_sep="=", dtype=float
        )
        groups[feature] = tuple(dummies.columns)
        parts.append(dummies)
    return pd.concat(parts, axis=1), groups
