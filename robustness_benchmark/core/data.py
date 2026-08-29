from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

HELOC_DROPPED_FEATURES = (
    "MSinceMostRecentDelq",
    "MSinceMostRecentInqexcl7days",
    "NetFractionInstallBurden",
)


def encode_favorable(
    y_raw: pd.Series, favorable_values: Collection[object]
) -> pd.Series:
    """Encode raw dataset labels into the fixed harness contract.

    The whole pipeline assumes 0 = adverse and 1 = favorable: adverse-factual
    selection, generation with ``neg_value=0``, base validity at class 1, and
    survival at probability >= 0.5. Raw label semantics are dataset dependent
    (e.g. Pima's raw 1 means diabetes, an adverse outcome), so every loader
    must remap through this helper rather than passing raw labels along.
    """

    encoded = y_raw.isin(list(favorable_values)).astype(int)
    if encoded.nunique() != 2:
        raise ValueError(
            "Encoded labels must contain both an adverse and a favorable class"
        )
    return encoded.rename("target")


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

    def __post_init__(self) -> None:
        if (self.adverse_label, self.favorable_label) != (0, 1):
            raise ValueError(
                "The label contract is fixed at 0 = adverse, 1 = favorable; "
                "loaders must remap raw labels via encode_favorable instead"
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


def _frame(values, columns: list[str], index: pd.Index) -> pd.DataFrame:
    frame = pd.DataFrame(values, columns=pd.Index(columns), index=index)
    frame.index.name = "dataset_row_id"
    return frame


def select_adverse_indices(
    probabilities: pd.Series,
    requested: int,
    seed: int,
    threshold: float = 0.5,
) -> pd.Index:
    """Select a seeded, capped sample of model-adverse row identifiers."""

    if requested < 1:
        raise ValueError("requested must be positive")
    available = probabilities.index[probabilities < threshold].to_numpy()
    size = min(requested, len(available))
    selected = pd.Index(
        pd.Series(available).sample(n=size, random_state=seed).to_numpy()
    )
    selected.name = probabilities.index.name
    return selected


def load_dataset(
    name: str, seed: int = 2026, split_version: str = "behavior_v3"
) -> DatasetSplit:
    """Load and freeze one numeric binary-classification benchmark dataset."""

    if split_version != "behavior_v3":
        raise ValueError(f"Unknown split version {split_version!r}")

    X, y, source, favorable_source_label = _load_raw_dataset(name)

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

    scaler = StandardScaler().fit(X_train)
    columns = X.columns.tolist()
    return DatasetSplit(
        name=name,
        source=source,
        split_version=split_version,
        favorable_source_label=favorable_source_label,
        X_train=_frame(scaler.transform(X_train), columns, X_train.index),
        y_train=y_train.rename("target"),
        X_update=_frame(scaler.transform(X_update), columns, X_update.index),
        y_update=y_update.rename("target"),
        X_val=_frame(scaler.transform(X_val), columns, X_val.index),
        y_val=y_val.rename("target"),
        X_test=_frame(scaler.transform(X_test), columns, X_test.index),
        y_test=y_test.rename("target"),
    )


def _load_raw_dataset(
    name: str,
) -> tuple[pd.DataFrame, pd.Series, str, str]:
    if name == "breast_cancer":
        raw = load_breast_cancer(as_frame=True)
        return (
            raw.data,
            encode_favorable(raw.target, favorable_values=(1,)),
            "sklearn.datasets.load_breast_cancer",
            "benign (raw target 1)",
        )

    specifications = {
        "diabetes": {
            "target": "Outcome",
            "favorable_values": (0,),
            "favorable_source_label": "no diabetes (raw Outcome 0)",
            "drop_negative_features": False,
        },
        "wine_quality": {
            "target": "quality",
            "favorable_values": (True,),
            "favorable_source_label": "good quality (raw quality True)",
            "drop_negative_features": False,
        },
        "heloc": {
            "target": "RiskPerformance",
            "favorable_values": ("Good",),
            "favorable_source_label": "good credit risk (raw RiskPerformance Good)",
            "drop_negative_features": True,
            "drop_columns_before_missing_rows": HELOC_DROPPED_FEATURES,
        },
    }
    if name not in specifications:
        supported = ["breast_cancer", *specifications]
        raise ValueError(f"Unknown dataset {name!r}; supported datasets: {supported}")

    path = DATA_DIR / f"{name}.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset {name!r} not found at {path}; expected the repository's "
            "data directory to contain it"
        )
    raw = pd.read_csv(path)
    specification = specifications[name]
    target = str(specification["target"])
    X = raw.drop(columns=[target]).apply(pd.to_numeric, errors="raise")
    dropped_columns = specification.get("drop_columns_before_missing_rows", ())
    if not isinstance(dropped_columns, Collection):
        raise TypeError("drop_columns_before_missing_rows must be a collection")
    X = X.drop(columns=list(dropped_columns))
    if bool(specification["drop_negative_features"]):
        # Follow the published RobX HELOC preprocessing: first remove the three
        # features with extensive missingness, then exclude rows that retain a
        # negative special-value sentinel. This leaves 8,291 complete rows and
        # 20 numeric features without introducing imputable CFE dimensions.
        X = X.loc[~X.lt(0).any(axis=1)]
    y_raw = raw.loc[X.index, target]
    favorable_values = specification["favorable_values"]
    if not isinstance(favorable_values, Collection):
        raise TypeError("favorable_values must be a collection")
    y = encode_favorable(y_raw, favorable_values=favorable_values)
    return (
        X,
        y,
        f"data/{name}.csv",
        str(specification["favorable_source_label"]),
    )
