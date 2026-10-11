from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from robustness_benchmark.core.schema import FeatureSchema

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

HELOC_DROPPED_FEATURES = (
    "MSinceMostRecentDelq",
    "MSinceMostRecentInqexcl7days",
    "NetFractionInstallBurden",
)

# Adult and Taiwan default are subsampled (stratified, fixed seed) before the
# split so that their runtimes stay comparable with HELOC.
SUBSAMPLE_SIZE = 10_000
SUBSAMPLE_SEED = 2026

# South German Credit nominal codes, labelled after the corrected code table
# (Grömping 2019, codetable.txt). Purpose code 7 (education) has no rows.
GERMAN_CREDIT_CATEGORIES = {
    "status": {1: "no_account", 2: "lt_0", 3: "0_to_200", 4: "ge_200_or_salary"},
    "credit_history": {
        0: "past_delay",
        1: "critical_or_credits_elsewhere",
        2: "none_or_all_paid",
        3: "existing_paid",
        4: "all_paid_at_bank",
    },
    "purpose": {
        0: "others",
        1: "car_new",
        2: "car_used",
        3: "furniture_equipment",
        4: "radio_television",
        5: "domestic_appliances",
        6: "repairs",
        7: "education",
        8: "vacation",
        9: "retraining",
        10: "business",
    },
    "savings": {
        1: "unknown_or_none",
        2: "lt_100",
        3: "100_to_500",
        4: "500_to_1000",
        5: "ge_1000",
    },
    "personal_status_sex": {
        1: "male_divorced_separated",
        2: "female_non_single_or_male_single",
        3: "male_married_widowed",
        4: "female_single",
    },
    "other_debtors": {1: "none", 2: "co_applicant", 3: "guarantor"},
    "property": {
        1: "unknown_or_none",
        2: "car_or_other",
        3: "savings_agreement_or_life_insurance",
        4: "real_estate",
    },
    "other_installment_plans": {1: "bank", 2: "stores", 3: "none"},
    "housing": {1: "for_free", 2: "rent", 3: "own"},
    "job": {
        1: "unskilled_non_resident",
        2: "unskilled_resident",
        3: "skilled",
        4: "highly_qualified",
    },
    "people_liable": {1: "3_or_more", 2: "0_to_2"},
    "telephone": {1: "no", 2: "yes"},
    "foreign_worker": {1: "yes", 2: "no"},
}

# Immutable features follow the recourse literature:
# - Age and sex: immutable in CARLA's data catalog (Pawelczyk et al. 2021) for
#   adult, compas and give-me-some-credit; race too for compas. MACE (Karimi et
#   al. 2020) lets age only increase on German credit, which the schema cannot
#   express, so age is fixed.
# - Marital status: not actionable (Ustun et al. 2019, whose credit experiments
#   also fix marital status and age on the Taiwan default data).
# - Nationality: MACE treats Adult's native country as a race proxy; foreign
#   worker status is the German credit counterpart.
# Per dataset:
# - german_credit: age; personal_status_sex (one code for sex and marital
#   status); foreign_worker.
# - adult: age, sex, race, native-country, marital-status; relationship, since
#   its levels (Husband, Wife, Own-child, ...) encode sex and marital status.
# - taiwan_default: AGE, SEX, MARRIAGE. EDUCATION stays mutable, as education
#   can be acquired.
IMMUTABLE_FEATURES = {
    "german_credit": ("age", "personal_status_sex", "foreign_worker"),
    "adult": (
        "age",
        "sex",
        "race",
        "native-country",
        "marital-status",
        "relationship",
    ),
    "taiwan_default": ("AGE", "SEX", "MARRIAGE"),
}


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
    schema: FeatureSchema | None = None

    @property
    def feature_schema(self) -> FeatureSchema:
        return self.schema or FeatureSchema.numeric_only(self.X_train.columns)

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


@dataclass(frozen=True)
class RawDataset:
    """Unencoded features (nominal ones as strings) and 0/1 labels."""

    X: pd.DataFrame
    y: pd.Series
    source: str
    favorable_source_label: str
    integer: tuple[str, ...] = ()
    immutable: tuple[str, ...] = ()


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
    """Load, encode and freeze one binary-classification benchmark dataset.

    Nominal features become one-hot 0/1 groups over every category in the
    loaded dataset, so all splits share columns. Only numeric columns are
    standardized, with statistics from the training split.
    """

    if split_version != "behavior_v3":
        raise ValueError(f"Unknown split version {split_version!r}")

    raw = _load_raw_dataset(name)
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


def _read_csv(name: str) -> pd.DataFrame:
    path = DATA_DIR / f"{name}.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset {name!r} not found at {path}; expected the repository's "
            "data directory to contain it"
        )
    return pd.read_csv(path)


def _mixed_dataset(
    name: str,
    X: pd.DataFrame,
    y_raw: pd.Series,
    favorable_values: Collection[object],
    favorable_source_label: str,
    subsample: bool = False,
) -> RawDataset:
    """Package a dataset with nominal features; its numeric ones are integers."""

    y = encode_favorable(y_raw, favorable_values=favorable_values)
    if subsample:
        kept, _ = train_test_split(
            X.index,
            train_size=SUBSAMPLE_SIZE,
            random_state=SUBSAMPLE_SEED,
            stratify=y,
        )
        kept = kept.sort_values()
        X, y = X.loc[kept], y.loc[kept]
    return RawDataset(
        X=X,
        y=y,
        source=f"data/{name}.csv",
        favorable_source_label=favorable_source_label,
        integer=tuple(X.select_dtypes("number").columns),
        immutable=IMMUTABLE_FEATURES[name],
    )


def _german_credit() -> RawDataset:
    """South German Credit, the corrected release of Statlog German Credit.

    duration, amount and age are quantitative. installment_rate,
    present_residence and number_credits are discretized quantities (Grömping
    2019) and employment_duration is ordered duration bins, so these keep
    their integer codes. The other codes, including the binarized
    people_liable, are nominal.
    """

    raw = _read_csv("german_credit")
    X = raw.drop(columns=["credit_risk"])
    for feature, labels in GERMAN_CREDIT_CATEGORIES.items():
        X[feature] = X[feature].map(labels)
    return _mixed_dataset(
        "german_credit",
        X,
        raw["credit_risk"],
        favorable_values=(1,),
        favorable_source_label="good credit risk (raw credit_risk 1)",
    )


def _adult() -> RawDataset:
    """UCI Adult, train and test files combined.

    Rows with a missing ("?") value are dropped and the test file's trailing
    "." is removed from labels. fnlwgt is a census sampling weight, not a
    personal attribute, and education duplicates the ordinal education-num.
    native-country is reduced to United-States vs other: 90% of rows are US
    and the other 40 countries are too sparse for one-hot columns.
    """

    raw = _read_csv("adult").replace("?", pd.NA).dropna()
    X = raw.drop(columns=["income", "fnlwgt", "education"])
    X["native-country"] = X["native-country"].where(
        X["native-country"] == "United-States", "other"
    )
    return _mixed_dataset(
        "adult",
        X,
        raw["income"].str.rstrip("."),
        favorable_values=(">50K",),
        favorable_source_label="income >50K (raw income >50K)",
        subsample=True,
    )


def _taiwan_default() -> RawDataset:
    """Default of Credit Card Clients (Yeh & Lien 2009).

    Undocumented EDUCATION codes (0, 5, 6) and MARRIAGE code 0 are merged into
    "other", as is common practice. PAY_0 and PAY_2..PAY_6 are ordered
    repayment statuses (-2 no consumption, -1 paid duly, 0 revolving credit,
    k months of delay) and stay integers.
    """

    raw = _read_csv("taiwan_default")
    target = "default payment next month"
    X = raw.drop(columns=["ID", target])
    X["SEX"] = X["SEX"].map({1: "male", 2: "female"})
    X["EDUCATION"] = (
        X["EDUCATION"]
        .map({1: "graduate_school", 2: "university", 3: "high_school"})
        .fillna("other")
    )
    X["MARRIAGE"] = X["MARRIAGE"].map({1: "married", 2: "single"}).fillna("other")
    return _mixed_dataset(
        "taiwan_default",
        X,
        raw[target],
        favorable_values=(0,),
        favorable_source_label="no default next month (raw default payment 0)",
        subsample=True,
    )


MIXED_DATASETS = {
    "german_credit": _german_credit,
    "adult": _adult,
    "taiwan_default": _taiwan_default,
}


def _load_raw_dataset(name: str) -> RawDataset:
    if name == "breast_cancer":
        raw = load_breast_cancer(as_frame=True)
        return RawDataset(
            X=raw.data,
            y=encode_favorable(raw.target, favorable_values=(1,)),
            source="sklearn.datasets.load_breast_cancer",
            favorable_source_label="benign (raw target 1)",
        )
    if name in MIXED_DATASETS:
        return MIXED_DATASETS[name]()

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
        supported = ["breast_cancer", *specifications, *MIXED_DATASETS]
        raise ValueError(f"Unknown dataset {name!r}; supported datasets: {supported}")

    raw = _read_csv(name)
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
    return RawDataset(
        X=X,
        y=encode_favorable(y_raw, favorable_values=favorable_values),
        source=f"data/{name}.csv",
        favorable_source_label=str(specification["favorable_source_label"]),
    )
