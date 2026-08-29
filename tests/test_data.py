import numpy as np
import pandas as pd
import pytest

from robustness_benchmark.core.data import (
    DatasetSplit,
    encode_favorable,
    load_dataset,
    select_adverse_indices,
)


def test_dataset_splits_are_disjoint_and_scaler_is_train_only():
    data = load_dataset("breast_cancer", seed=2026)
    index_sets = [
        set(data.X_train.index),
        set(data.X_update.index),
        set(data.X_val.index),
        set(data.X_test.index),
    ]

    assert sum(map(len, index_sets)) == 569
    assert all(
        index_sets[i].isdisjoint(index_sets[j])
        for i in range(4)
        for j in range(i + 1, 4)
    )
    assert [len(indices) for indices in index_sets] == [284, 57, 57, 171]
    assert np.allclose(data.X_train.mean(axis=0), 0.0, atol=1e-6)


def _split_with_labels(y: pd.Series, **overrides) -> DatasetSplit:
    frame = pd.DataFrame({"a": range(len(y))})
    fields = {
        "name": "test",
        "source": "test",
        "split_version": "behavior_v3",
        "favorable_source_label": "raw 1",
        "X_train": frame,
        "y_train": y,
        "X_update": frame,
        "y_update": y,
        "X_val": frame,
        "y_val": y,
        "X_test": frame,
        "y_test": y,
        **overrides,
    }
    return DatasetSplit(**fields)


def test_encode_favorable_maps_dataset_specific_raw_labels():
    y_raw = pd.Series(["diabetes", "healthy", "diabetes"])

    encoded = encode_favorable(y_raw, favorable_values=("healthy",))

    assert encoded.tolist() == [0, 1, 0]
    assert encoded.name == "target"


def test_encode_favorable_requires_both_classes():
    with pytest.raises(ValueError, match="both"):
        encode_favorable(pd.Series([1, 1]), favorable_values=(1,))


def test_dataset_split_rejects_labels_outside_contract():
    with pytest.raises(ValueError, match="contract"):
        _split_with_labels(pd.Series([0, 1, 2]))


def test_dataset_split_rejects_relabeled_contract_fields():
    with pytest.raises(ValueError, match="remap"):
        _split_with_labels(pd.Series([0, 1]), adverse_label=1, favorable_label=0)


def test_loaded_labels_follow_the_contract():
    data = load_dataset("breast_cancer", seed=2026)

    assert data.favorable_source_label == "benign (raw target 1)"
    assert set(pd.unique(data.y_train)) == {0, 1}


@pytest.mark.parametrize(
    ("name", "rows", "features", "favorable_n", "favorable_source_label"),
    [
        ("diabetes", 768, 8, 500, "no diabetes (raw Outcome 0)"),
        ("wine_quality", 6_497, 11, 4_113, "good quality (raw quality True)"),
        ("heloc", 8_291, 20, 4_040, "good credit risk (raw RiskPerformance Good)"),
    ],
)
def test_reference_dataset_loaders_follow_the_shared_contract(
    name, rows, features, favorable_n, favorable_source_label
):
    data = load_dataset(name, seed=2026)
    feature_splits = (
        data.X_train,
        data.X_update,
        data.X_val,
        data.X_test,
    )
    label_splits = (
        data.y_train,
        data.y_update,
        data.y_val,
        data.y_test,
    )

    assert sum(len(frame) for frame in feature_splits) == rows
    assert all(frame.shape[1] == features for frame in feature_splits)
    assert sum(int(labels.sum()) for labels in label_splits) == favorable_n
    assert all(set(pd.unique(labels)) == {0, 1} for labels in label_splits)
    assert data.favorable_source_label == favorable_source_label
    assert np.allclose(data.X_train.mean(axis=0), 0.0, atol=1e-6)
    assert all(
        set(feature_splits[i].index).isdisjoint(feature_splits[j].index)
        for i in range(len(feature_splits))
        for j in range(i + 1, len(feature_splits))
    )


def test_adverse_selection_is_seeded_and_caps_to_availability():
    probabilities = pd.Series(
        [0.1, 0.9, 0.2], index=pd.Index([10, 11, 12], name="dataset_row_id")
    )

    first = select_adverse_indices(probabilities, requested=50, seed=7)
    second = select_adverse_indices(probabilities, requested=50, seed=7)

    assert first.equals(second)
    assert set(first) == {10, 12}
    assert first.name == "dataset_row_id"
