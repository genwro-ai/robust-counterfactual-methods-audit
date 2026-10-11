import numpy as np
import pandas as pd
import pytest

from robustness_benchmark.core.data import load_dataset

MIXED_DATASETS = [
    # name, rows, columns, favorable rows, one-hot group sizes
    (
        "german_credit",
        1_000,
        58,
        700,
        {
            "status": 4,
            "credit_history": 5,
            "purpose": 10,
            "savings": 5,
            "personal_status_sex": 4,
            "other_debtors": 3,
            "property": 4,
            "other_installment_plans": 3,
            "housing": 3,
            "job": 4,
            "people_liable": 2,
            "telephone": 2,
            "foreign_worker": 2,
        },
    ),
    (
        "adult",
        10_000,
        48,
        2_478,
        {
            "workclass": 7,
            "marital-status": 7,
            "occupation": 14,
            "relationship": 6,
            "race": 5,
            "sex": 2,
            "native-country": 2,
        },
    ),
    (
        "taiwan_default",
        10_000,
        29,
        7_788,
        {"SEX": 2, "EDUCATION": 4, "MARRIAGE": 3},
    ),
]


@pytest.fixture(scope="module", params=MIXED_DATASETS, ids=lambda case: case[0])
def case(request):
    name, *expected = request.param
    return load_dataset(name, seed=2026), *expected


def _splits(data):
    return (data.X_train, data.X_update, data.X_val, data.X_test)


def _labels(data):
    return (data.y_train, data.y_update, data.y_val, data.y_test)


def test_mixed_dataset_follows_split_and_label_contract(case):
    data, rows, columns, favorable_n, _ = case

    assert [len(frame) for frame in _splits(data)] == [
        rows // 2,
        rows // 10,
        rows // 10,
        rows * 3 // 10,
    ]
    assert all(frame.shape[1] == columns for frame in _splits(data))
    assert sum(int(labels.sum()) for labels in _labels(data)) == favorable_n
    assert all(set(pd.unique(labels)) == {0, 1} for labels in _labels(data))
    indices = [set(frame.index) for frame in _splits(data)]
    assert sum(map(len, indices)) == len(set().union(*indices))


def test_mixed_dataset_schema_matches_encoding(case):
    data, _, _, _, group_sizes = case
    schema = data.schema

    assert schema.columns == tuple(data.X_train.columns)
    assert {feature: len(group) for feature, group in schema.categorical.items()} == (
        group_sizes
    )
    assert all(
        column.startswith(f"{feature}=")
        for feature, group in schema.categorical.items()
        for column in group
    )
    assert set(schema.immutable) <= set(schema.raw_features)
    assert set(schema.integer) == set(schema.numeric) == set(schema.numeric_scale)
    assert np.allclose(data.X_train[list(schema.numeric)].mean(), 0.0, atol=1e-6)


def test_every_one_hot_group_has_exactly_one_active_column(case):
    data = case[0]

    for frame in _splits(data):
        for group in data.schema.categorical.values():
            block = frame[list(group)].to_numpy()
            assert set(np.unique(block)) <= {0.0, 1.0}
            assert (block.sum(axis=1) == 1).all()


def test_integer_features_are_whole_numbers_in_raw_units(case):
    data = case[0]
    schema = data.schema

    for feature in schema.integer:
        mean, scale = schema.numeric_scale[feature]
        raw = data.X_test[feature] * scale + mean
        assert np.allclose(raw, raw.round(), atol=1e-6)


def test_subsample_is_fixed_across_split_seeds():
    first, other = load_dataset("adult", seed=2026), load_dataset("adult", seed=7)

    def rows(data):
        return set().union(*(frame.index for frame in _splits(data)))

    pd.testing.assert_frame_equal(first.X_test, load_dataset("adult", seed=2026).X_test)
    assert rows(first) == rows(other)
    assert set(first.X_train.index) != set(other.X_train.index)
