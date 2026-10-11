import numpy as np
import pytest

from robustness_benchmark.core.data import load_dataset
from robustness_benchmark.core.schema import FeatureSchema
from robustness_benchmark.methods.registry import make_task

SCHEMA = FeatureSchema(
    columns=("age", "income", "job_a", "job_b", "job_c", "sex_f", "sex_m"),
    categorical={"job": ("job_a", "job_b", "job_c"), "sex": ("sex_f", "sex_m")},
    numeric_scale={"age": (40.0, 10.0), "income": (3.0, 2.0)},
    integer=("age",),
    immutable=("sex", "income"),
)


def test_schema_describes_raw_features_and_groups():
    assert SCHEMA.numeric == ("age", "income")
    assert SCHEMA.raw_features == ("age", "income", "job", "sex")
    assert SCHEMA.one_hot_groups() == {"job": [2, 3, 4], "sex": [5, 6]}
    assert sorted(SCHEMA.immutable_columns()) == [1, 5, 6]


def test_project_restores_a_valid_encoding():
    factual = np.array([0.0, 0.5, 1, 0, 0, 0, 1])
    candidate = np.array([0.33, 2.0, 0.2, 0.7, 0.4, 0.9, 0.1])

    projected = SCHEMA.project(candidate, factual)

    assert projected[1] == 0.5  # immutable numeric copied
    np.testing.assert_array_equal(projected[2:5], [0, 1, 0])
    np.testing.assert_array_equal(projected[5:7], [0, 1])  # immutable group copied
    assert projected[0] * 10 + 40 == pytest.approx(43.0)  # 43.3 rounded


def test_project_handles_batches():
    factual = np.array([0.0, 0.5, 1, 0, 0, 0, 1])
    batch = np.tile([0.33, 2.0, 0.2, 0.7, 0.4, 0.9, 0.1], (3, 1))

    projected = SCHEMA.project(batch, factual)

    assert projected.shape == (3, 7)
    assert (projected[:, 2:5].sum(axis=1) == 1).all()


def test_invalid_schema_is_rejected():
    with pytest.raises(ValueError):
        FeatureSchema(columns=("a",), categorical={"g": ("a", "missing")})
    with pytest.raises(ValueError):
        FeatureSchema(columns=("a",), immutable=("unknown",))


def test_numeric_datasets_get_an_identity_schema():
    data = load_dataset("diabetes", seed=2026)
    schema = data.feature_schema
    task = make_task(None, data.X_train, data.y_train, seed=0, schema=schema)

    assert schema.numeric == tuple(data.X_train.columns)
    assert task.schema is schema
    row = data.X_test.iloc[0].to_numpy()
    np.testing.assert_array_equal(schema.project(row, row), row)
    default = make_task(None, data.X_train, data.y_train, seed=0).schema
    assert default.numeric == tuple(data.X_train.columns)
