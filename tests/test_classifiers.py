import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.linear_model import LogisticRegression

from robustness_benchmark.core.classifiers import (
    DIFFERENTIABLE,
    LINEAR,
    TORCH_NETWORK,
    LogisticClassifier,
    TorchBinaryModel,
    XGBoostClassifier,
)
from robustness_benchmark.core.provenance import (
    load_model_checkpoint,
    save_model_checkpoint,
)
from robustness_benchmark.methods.configuration import METHODS, unsupported_methods
from robustness_benchmark.methods.registry import (
    UnsupportedModelError,
    create_generator,
    make_task,
)


def linear_data(n: int = 200, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=["a", "b", "c"])
    y = pd.Series((X["a"] - 0.5 * X["b"] > 0).astype(int), name="target")
    return X, y


def trained(model_class, **kwargs):
    X, y = linear_data()
    model = model_class(input_dim=3, seed=7, **kwargs)
    model.train(X, y, X_val=X, y_val=y)
    return model, X, y


def test_logistic_matches_scikit_learn_probabilities():
    model, X, y = trained(LogisticClassifier, C=0.5)
    reference = LogisticRegression(C=0.5, max_iter=5_000).fit(X.to_numpy(), y)

    np.testing.assert_allclose(
        model.predict(X).iloc[:, 0].to_numpy(),
        reference.predict_proba(X.to_numpy())[:, 1],
        atol=1e-5,
    )
    assert model.hidden_dim == []
    assert model.model_type == "logistic"


def test_logistic_torch_view_is_differentiable():
    model, X, _ = trained(LogisticClassifier)
    inputs = torch.tensor(X.to_numpy()[:5], dtype=torch.float32, requires_grad=True)

    model.predict_proba_tensor(inputs).sum().backward()

    assert inputs.grad is not None
    assert torch.count_nonzero(inputs.grad) > 0


@pytest.mark.parametrize("l1_ratio", [1.0, 0.5])
def test_logistic_sparse_penalties_fit(l1_ratio):
    model, X, y = trained(LogisticClassifier, C=1.0, l1_ratio=l1_ratio)

    assert model.evaluate(X, y) > 0.9


def test_xgboost_predicts_probabilities_and_seed_matters():
    model, X, y = trained(XGBoostClassifier, n_estimators=20, max_depth=2)
    other = XGBoostClassifier(input_dim=3, seed=8, n_estimators=20, max_depth=2)
    other.train(X, y)

    probability = model.predict(X).iloc[:, 0].to_numpy()
    assert probability.shape == (len(X),)
    assert np.all((probability > 0) & (probability < 1))
    assert model.evaluate(X, y) > 0.9
    assert model.predict_single(X.iloc[0]) in {0, 1}
    assert not np.allclose(probability, other.predict(X).iloc[:, 0].to_numpy())


def test_xgboost_leaf_offsets_shift_margins_exactly():
    model, X, _ = trained(XGBoostClassifier, n_estimators=5, max_depth=2)
    offsets = np.linspace(-0.05, 0.05, len(model.leaf_values()))

    shifted = model.with_leaf_offsets(offsets)

    np.testing.assert_allclose(
        shifted.leaf_values() - model.leaf_values(), offsets, atol=1e-6
    )
    assert not np.allclose(shifted.predict(X).to_numpy(), model.predict(X).to_numpy())


def test_xgboost_continued_training_adds_trees_without_touching_base():
    model, X, y = trained(XGBoostClassifier, n_estimators=10, max_depth=2)
    before = model.predict(X).to_numpy()

    updated = model.continue_training(X.iloc[:50], y.iloc[:50], rounds=3)

    assert model.num_trees == 10
    assert updated.num_trees == 13
    np.testing.assert_array_equal(model.predict(X).to_numpy(), before)


@pytest.mark.parametrize(
    ("model_class", "kwargs"),
    [
        (TorchBinaryModel, {"hidden_dim": (4,)}),
        (LogisticClassifier, {"C": 0.3, "class_weight": "balanced"}),
        (XGBoostClassifier, {"n_estimators": 5, "max_depth": 2}),
    ],
)
def test_checkpoint_round_trip_for_every_model_class(tmp_path, model_class, kwargs):
    model, X, _ = trained(model_class, **kwargs)
    path = tmp_path / "model.pt"

    save_model_checkpoint(model, "model:1", path, {"role": "base"})
    restored, payload = load_model_checkpoint(path)

    assert type(restored) is model_class
    assert payload["model_type"] == model_class.model_type
    assert payload["metadata"] == {"role": "base"}
    np.testing.assert_array_equal(
        restored.predict(X).to_numpy(), model.predict(X).to_numpy()
    )


def test_legacy_mlp_checkpoint_without_model_type_still_loads(tmp_path):
    model = TorchBinaryModel(input_dim=2, hidden_dim=(3,), seed=12)
    path = tmp_path / "legacy.pt"
    payload = {
        "model_id": "legacy",
        "metadata": {},
        **model.checkpoint_payload(),
    }
    torch.save(payload, path)

    restored, _ = load_model_checkpoint(path)

    X = pd.DataFrame({"a": [0.0, 1.0], "b": [1.0, 0.0]})
    assert type(restored) is TorchBinaryModel
    np.testing.assert_array_equal(
        restored.predict(X).to_numpy(), model.predict(X).to_numpy()
    )


def test_capabilities_by_model_class():
    assert TorchBinaryModel.capabilities == {DIFFERENTIABLE, TORCH_NETWORK}
    assert LogisticClassifier.capabilities == {DIFFERENTIABLE, TORCH_NETWORK, LINEAR}
    assert XGBoostClassifier.capabilities == frozenset()


def test_method_support_matrix():
    methods = list(METHODS)

    assert set(unsupported_methods(methods, TorchBinaryModel.capabilities)) == {"roar"}
    assert unsupported_methods(methods, LogisticClassifier.capabilities) == {}
    assert set(unsupported_methods(methods, XGBoostClassifier.capabilities)) == {
        "wachter",
        "apas",
        "rnce",
        "trex",
        "roar",
    }


def test_registry_refuses_unsupported_method_model_pair():
    model, X, y = trained(XGBoostClassifier, n_estimators=5, max_depth=2)
    task = make_task(model, X, y, seed=0)

    with pytest.raises(UnsupportedModelError, match="wachter cannot explain"):
        create_generator("wachter", task)
    generator, _ = create_generator("kdtree", task)
    assert generator is not None
