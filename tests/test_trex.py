import numpy as np
import pandas as pd
import pytest
import torch

from robustness_benchmark.core.classifiers import (
    LogisticClassifier,
    MLPClassifier,
    XGBoostClassifier,
)
from robustness_benchmark.core.protocol import STABILITY_TAU_SEED_OFFSET
from robustness_benchmark.methods.configuration import LOCKED_CONFIGS
from robustness_benchmark.methods.registry import (
    UnsupportedModelError,
    create_generator,
    generate_counterfactuals,
    make_task,
)
from robustness_benchmark.methods.stability import (
    counterfactual_stability,
    tau_from_stability_quantile,
    training_target_stability_scores,
)
from robustness_benchmark.methods.trex import target_stability


def linear_data(n: int = 200, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=["a", "b", "c"])
    y = pd.Series((X["a"] - 0.5 * X["b"] > 0).astype(int), name="target")
    return X, y


def trained(model_class, **kwargs):
    X, y = linear_data()
    model = model_class(input_dim=3, seed=7, **kwargs)
    model.train(X, y, X_val=X, y_val=y, learning_rate=1e-2)
    return model, X, y


def adverse_factuals(model, X: pd.DataFrame, n: int = 4) -> pd.DataFrame:
    return X[model.predict(X).iloc[:, 0].to_numpy() < 0.5].iloc[:n]


def target_probability(model):
    return lambda values: model.predict_proba(values).iloc[:, 1].to_numpy()


def test_stability_matches_robx_estimator_on_the_same_noise():
    model, X, _ = trained(MLPClassifier, hidden_dim=(8,))
    point = X[model.predict(X).iloc[:, 0].to_numpy() >= 0.5].iloc[0].to_numpy()
    noise = np.random.default_rng(3).standard_normal((1_000, 3)) * 0.1

    with torch.no_grad():
        ours = target_stability(
            model,
            torch.as_tensor(point, dtype=torch.float32),
            torch.as_tensor(noise, dtype=torch.float32),
        )
    robx = counterfactual_stability(
        point,
        target_probability(model),
        variance=0.01,
        N=1_000,
        rng=np.random.default_rng(3),
    )

    assert float(ours) == pytest.approx(robx, abs=1e-5)


@pytest.mark.parametrize(
    ("model_class", "kwargs"),
    [(MLPClassifier, {"hidden_dim": (8,)}), (LogisticClassifier, {})],
)
def test_trex_reaches_robx_calibrated_tau(model_class, kwargs):
    model, X, y = trained(model_class, **kwargs)
    task = make_task(model, X, y, seed=5)
    factuals = adverse_factuals(model, X)

    counterfactuals, rows = generate_counterfactuals(
        "trex", task, factuals, method_config=LOCKED_CONFIGS["trex"]
    )

    scores, points = training_target_stability_scores(
        X.to_numpy(), target_probability(model), seed=5 + STABILITY_TAU_SEED_OFFSET
    )
    tau, _ = tau_from_stability_quantile(
        scores, quantile=0.5, target_training_points=points
    )
    rng = np.random.default_rng(0)
    for counterfactual, row in zip(counterfactuals.to_numpy(), rows, strict=True):
        assert row["generation_status"] == "success"
        assert row["trex_status"] == "ok"
        assert row["trex_tau"] == tau
        assert row["trex_stability"] >= tau
        assert row["trex_steps"] > 0
        stability = counterfactual_stability(
            counterfactual, target_probability(model), 0.01, 1_000, rng=rng
        )
        assert stability >= tau - 0.01


def test_trex_is_refused_on_xgboost():
    model, X, y = trained(XGBoostClassifier, n_estimators=5, max_depth=2)

    with pytest.raises(UnsupportedModelError, match="trex cannot explain"):
        create_generator("trex", make_task(model, X, y, seed=0))
