import numpy as np
import pandas as pd
import pytest
import torch

from robustness_benchmark.core.classifiers import (
    LogisticClassifier,
    MLPClassifier,
    XGBoostClassifier,
)
from robustness_benchmark.methods.configuration import (
    EXTENDED_APAS_WALL_TIME,
    EXTENDED_APAS_WALL_TIME_SECONDS,
    EXTENDED_GS_MAX_ITER,
    EXTENDED_GS_SEARCH,
    GROWING_SPHERES_TARGETS,
    GS_MAX_ITER_KEYS,
    GS_METHODS,
    LOCKED_CONFIGS,
    TUNING_GRIDS,
    default_config,
    locked_config,
)
from robustness_benchmark.methods.registry import (
    UnsupportedModelError,
    create_generator,
    generate_counterfactuals,
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
    model.train(X, y, X_val=X, y_val=y, learning_rate=1e-2)
    return model, X, y


def hand_made_logistic(weights, bias):
    model = LogisticClassifier(input_dim=len(weights))
    linear = model.get_torch_model()[0]
    with torch.no_grad():
        linear.weight.copy_(torch.tensor([weights], dtype=torch.float32))
        linear.bias.fill_(bias)
    return model


@pytest.mark.parametrize(
    ("model_class", "kwargs"),
    [
        (MLPClassifier, {"hidden_dim": (8,)}),
        (LogisticClassifier, {}),
        (XGBoostClassifier, {"n_estimators": 10, "max_depth": 2}),
    ],
)
def test_growing_spheres_runs_on_every_model_class(model_class, kwargs):
    model, X, y = trained(model_class, **kwargs)
    factuals = X[model.predict(X).iloc[:, 0].to_numpy() < 0.5].iloc[:3]

    _, rows = generate_counterfactuals(
        "growing_spheres",
        make_task(model, X, y, seed=0),
        factuals,
        method_config=LOCKED_CONFIGS["growing_spheres"],
    )

    assert [row["generation_status"] for row in rows] == ["success"] * 3


def test_margin_variants_meet_their_threshold_at_growing_distance():
    model = hand_made_logistic([1.0, 0.0], 0.0)
    X = pd.DataFrame({"a": [-1.0, 1.0], "b": [0.0, 0.0]})
    task = make_task(model, X, pd.Series([0, 1]), seed=0)
    factuals = pd.DataFrame({"a": [-1.0, -0.5, -2.0], "b": [0.3, -0.2, 0.0]})

    mean_distances = []
    for method, threshold in GROWING_SPHERES_TARGETS.items():
        counterfactuals, rows = generate_counterfactuals(
            method, task, factuals, method_config=LOCKED_CONFIGS[method]
        )
        probabilities = model.predict(counterfactuals).iloc[:, 0].to_numpy()
        assert all(row["base_valid"] for row in rows)
        assert np.all(probabilities >= threshold)
        mean_distances.append(
            np.linalg.norm(counterfactuals - factuals.to_numpy(), axis=1).mean()
        )

    assert mean_distances == sorted(mean_distances)
    assert mean_distances[-1] > mean_distances[0] + 1.0


@pytest.mark.parametrize(("dataset", "model_type"), sorted(EXTENDED_GS_SEARCH))
def test_only_listed_methods_get_the_extended_growing_spheres_search(
    dataset, model_type
):
    extended_methods = EXTENDED_GS_SEARCH[(dataset, model_type)]
    assert extended_methods <= GS_METHODS

    for method, locked in LOCKED_CONFIGS.items():
        config = locked_config(method, dataset, model_type)
        extended = GS_MAX_ITER_KEYS[method] if method in extended_methods else ()
        assert {key: config[key] for key in extended} == {
            key: EXTENDED_GS_MAX_ITER for key in extended
        }
        unchanged = {k: v for k, v in config.items() if k not in extended}
        if method == "apas":
            unchanged.pop("wall_time_seconds")
            locked = {k: v for k, v in locked.items() if k != "wall_time_seconds"}
        assert unchanged == {k: v for k, v in locked.items() if k not in extended}
    assert locked_config("robx_balanced", dataset, model_type)["max_iter"] == 100


def test_breast_cancer_extends_every_growing_spheres_search():
    for model_type in ("logistic", "mlp", "xgboost"):
        assert EXTENDED_GS_SEARCH[("breast_cancer", model_type)] == GS_METHODS
    assert GS_METHODS == {
        *GROWING_SPHERES_TARGETS,
        "betarce",
        "robx_balanced",
        "robx_robust",
    }


@pytest.mark.parametrize(
    ("dataset", "model_type"),
    [("heloc", "logistic"), ("diabetes", "mlp"), ("wine_quality", "xgboost")],
)
def test_other_datasets_keep_the_authors_settings(dataset, model_type):
    for method, locked in LOCKED_CONFIGS.items():
        assert locked_config(method, dataset, model_type) == locked
    for method, locked in LOCKED_CONFIGS.items():
        assert locked_config(method) == locked


def test_apas_gets_one_minute_only_where_it_timed_out():
    for dataset, model_type in EXTENDED_APAS_WALL_TIME:
        config = locked_config("apas", dataset, model_type)
        assert config["wall_time_seconds"] == EXTENDED_APAS_WALL_TIME_SECONDS == 60.0
    for dataset in ("adult", "diabetes"):
        assert locked_config("apas", dataset, "mlp") == LOCKED_CONFIGS["apas"]
    assert LOCKED_CONFIGS["apas"]["wall_time_seconds"] == 30.0


def test_a_run_that_never_reaches_the_authors_limit_is_unchanged():
    model, X, y = trained(LogisticClassifier)
    factuals = X[model.predict(X).iloc[:, 0].to_numpy() < 0.5].iloc[:10]
    task = make_task(model, X, y, seed=0)

    authors, authors_rows = generate_counterfactuals(
        "gs_margin_90", task, factuals, method_config=LOCKED_CONFIGS["gs_margin_90"]
    )
    extended, _ = generate_counterfactuals(
        "gs_margin_90",
        task,
        factuals,
        method_config={
            **LOCKED_CONFIGS["gs_margin_90"],
            "max_iter": EXTENDED_GS_MAX_ITER,
        },
    )

    assert [row["generation_status"] for row in authors_rows] == ["success"] * 10
    pd.testing.assert_frame_equal(authors, extended)


def test_extended_search_reaches_a_boundary_the_authors_limit_cannot():
    # 30 features with the boundary 8 away: 100 shells of width 0.1 almost
    # never sample that far out in this dimension, 300 shells do.
    n_features = 30
    model = hand_made_logistic([1.0] + [0.0] * (n_features - 1), -8.0)
    columns = [f"x{i}" for i in range(n_features)]
    X = pd.DataFrame(np.zeros((2, n_features)), columns=columns)
    task = make_task(model, X, pd.Series([0, 1]), seed=0)
    factuals = X.iloc[:2]

    _, authors = generate_counterfactuals(
        "growing_spheres",
        task,
        factuals,
        method_config=LOCKED_CONFIGS["growing_spheres"],
    )
    _, extended = generate_counterfactuals(
        "growing_spheres",
        task,
        factuals,
        method_config=locked_config("growing_spheres", "breast_cancer", "logistic"),
    )

    assert [row["generation_status"] for row in authors] == ["no_counterfactual"] * 2
    assert [row["generation_status"] for row in extended] == ["success"] * 2


def test_roar_runs_on_logistic_regression():
    model, X, y = trained(LogisticClassifier)
    factuals = X[model.predict(X).iloc[:, 0].to_numpy() < 0.5].iloc[:3]

    _, rows = generate_counterfactuals(
        "roar",
        make_task(model, X, y, seed=0),
        factuals,
        method_config=TUNING_GRIDS["roar"][0],
    )

    assert [row["generation_status"] for row in rows] == ["success"] * 3
    assert all(row["roar_delta"] == 0.1 for row in rows)


@pytest.mark.parametrize(
    ("model_class", "kwargs"),
    [
        (MLPClassifier, {"hidden_dim": (8,)}),
        (XGBoostClassifier, {"n_estimators": 5, "max_depth": 2}),
    ],
)
def test_roar_is_refused_on_non_linear_models(model_class, kwargs):
    model, X, y = trained(model_class, **kwargs)

    with pytest.raises(UnsupportedModelError, match="roar cannot explain"):
        create_generator("roar", make_task(model, X, y, seed=0))


def test_roar_is_valid_under_its_worst_case_coefficient_shift():
    weights, bias, delta = np.array([2.0, -1.0]), -0.5, 0.1
    model = hand_made_logistic(weights.tolist(), bias)
    task = make_task(model, pd.DataFrame({"a": [0.0], "b": [0.0]}), pd.Series([0]), 0)
    generator, kwargs = create_generator(
        "roar",
        task,
        method_config={
            **default_config("roar"),
            "delta": delta,
            "lambda_param": 0.01,
            "learning_rate": 0.02,
        },
    )

    counterfactual = (
        generator.generate_for_instance(pd.Series({"a": -1.0, "b": 0.5}), **kwargs)
        .iloc[0]
        .to_numpy()
    )

    worst_weights = weights - delta * np.sign(counterfactual)
    assert worst_weights @ counterfactual + bias - delta > 0.0
