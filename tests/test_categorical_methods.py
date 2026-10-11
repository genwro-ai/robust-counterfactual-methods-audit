import numpy as np
import pandas as pd
import pytest
import torch

from robustness_benchmark.core.classifiers import LogisticClassifier, TorchBinaryModel
from robustness_benchmark.core.schema import FeatureSchema
from robustness_benchmark.evaluation.plausibility import (
    GowerDistance,
    PlausibilityReference,
    feature_range,
)
from robustness_benchmark.methods import registry
from robustness_benchmark.methods.kdtree import KDTreeNNCE
from robustness_benchmark.methods.mce import MCE
from robustness_benchmark.methods.registry import generate_counterfactuals, make_task

NUMERIC = ("age", "income", "hours")
JOBS = ("job_a", "job_b", "job_c")
SEXES = ("sex_f", "sex_m")
ATOL = 1e-5


def mixed_data(n: int = 240) -> tuple[pd.DataFrame, pd.Series, FeatureSchema]:
    """Integer age, numeric income, mutable job; immutable hours and sex.

    Immutable values are discrete so that neighbour methods, which need an
    exact match, have candidates.
    """

    rng = np.random.default_rng(0)
    raw = pd.DataFrame(
        {
            "age": rng.integers(20, 70, n).astype(float),
            "income": rng.normal(50, 15, n),
            "hours": rng.choice([20.0, 30.0, 40.0], n),
        }
    )
    job, sex = rng.integers(0, 3, n), rng.integers(0, 2, n)
    score = (
        0.08 * (raw["age"] - 45)
        + 0.1 * (raw["income"] - 50)
        + 0.04 * (raw["hours"] - 30)
        + np.array([-1.5, 0.0, 1.5])[job]
        + 0.5 * sex
    )
    y = pd.Series((score + rng.normal(0, 0.5, n) > 0).astype(int))
    scale = {name: (raw[name].mean(), raw[name].std()) for name in NUMERIC}
    X = pd.concat(
        [
            (raw - raw.mean()) / raw.std(),
            pd.DataFrame(np.eye(3)[job], columns=JOBS),
            pd.DataFrame(np.eye(2)[sex], columns=SEXES),
        ],
        axis=1,
    )
    schema = FeatureSchema(
        columns=tuple(X.columns),
        categorical={"job": JOBS, "sex": SEXES},
        numeric_scale=scale,
        integer=("age", "hours"),
        immutable=("hours", "sex"),
    )
    return X, y, schema


def mlp(X, y, seed: int = 1) -> TorchBinaryModel:
    model = TorchBinaryModel(input_dim=X.shape[1], hidden_dim=(8,), seed=seed)
    model.train(X, y, X_val=X, y_val=y, learning_rate=1e-2, epochs=100)
    return model


def adverse(model, X: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    probability = model.predict(X).iloc[:, 0].to_numpy()
    return X[probability < 0.5].iloc[:n].reset_index(drop=True)


def assert_valid(ce: np.ndarray, factual: np.ndarray, schema: FeatureSchema) -> None:
    """One-hot groups, immutables and integer features of a valid encoding."""

    for group in schema.one_hot_groups().values():
        np.testing.assert_allclose(np.sort(ce[group]), [0.0] * (len(group) - 1) + [1.0])
    immutable = schema.immutable_columns()
    np.testing.assert_allclose(ce[immutable], factual[immutable], atol=ATOL)
    for name in schema.integer:
        (index,) = schema.indices([name])
        mean, scale = schema.numeric_scale[name]
        raw = ce[index] * scale + mean
        assert raw == pytest.approx(round(raw), abs=1e-4)


@pytest.fixture(scope="module")
def mixed_setting():
    X, y, schema = mixed_data()
    model = mlp(X, y)
    ensemble = [mlp(X, y, seed=10 + member) for member in range(4)]
    return X, y, schema, model, ensemble


FAST_CONFIGS = {
    "kdtree": {},
    "wachter": {},
    "rnce": {"delta": 0.005, "bias_delta": 0.005},
    "apas": {"delta": 0.01, "wall_time_seconds": 10.0},
    "rbr": {"num_counterfactuals": 20, "num_samples": 50, "maximum_iterations": 50},
    "roar_lime": {"num_samples": 500, "maximum_iterations": 500, "lr": 0.01},
    "robx": {"variance": 0.01, "tau": 0.5, "stability_samples": 100},
    "betarce": {
        "generation_ensemble_n": 4,
        "delta": 0.5,
        "base_gs_max_iter": 50,
        "gs_max_iter": 50,
    },
    "trex": {"stability_samples": 100, "max_steps": 20},
    "growing_spheres": {},
    "gs_margin_80": {},
}
# Methods whose own output, before the registry projects it, is always a valid
# encoding; Wachter returns one only once its projection is valid.
VALID_BEFORE_PROJECTION = {
    "kdtree",
    "apas",
    "betarce",
    "growing_spheres",
    "gs_margin_80",
}
# Faithful to their original code, these search one-hot and integer columns as
# continuous (RNCE in its line search) and return the relaxed point; only the
# registry's projection makes their output valid. All but RobX keep immutables.
RELAXED_SEARCH = {"robx", "trex", "roar_lime", "rbr", "rnce"}
RELAXED_IMMUTABLES = {"robx"}


class RecordingGenerator:
    """Keep each raw method output to inspect it before projection."""

    def __init__(self, generator):
        self.generator = generator
        self.raw: list[tuple[pd.Series, pd.DataFrame]] = []

    @property
    def last_metadata(self):
        return getattr(self.generator, "last_metadata", {})

    def generate_for_instance(self, factual, **kwargs):
        result = self.generator.generate_for_instance(factual, **kwargs)
        self.raw.append((factual, result))
        return result


def run_recorded(monkeypatch, method, task, factuals, ensemble, config):
    create = registry.create_generator
    recorders: list[RecordingGenerator] = []

    def recording_create(*args, **kwargs):
        generator, method_kwargs = create(*args, **kwargs)
        recorders.append(RecordingGenerator(generator))
        return recorders[-1], method_kwargs

    monkeypatch.setattr(registry, "create_generator", recording_create)
    counterfactuals, rows = generate_counterfactuals(
        method,
        task,
        factuals,
        generation_models=ensemble if method == "betarce" else None,
        method_config=config,
    )
    return counterfactuals, rows, recorders[0].raw


@pytest.mark.parametrize("method", sorted(FAST_CONFIGS))
def test_mlp_methods_return_valid_encodings(monkeypatch, mixed_setting, method):
    X, y, schema, model, ensemble = mixed_setting
    task = make_task(model, X, y, seed=3, schema=schema)
    factuals = adverse(model, X)

    counterfactuals, rows, raw = run_recorded(
        monkeypatch, method, task, factuals, ensemble, FAST_CONFIGS[method]
    )

    assert any(row["base_valid"] for row in rows), [
        r["generation_status"] for r in rows
    ]
    for (_, factual), (_, ce), row in zip(
        factuals.iterrows(), counterfactuals.iterrows(), rows, strict=True
    ):
        if row["generation_status"] in {"success", "base_invalid"}:
            assert_valid(ce.to_numpy(), factual.to_numpy(), schema)
        if row["base_valid"] and method not in RELAXED_SEARCH:
            # Validity was judged on the returned point, not on a relaxed one.
            assert row["projection_changed"] is False
    if method in RELAXED_SEARCH - {"rnce"}:
        # The method returns its relaxed point; only the registry projects it.
        # (RNCE's line points here happen to keep the job and whole years.)
        assert any(row["projection_changed"] for row in rows)
    if method in RELAXED_IMMUTABLES:
        return
    immutable = schema.immutable_columns()
    for factual, result in raw:
        if result.empty or not np.isfinite(result.to_numpy()).all():
            continue
        values = result.iloc[0].to_numpy(dtype=float)
        np.testing.assert_allclose(
            values[immutable], factual.to_numpy()[immutable], atol=ATOL
        )
        if method in VALID_BEFORE_PROJECTION:
            assert_valid(values, factual.to_numpy(), schema)


@pytest.mark.parametrize("method", ["trex", "roar_lime", "rbr"])
def test_reverted_methods_return_their_relaxed_point(
    monkeypatch, mixed_setting, method
):
    X, y, schema, model, ensemble = mixed_setting
    task = make_task(model, X, y, seed=3, schema=schema)
    factuals = adverse(model, X)

    counterfactuals, rows, raw = run_recorded(
        monkeypatch, method, task, factuals, ensemble, FAST_CONFIGS[method]
    )

    relaxed = 0
    for (factual, result), (_, ce), row in zip(
        raw, counterfactuals.iterrows(), rows, strict=True
    ):
        if row["generation_status"] not in {"success", "base_invalid"}:
            continue
        values = result.iloc[0].to_numpy(dtype=float)
        projected = schema.project(values, factual.to_numpy())
        # The registry, not the method, projects the output.
        np.testing.assert_allclose(ce.to_numpy(), projected)
        relaxed += not np.allclose(values, projected)
    assert relaxed


@pytest.mark.parametrize("method", ["roar", "wachter", "growing_spheres"])
def test_logistic_methods_return_valid_encodings(mixed_setting, method):
    X, y, schema, _, _ = mixed_setting
    model = LogisticClassifier(input_dim=X.shape[1], seed=1)
    model.train(X, y, X_val=X, y_val=y, learning_rate=1e-2)
    factuals = adverse(model, X)
    config = {"lr": 0.01, "maximum_iterations": 2_000} if method == "roar" else {}

    counterfactuals, rows = generate_counterfactuals(
        method,
        make_task(model, X, y, seed=3, schema=schema),
        factuals,
        method_config=config,
    )

    assert any(row["base_valid"] for row in rows)
    for (_, factual), (_, ce), row in zip(
        factuals.iterrows(), counterfactuals.iterrows(), rows, strict=True
    ):
        if row["base_valid"]:
            assert_valid(ce.to_numpy(), factual.to_numpy(), schema)
            if method != "roar":
                assert row["projection_changed"] is False
    if method == "roar":
        # ROAR returns its relaxed point; only the registry projects it.
        assert any(row["projection_changed"] for row in rows)


def test_wachter_judges_validity_after_rounding():
    # P(y=1) passes 0.5 at age 10.3: a search that stops just past the
    # boundary rounds back to the adverse age 10, so it must reach 11.
    schema = FeatureSchema(
        columns=("age",), numeric_scale={"age": (0.0, 1.0)}, integer=("age",)
    )
    X = pd.DataFrame({"age": np.arange(21.0)})
    model = LogisticClassifier(input_dim=1)
    with torch.no_grad():
        model.get_torch_model()[0].weight.fill_(5.0)
        model.get_torch_model()[0].bias.fill_(-51.5)
    task = make_task(model, X, (X["age"] > 10.3).astype(int), seed=0, schema=schema)

    counterfactuals, (row,) = generate_counterfactuals("wachter", task, X.iloc[[5]])

    assert row["generation_status"] == "success"
    assert row["projection_changed"] is False
    assert counterfactuals.iloc[0, 0] >= 11.0


def milp_solution(job_c_weight: float) -> tuple[np.ndarray, TorchBinaryModel]:
    """MCE's own solution for logit = relu(age + 10 hours + w job_c + 5) - 6.

    The factual (age 0, hours 0, job a) has logit -1. Hours is immutable, age
    an integer in steps of 0.1 (scale 10 years), job a mutable category.
    """

    schema = FeatureSchema(
        columns=("age", "hours", "job_a", "job_b", "job_c"),
        categorical={"job": ("job_a", "job_b", "job_c")},
        numeric_scale={"age": (40.0, 10.0), "hours": (30.0, 5.0)},
        integer=("age",),
        immutable=("hours",),
    )
    model = TorchBinaryModel(input_dim=5, hidden_dim=(1,), seed=0)
    network = model.get_torch_model()
    with torch.no_grad():
        network[0].weight.copy_(torch.tensor([[1.0, 10.0, 0.0, 0.0, job_c_weight]]))
        network[0].bias.fill_(5.0)
        network[2].weight.fill_(1.0)
        network[2].bias.fill_(-6.0)
    X = pd.DataFrame(
        [[-2, -1, 1, 0, 0], [2, 1, 0, 1, 0], [0, 0, 0, 0, 1]],
        columns=schema.columns,
        dtype=float,
    )
    task = make_task(model, X, pd.Series([0, 1, 1]), seed=0, schema=schema)
    factual = pd.Series([0.0, 0.0, 1.0, 0.0, 0.0], index=schema.columns)
    mce = MCE(task)

    mce.generate_for_instance(factual, neg_value=0)

    # The solver's own values, not the projected output.
    solution = np.array([mce.opt.inputNodes[f"v_0_{i}"].X for i in range(5)])
    return solution, model


def test_milp_encodes_groups_immutables_and_integers():
    # Unconstrained, the cheapest L1 fix is hours +0.1 (immutable); job c alone
    # is not enough, so the valid optimum is age +1.1, i.e. 11 whole years.
    solution, model = milp_solution(job_c_weight=0.5)

    np.testing.assert_allclose(solution, [1.1, 0.0, 1.0, 0.0, 0.0], atol=1e-6)
    assert model.predict_single(solution) == 1


def test_milp_category_switch_costs_one():
    # Job c alone now suffices. It costs 1 (the largest one-hot change), less
    # than 11 years at 1.1; at the summed L1 cost of 2, age would win.
    solution, model = milp_solution(job_c_weight=3.0)

    np.testing.assert_allclose(solution, [0.0, 0.0, 0.0, 0.0, 1.0], atol=1e-6)
    assert model.predict_single(solution) == 1


def test_kdtree_keeps_the_factuals_immutables():
    schema = FeatureSchema(
        columns=("a", "sex_f", "sex_m"),
        categorical={"sex": ("sex_f", "sex_m")},
        immutable=("sex",),
    )
    X = pd.DataFrame(
        {"a": [1.0, 3.0, -1.0], "sex_f": [0.0, 1.0, 1.0], "sex_m": [1.0, 0.0, 0.0]}
    )
    model = LogisticClassifier(input_dim=3)
    linear = model.get_torch_model()[0]
    with torch.no_grad():
        linear.weight.copy_(torch.tensor([[5.0, 0.0, 0.0]]))
        linear.bias.zero_()
    task = make_task(model, X, pd.Series([1, 1, 0]), seed=0, schema=schema)
    generator = KDTreeNNCE(task)

    # The nearest positive point (a=1) is male; it takes the factual's sex and
    # stays positive, so the female factual gets a=1 rather than a=3.
    female = pd.Series({"a": -1.0, "sex_f": 1.0, "sex_m": 0.0})
    ce = generator.generate_for_instance(female).iloc[0]

    assert ce.tolist() == [1.0, 1.0, 0.0]


def test_mixed_distance_and_sparsity_on_a_hand_made_example():
    schema = FeatureSchema(
        columns=("age", "job_a", "job_b", "job_c", "sex_f", "sex_m"),
        categorical={"job": ("job_a", "job_b", "job_c"), "sex": ("sex_f", "sex_m")},
    )
    X = pd.DataFrame(
        [
            [0.0, 1, 0, 0, 1, 0],
            [1.0, 0, 1, 0, 0, 1],
            [2.0, 0, 0, 1, 1, 0],
            [3.0, 1, 0, 0, 0, 1],
            [4.0, 0, 1, 0, 1, 0],
        ],
        columns=schema.columns,
    )
    distance = GowerDistance(X, schema)
    age_range = feature_range(X)["age"]
    factual = np.array([1.0, 1, 0, 0, 1, 0])
    # Age moves by half its range and job changes; sex stays.
    counterfactual = np.array([1.0 + age_range / 2, 0, 0, 1, 1, 0])

    assert age_range == 4.0
    assert distance(factual, counterfactual) == pytest.approx((0.5 + 1.0 + 0.0) / 3)
    assert distance.n_changed(factual, counterfactual) == 2

    class AllTarget:
        def predict(self, values):
            return pd.DataFrame(np.ones(len(np.asarray(values))))

    reference = PlausibilityReference(X, AllTarget(), schema=schema, k=1)
    (row,) = reference.evaluate(
        pd.DataFrame([factual], columns=X.columns),
        pd.DataFrame([counterfactual], columns=X.columns),
        [True],
    )
    assert row["n_changed_features"] == 2
    assert row["fraction_changed_features"] == pytest.approx(2 / 3)
    # Nearest training point is row 2: same job and sex, age 2 versus 3.
    assert row["knn_distance"] == pytest.approx(0.25 / 3)


def test_numeric_data_outputs_match_without_and_with_identity_schema():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(120, 4)), columns=list("abcd"))
    y = pd.Series((X["a"] - 0.5 * X["b"] + 0.3 * X["c"] > 0).astype(int))
    model = mlp(X, y)
    logistic = LogisticClassifier(input_dim=X.shape[1], seed=1)
    logistic.train(X, y, X_val=X, y_val=y, learning_rate=1e-2)
    configs = {**FAST_CONFIGS, "roar": {"lr": 0.01, "maximum_iterations": 2_000}}
    identity = FeatureSchema(
        columns=tuple(X.columns),
        numeric_scale={name: (0.0, 1.0) for name in X.columns},
    )
    scale = feature_range(X.astype(np.float32)).to_numpy()

    methods = ("kdtree", "wachter", "growing_spheres", "robx", "rbr", "trex")
    for method in (*methods, "roar_lime", "roar"):
        base = logistic if method == "roar" else model
        factuals = adverse(base, X)
        outputs = [
            generate_counterfactuals(
                method,
                make_task(base, X, y, seed=3, schema=schema),
                factuals,
                method_config=configs[method],
            )
            for schema in (None, identity)
        ]
        (plain, plain_rows), (schematic, schematic_rows) = outputs
        np.testing.assert_array_equal(plain.to_numpy(), schematic.to_numpy())
        for row, other, (_, ce), (_, factual) in zip(
            plain_rows,
            schematic_rows,
            plain.iterrows(),
            factuals.iterrows(),
            strict=True,
        ):
            assert row["projection_changed"] is False
            assert row["gower_distance"] == other["gower_distance"]
            if row["base_valid"]:
                # On numeric data Gower is the mean range-scaled L1 over columns.
                legacy = np.nanmean(np.abs(ce.to_numpy() - factual.to_numpy()) / scale)
                assert row["gower_distance"] == legacy


def test_validity_before_projection_is_recorded(mixed_setting):
    X, y, schema, model, _ = mixed_setting
    task = make_task(model, X, y, seed=3, schema=schema)

    _, rows = generate_counterfactuals(
        "robx", task, adverse(model, X), method_config=FAST_CONFIGS["robx"]
    )

    generated = [
        row for row in rows if row["generation_status"] in {"success", "base_invalid"}
    ]
    assert generated
    assert all(
        isinstance(row["base_valid_before_projection"], bool) for row in generated
    )
    assert all(row["projection_changed"] for row in generated)
