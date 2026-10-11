import numpy as np
import pandas as pd
import torch

from robustness_benchmark.core.classifiers import TorchBinaryModel
from robustness_benchmark.core.schema import FeatureSchema
from robustness_benchmark.methods.rbr import LabelPredictor, RBRAdapter
from robustness_benchmark.methods.registry import generate_counterfactuals, make_task


class FixedRBR:
    feasible = True

    def __init__(self, candidate: np.ndarray):
        self.candidate = candidate
        self.arguments = None

    def fit_instance(self, *arguments):
        self.arguments = arguments
        return self.candidate


def linear_task():
    X = pd.DataFrame({"a": [-1.0, 0.0, 1.0], "b": [0.0, 0.0, 0.0]})
    y = pd.Series([0, 0, 1])
    model = TorchBinaryModel(input_dim=2, hidden_dim=(), seed=0)
    with torch.no_grad():
        linear = model.get_torch_model()[0]
        linear.weight.copy_(torch.tensor([[5.0, 0.0]]))
        linear.bias.copy_(torch.tensor([-1.0]))
    return X, model, make_task(model, X, y, seed=4)


def test_rbr_label_predictor_returns_classes():
    X, model, _ = linear_task()

    predicted = LabelPredictor(model).predict(X.to_numpy())

    assert predicted.tolist() == [0, 0, 1]
    assert predicted.dtype.kind in {"i", "u"}


def test_rbr_adapter_records_scale_and_author_settings():
    X, _, task = linear_task()
    fixed = FixedRBR(np.array([1.0, 0.0]))
    adapter = RBRAdapter(
        task,
        seed=4,
        num_counterfactuals=1_000,
        num_samples=200,
        perturb_radius_fraction=0.2,
        delta_plus=0.2,
        sigma=1.0,
        epsilon_op=0.5,
        epsilon_pe=0.5,
        maximum_iterations=500,
        generator=fixed,
    )

    result = adapter.generate_for_instance(X.iloc[0])

    assert not result.empty
    assert fixed.arguments is not None
    assert fixed.arguments[1] == 200
    assert fixed.arguments[2] == 0.4
    assert adapter.last_metadata["rbr_outcome"] == "success"
    assert adapter.last_metadata["rbr_epsilon_op"] == 0.5
    assert adapter.last_metadata["rbr_epsilon_pe"] == 0.5


def test_rbr_returns_no_counterfactual_when_the_loss_is_undefined():
    # The boundary point sits at a ~ 0.21 against a boundary at a = 0.2. A ball
    # of radius 4e-4 around it holds only favorable samples, so the loss has
    # no adverse term; the authors' code crashed reshaping the empty set.
    X, _, task = linear_task()

    _, rows = generate_counterfactuals(
        "rbr",
        task,
        X.iloc[[0]],
        method_config={"perturb_radius_fraction": 2e-4, "num_samples": 50},
    )

    assert rows[0]["generation_status"] == "no_counterfactual"
    assert rows[0]["rbr_outcome"] == "no_counterfactual"


def test_rbr_returns_no_counterfactual_without_a_boundary_point():
    # The only favorable training point differs in the immutable b; with the
    # factual's b it is adverse, so the line search finds no boundary.
    X = pd.DataFrame({"a": [-1.0, 0.0], "b": [0.0, 1.0]})
    model = TorchBinaryModel(input_dim=2, hidden_dim=(), seed=0)
    with torch.no_grad():
        linear = model.get_torch_model()[0]
        linear.weight.copy_(torch.tensor([[5.0, 10.0]]))
        linear.bias.copy_(torch.tensor([-1.0]))
    schema = FeatureSchema(columns=("a", "b"), immutable=("b",))
    task = make_task(model, X, pd.Series([0, 1]), seed=4, schema=schema)

    _, rows = generate_counterfactuals("rbr", task, X.iloc[[0]])

    assert rows[0]["generation_status"] == "no_counterfactual"
