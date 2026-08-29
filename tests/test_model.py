import numpy as np
import pandas as pd

from robustness_benchmark.core.model import TorchBinaryModel
from robustness_benchmark.core.provenance import (
    file_sha256,
    load_model_checkpoint,
    save_model_checkpoint,
)
from robustness_benchmark.core.task import FrameDataset


def test_frame_dataset_preserves_contract():
    X = pd.DataFrame({"a": [0.0, 1.0], "b": [1.0, 0.0]})
    dataset = FrameDataset(X, pd.Series([0, 1]))

    assert dataset.X.shape == (2, 2)
    assert dataset.y.tolist() == [0, 1]
    assert dataset.data.columns.tolist() == ["a", "b", "target"]


def test_prediction_and_probability_labels_are_consistent():
    model = TorchBinaryModel(input_dim=2, hidden_dim=(), seed=0)
    linear = model.get_torch_model()[0]
    linear.weight.data[:] = 1.0
    linear.bias.data[:] = -1.0

    X = pd.DataFrame({"a": [0.0, 1.0], "b": [0.0, 1.0]})
    probability = model.predict(X).iloc[:, 0].to_numpy()

    assert np.allclose(probability, [1 / (1 + np.exp(1)), 1 / (1 + np.exp(-1))])
    assert model.predict_single(X.iloc[0]) == 0
    assert model.predict_single(X.iloc[1]) == 1
    assert np.allclose(model.predict_proba(X).sum(axis=1), 1.0)


def test_model_checkpoint_round_trip(tmp_path):
    model = TorchBinaryModel(input_dim=2, hidden_dim=(3,), seed=12)
    X = pd.DataFrame({"a": [0.0, 1.0], "b": [1.0, 0.0]})
    path = tmp_path / "variant.pt"

    save_model_checkpoint(model, "variant:12", path, {"role": "changed"})
    restored, payload = load_model_checkpoint(path)

    assert payload["model_id"] == "variant:12"
    assert payload["metadata"] == {"role": "changed"}
    assert np.array_equal(model.predict(X).to_numpy(), restored.predict(X).to_numpy())
    assert len(file_sha256(path)) == 64
