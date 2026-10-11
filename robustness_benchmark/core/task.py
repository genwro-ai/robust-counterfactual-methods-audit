import numpy as np
import pandas as pd

from robustness_benchmark.core.schema import FeatureSchema


class FrameDataset:
    """In-memory numeric binary-classification dataset."""

    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        seed: int | None = None,
        schema: FeatureSchema | None = None,
    ):
        self._seed = seed
        self._schema = schema or FeatureSchema.numeric_only(X.columns)
        self._X = X.reset_index(drop=True).astype(np.float32)
        self._y = pd.Series(y, name="target").reset_index(drop=True).astype(int)
        self._data = self._X.assign(target=self._y)

    @property
    def data(self) -> pd.DataFrame:
        return self._data

    @property
    def X(self) -> pd.DataFrame:
        return self._X

    @property
    def y(self) -> pd.Series:
        return self._y

    @property
    def seed(self) -> int | None:
        return self._seed

    @property
    def schema(self) -> FeatureSchema:
        return self._schema


class ClassificationTask:
    """Pair one prediction model with the data used to fit it."""

    def __init__(self, model, training_data: FrameDataset):
        self._training_data = training_data
        self._model = model

    @property
    def training_data(self) -> FrameDataset:
        return self._training_data

    @property
    def model(self):
        return self._model

    @property
    def schema(self) -> FeatureSchema:
        return self.training_data.schema
