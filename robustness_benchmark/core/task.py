import numpy as np
import pandas as pd


class FrameDataset:
    """In-memory numeric binary-classification dataset."""

    def __init__(self, X: pd.DataFrame, y: pd.Series, seed: int | None = None):
        self._seed = seed
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

    def get_negative_instances(
        self, neg_value: int, column_name: str = "target"
    ) -> pd.DataFrame:
        return self.data[self.data[column_name] == neg_value].drop(
            columns=[column_name]
        )

    def get_random_positive_instance(
        self, neg_value: int, column_name: str = "target"
    ) -> pd.Series:
        return (
            self.data[self.data[column_name] != neg_value]
            .drop(columns=[column_name])
            .sample(random_state=self._seed)
        )


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

    def get_random_positive_instance(
        self, neg_value: int, column_name: str = "target"
    ) -> pd.Series:
        positive = self.training_data.get_random_positive_instance(
            neg_value, column_name=column_name
        )
        while self.model.predict_single(positive) == neg_value:
            positive = self.training_data.get_random_positive_instance(
                neg_value, column_name=column_name
            )
        return positive

    def get_negative_instances(
        self, neg_value: int = 0, column_name: str = "target"
    ) -> pd.DataFrame:
        predictions = self.model.predict(self.training_data.X).to_numpy().reshape(-1)
        indices = np.where(predictions < 0.5 if neg_value == 0 else predictions >= 0.5)[
            0
        ]
        features = self.training_data.data.drop(columns=[column_name])
        return pd.DataFrame(features.to_numpy()[indices], columns=features.columns)
