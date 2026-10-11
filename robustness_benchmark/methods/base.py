"""The generator interface every method exposes to the registry."""

from abc import ABC, abstractmethod
from typing import Protocol

import pandas as pd

from robustness_benchmark.core.task import ClassificationTask


class Generator(Protocol):
    """Produces at most one counterfactual per factual; empty frame for none."""

    last_metadata: dict[str, object]

    def generate_for_instance(
        self, instance: pd.Series, *, neg_value: int = 0, **kwargs: object
    ) -> pd.DataFrame: ...


class CEGenerator(ABC):
    """Base class of the generators inherited from RobustX."""

    def __init__(self, task: ClassificationTask):
        self._task = task
        self.last_metadata: dict[str, object] = {}

    @property
    def task(self) -> ClassificationTask:
        return self._task

    def generate_for_instance(
        self, instance: pd.Series, *, neg_value: int = 0, **kwargs: object
    ) -> pd.DataFrame:
        return self._generation_method(instance, neg_value=neg_value, **kwargs)

    @abstractmethod
    def _generation_method(
        self, instance: pd.Series, neg_value: int = 0, **kwargs: object
    ) -> pd.DataFrame:
        """Return a one-row frame with the counterfactual, or an empty frame."""
