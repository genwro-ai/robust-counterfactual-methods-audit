"""Base classes inherited from RobustX, and prediction helpers for adapters."""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from robustness_benchmark.core.task import ClassificationTask


class CEGenerator(ABC):
    """
    Abstract class for generating counterfactual explanations for a given task.

    This class provides a framework for generating counterfactuals based on a distance function
    and a given task. It supports default distance functions such as Euclidean and Manhattan,
    and allows for custom distance functions.

    Attributes:
        _task (Task): The task to solve.
        __customFunc (callable, optional): A custom distance function.
    """

    def __init__(self, ct: ClassificationTask, custom_distance_func=None):
        """
        Initializes the CEGenerator with a task and an optional custom distance function.

        @param ct: The Task instance to solve.
        @param custom_distance_func: An optional custom distance function.
        """
        self._task = ct
        self.__customFunc = custom_distance_func

    @property
    def task(self):
        return self._task

    def generate(
        self, instances: pd.DataFrame, neg_value=0, column_name="target", **kwargs
    ) -> pd.DataFrame:
        """
        Generates counterfactuals for a given DataFrame of instances.

        @param instances: A DataFrame of instances for which you want to generate counterfactuals explanations.
        @param distance_func: The method to calculate the distance between two points. Options are 'l1' / 'manhattan', 'l2' / 'euclidean', and 'custom'.
        @param column_name: The name of the target column.
        @param neg_value: The value considered negative in the target variable.
        @return: A DataFrame of the counterfactual explanations for the provided instances.
        """
        cs = []

        for _, instance in instances.iterrows():
            cs.append(
                self.generate_for_instance(
                    instance, neg_value=neg_value, column_name=column_name, **kwargs
                )
            )

        res = pd.concat(cs)

        return res

    def generate_for_instance(
        self, instance, neg_value=0, column_name="target", **kwargs
    ) -> pd.DataFrame:
        """
        Generates a counterfactual for a provided instance.

        @param instance: The instance for which you would like to generate a counterfactual.
        @param distance_func: The method to calculate the distance between two points. Options are 'l1' / 'manhattan', 'l2' / 'euclidean', and 'custom'.
        @param column_name: The name of the target column.
        @param neg_value: The value considered negative in the target variable.
        @return: A DataFrame containing the counterfactual explanations for the instance.
        """

        return self._generation_method(
            instance, neg_value=neg_value, column_name=column_name, **kwargs
        )

    def generate_for_all(
        self, neg_value=0, column_name="target", **kwargs
    ) -> pd.DataFrame:
        """
        Generates counterfactuals for all instances with a given negative value in their target column.

        @param neg_value: The value in the target column which counts as a negative instance.
        @param column_name: The name of the target variable.
        @param distance_func: The method to calculate the distance between two points. Options are 'l1' / 'manhattan', 'l2' / 'euclidean', and 'custom'.
        @return: A DataFrame of the counterfactuals for all negative values.
        """

        negatives = self.task.get_negative_instances(neg_value, column_name=column_name)
        # preds = self.task.model.predict(self.task.training_data.X).values.flatten()
        #
        # if neg_value == 0:
        #     idxs = np.where(preds < 0.5)[0]
        #     negatives = self.task.training_data.data.drop(columns=[column_name])
        #     negatives = pd.DataFrame(negatives.values[idxs], columns=negatives.columns)
        # else:
        #     idxs = np.where(preds >= 0.5)[0]
        #     negatives = self.task.training_data.data.drop(columns=[column_name])
        #     negatives = pd.DataFrame(negatives.values[idxs], columns=negatives.columns)

        counterfactuals = self.generate(
            negatives, column_name=column_name, neg_value=neg_value, **kwargs
        )

        counterfactuals.index = negatives.index
        return counterfactuals

    @abstractmethod
    def _generation_method(self, instance, column_name="target", neg_value=0, **kwargs):
        """
        Abstract method to be implemented by subclasses for generating counterfactuals.

        @param instance: The instance for which to generate a counterfactual.
        @param distance_func: The function to calculate distances.
        @param column_name: The name of the target column.
        @param neg_value: The value considered negative in the target variable.
        @return: A DataFrame containing the generated counterfactuals.
        """

    @property
    def custom_distance_func(self):
        """
        Returns custom distance function passed at instantiation
        @return: distance Function, (DataFrame, DataFrame) -> Int
        """
        return self.__customFunc


class ModelChangesRobustnessEvaluator(ABC):
    """
    Abstract base class for evaluating the robustness of CEs with respect to model changes.

    This class defines an interface for evaluating how robust a CE's validity are
    when the model parameters are changed.
    """

    def __init__(self, ct: ClassificationTask):
        """
        Initializes the ModelChangesRobustnessEvaluator with a given task.

        @param ct: The task for which robustness evaluations are being made.
                   Provided as a Task instance.
        """
        self.task = ct

    @abstractmethod
    def evaluate(self, instance, neg_value=0):
        """
        Abstract method to evaluate the robustness of a model's prediction on a given instance.

        Must be implemented by subclasses.

        @param instance: The instance for which to evaluate robustness.
                         This could be a single data point for the model.
        @param neg_value: The value considered negative in the target variable.
                          Used to determine if the counterfactual flips the prediction.
        @return: Result of the robustness evaluation. The return type should be defined by the subclass.
        """


def predict_crisp(model, values: np.ndarray) -> np.ndarray:
    probability = model.predict(np.asarray(values)).iloc[:, 0].to_numpy()
    return (probability >= 0.5).astype(int)


def predict_proba(model, values: np.ndarray) -> np.ndarray:
    return model.predict_proba(np.asarray(values)).to_numpy()
