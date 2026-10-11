"""RNCE (Jiang et al., AIJ 2024, Algorithms 2-4) on RobustX's interval encoding.

Aligned with the paper and the authors' ``intabs/methods.py``:
- Neighbours are found with the Manhattan (L1) metric, as in ``rnce``
  (methods.py:332), the paper's normalised-L1 proximity; RobustX uses L2.
- ``optimal`` runs Algorithm 4's line search (lines 7-10), which RobustX
  ignores: from the factual towards the robust neighbour in steps of 0.05,
  the first certified point is returned, as ``optimise_distance``
  (methods.py:338) does. Its points interpolate one-hot columns; the
  registry's final projection makes them a valid encoding.
- Weights and biases share one radius: the paper's delta bounds the
  L-infinity distance over all parameters (§8.2). Its incremental-retraining
  value is the mean distance over updated models, see ``rnce_method_config``.

With immutable features, each candidate takes the factual's immutable values
and is certified again if they changed, so the returned point is the one
certified.
"""

from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree

from robustness_benchmark.core.task import ClassificationTask
from robustness_benchmark.methods.base import CEGenerator
from robustness_benchmark.methods.feature_constraints import matches_immutable
from robustness_benchmark.methods.milp.delta_robustness import DeltaRobustnessEvaluator

RNCE_INTERVAL_ENCODING = "sign_aware_first_layer_v2"
RNCE_IMPLEMENTATION = "jiang2024_algorithm4_l1_line_search_v1"
# Algorithm 4, line 2: step s = 0.05 from x' (a = 1) towards x; a = 1 is x'.
LINE_SEARCH_WEIGHTS = np.linspace(0.05, 0.95, 19)


class RNCE(CEGenerator):
    """Robust nearest-neighbour counterfactuals certified by interval abstraction."""

    def __init__(self, task: ClassificationTask):
        super().__init__(task)
        self.intabs = DeltaRobustnessEvaluator(task)

    def _generation_method(
        self,
        x,
        robustInit=True,
        optimal=True,
        column_name="target",
        neg_value=0,
        delta=0.005,
        bias_delta=0.005,
        interval_encoding=RNCE_INTERVAL_ENCODING,
        **kwargs,
    ):
        """Algorithm 2: certified candidates, nearest one in L1, optional line search.

        Returns the factual unchanged when no candidate is certified, which the
        registry records as no counterfactual.
        """
        if interval_encoding != RNCE_INTERVAL_ENCODING:
            raise ValueError(
                f"Unsupported RNCE interval encoding {interval_encoding!r}"
            )

        S = self.getCandidates(
            robustInit, delta, bias_delta, column_name=column_name, neg_value=neg_value
        )
        if S.empty:
            print("No instance in the dataset is robust for the given perturbations!")
            return pd.DataFrame(x).T

        def certified(candidate: pd.Series) -> bool:
            return self.intabs.evaluate(
                candidate,
                delta=delta,
                bias_delta=bias_delta,
                desired_output=1 - neg_value,
            )

        if self.task.schema.immutable:
            nearest = self._nearest_robust_keeping_immutables(S, x, certified)
            if nearest is None:
                return pd.DataFrame(x).T
        else:
            tree = KDTree(S.to_numpy(dtype=float), leaf_size=40, metric="manhattan")
            (index,) = tree.query(x.to_numpy(dtype=float).reshape(1, -1), k=1)[1][0]
            nearest = S.iloc[index]
        if optimal:
            nearest = self._closest_certified_on_line(x, nearest, certified)
        return nearest.to_frame().T

    @staticmethod
    def _closest_certified_on_line(x, counterfactual, certified):
        """Algorithm 4, lines 7-10: the certified point nearest x on the segment."""

        factual = x.to_numpy(dtype=float)
        target = counterfactual.to_numpy(dtype=float)
        for a in LINE_SEARCH_WEIGHTS:
            point = pd.Series(
                a * target + (1 - a) * factual, index=counterfactual.index
            )
            if certified(point):
                return point
        return counterfactual

    def _nearest_robust_keeping_immutables(self, S, x, certified):
        """L1-nearest certified candidate after it takes the factual's immutables."""

        schema = self.task.schema
        factual = x.to_numpy(dtype=float)
        moved = schema.project(S.to_numpy(dtype=float), factual)
        unchanged = matches_immutable(S, factual, schema)
        for index in np.argsort(np.abs(moved - factual).sum(axis=1), kind="stable"):
            candidate = pd.Series(moved[index], index=S.columns)
            if unchanged[index] or certified(candidate):
                return candidate
        return None

    @lru_cache  # noqa: B019 - retained from RobustX; one generator per task
    def getCandidates(
        self, robustInit, delta, bias_delta, column_name="target", neg_value=0
    ):
        """Training points certified for the target class (Algorithm 3)."""
        S = []

        for _, instance in self.task.training_data.data.iterrows():
            instance_x = instance.drop(column_name)
            if robustInit:
                if self.intabs.evaluate(
                    instance_x,
                    delta=delta,
                    bias_delta=bias_delta,
                    desired_output=1 - neg_value,
                ):
                    S.append(instance_x)
            else:
                if self.task.model.predict_single(instance_x):
                    S.append(instance_x)

        return pd.DataFrame(S)
