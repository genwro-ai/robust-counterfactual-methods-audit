"""ROAR on a logistic base model (Upadhyay et al., NeurIPS 2021).

ROAR was posed for linear models: it minimizes the binary cross-entropy under
the worst-case shift of the coefficients and intercept inside an L-infinity box,
plus an L1 cost. ``RoarLimeAdapter`` applies that objective to a local LIME
surrogate of a black box; when the base model is itself logistic, the
surrogate is unnecessary and the box is centred on the model's exact
parameters. The optimizer is shared with ROAR-LIME, including its masking of
immutable columns; projection after the search restores one-hot and integer
validity.

Deviation on constrained schemas (shared with ROAR-LIME): the optimizer stops
only once the loss has converged and the projected iterate is valid under the
worst-case shift, and returns that projected point.
"""

import pandas as pd

from robustness_benchmark.methods.feature_constraints import search_projection
from robustness_benchmark.methods.roar_lime import optimize_roar

ROAR_IMPLEMENTATION = "upadhyay2021_box_logistic_exact_v1"


class RoarAdapter:
    def __init__(
        self,
        task,
        *,
        delta: float = 0.1,
        lambda_param: float = 0.01,
        learning_rate: float = 0.001,
        loss_threshold: float = 0.0001,
        maximum_iterations: int = 20_000,
    ) -> None:
        self.model = task.model
        self.schema = task.schema
        self.frozen = task.schema.immutable_columns()
        self.coefficients, self.intercept = task.model.linear_parameters()
        self.delta = delta
        self.lambda_param = lambda_param
        self.learning_rate = learning_rate
        self.loss_threshold = loss_threshold
        self.maximum_iterations = maximum_iterations
        self.last_metadata: dict[str, object] = {}

    def generate_for_instance(
        self, instance: pd.Series, *, neg_value: int = 0, **_kwargs: object
    ) -> pd.DataFrame:
        ce, iterations, converged, validity_loss, cost = optimize_roar(
            instance.to_numpy(dtype=float),
            self.coefficients,
            self.intercept,
            target=1 - neg_value,
            delta=self.delta,
            lambda_param=self.lambda_param,
            learning_rate=self.learning_rate,
            loss_threshold=self.loss_threshold,
            maximum_iterations=self.maximum_iterations,
            frozen=self.frozen,
            project=search_projection(self.schema, instance.to_numpy()),
        )
        result = pd.DataFrame([ce], columns=instance.index)
        self.last_metadata = {
            "roar_implementation": ROAR_IMPLEMENTATION,
            "roar_delta": self.delta,
            "roar_lambda": self.lambda_param,
            "roar_learning_rate": self.learning_rate,
            "roar_loss": "binary_cross_entropy",
            "roar_uncertainty_set": "linf_box_weights_and_intercept",
            "roar_iterations": iterations,
            "roar_converged": converged,
            "roar_final_worst_case_bce": validity_loss,
            "roar_final_l1": cost,
            "base_ce_probability": float(self.model.predict(result).iloc[0, 0]),
        }
        return result
