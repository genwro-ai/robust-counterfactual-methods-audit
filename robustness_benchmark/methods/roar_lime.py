from typing import Any

import numpy as np
import pandas as pd
import torch
from lime.lime_tabular import LimeTabularExplainer
from sklearn.linear_model import LogisticRegression

ROAR_LIME_IMPLEMENTATION = "upadhyay2021_box_logistic_lime_v2"
LIME_PARAMETER_DECIMALS = 4


def _as_scalar(value: Any) -> float:
    array = np.asarray(value, dtype=float).reshape(-1)
    if len(array) != 1:
        raise ValueError(f"Expected a scalar, got shape {np.asarray(value).shape}")
    return float(array[0])


def logistic_lime_parameters(
    explanation: Any, n_features: int
) -> tuple[np.ndarray, float]:
    """Extract parameters from LIME's logistic-regression compatibility path.

    LIME normally expects a regressor with a one-dimensional ``coef_``. Binary
    ``LogisticRegression`` exposes ``coef_`` as ``(1, n_features)``, so LIME
    stores the whole coefficient vector in the first explanation tuple. This
    is also the representation consumed by the available ROAR-LIME reproduction.
    """

    entries = explanation.local_exp[1]
    if len(entries) == 1:
        coefficient_array = np.asarray(entries[0][1], dtype=float).reshape(-1)
        if len(coefficient_array) == n_features:
            return coefficient_array, _as_scalar(explanation.intercept[1])

    coefficients = np.zeros(n_features, dtype=float)
    for feature_index, coefficient in entries:
        coefficients[int(feature_index)] = _as_scalar(coefficient)
    return coefficients, _as_scalar(explanation.intercept[1])


def hard_label_probabilities(probabilities: np.ndarray) -> np.ndarray:
    """Encode binary decisions in the two-column format expected by LIME.

    Upadhyay et al. specify logistic regression as LIME's local model. Scikit-learn's
    logistic regression requires class labels rather than continuous probability
    targets, so reproducing that setup requires thresholding the black-box
    probabilities before fitting the surrogate. The APAS comparison code follows
    the same compatibility path.
    """

    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError(
            f"Expected binary probabilities with shape (n, 2), got {values.shape}"
        )
    favorable = (values[:, 1] >= 0.5).astype(float)
    return np.column_stack((1.0 - favorable, favorable))


def input_space_parameters(
    scaled_coefficients: np.ndarray,
    scaled_intercept: float,
    scale: np.ndarray,
    mean: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Convert LIME's standardized linear model to the task input coordinates."""

    coefficients = scaled_coefficients / scale
    intercept = scaled_intercept - float(np.sum(scaled_coefficients * mean / scale))
    return coefficients, intercept


def worst_case_parameter_shift(
    values: torch.Tensor, target: int, delta: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the exact box-constrained shift that maximizes binary CE loss."""

    direction = -1.0 if target == 1 else 1.0
    weight_shift = direction * delta * torch.sign(values)
    intercept_shift = values.new_tensor(direction * delta)
    return weight_shift, intercept_shift


def optimize_roar(
    factual: np.ndarray,
    coefficients: np.ndarray,
    intercept: float,
    *,
    target: int,
    delta: float,
    lambda_param: float,
    learning_rate: float,
    loss_threshold: float,
    maximum_iterations: int,
) -> tuple[np.ndarray, int, bool, float, float]:
    """Optimize the published ROAR min-max binary-cross-entropy objective."""

    x = torch.as_tensor(factual, dtype=torch.float32)
    weights = torch.as_tensor(coefficients, dtype=torch.float32)
    bias = x.new_tensor(intercept)
    target_tensor = x.new_tensor(float(target))
    candidate = x.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([candidate], lr=learning_rate)

    previous_loss: torch.Tensor | None = None
    converged = False
    iterations = 0
    validity_loss = x.new_tensor(float("nan"))
    cost = x.new_tensor(float("nan"))
    for iterations in range(1, maximum_iterations + 1):
        weight_shift, intercept_shift = worst_case_parameter_shift(
            candidate, target, delta
        )
        optimizer.zero_grad()
        worst_logit = (
            torch.dot(weights + weight_shift, candidate) + bias + intercept_shift
        )
        validity_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            worst_logit, target_tensor
        )
        cost = torch.linalg.vector_norm(candidate - x, ord=1)
        loss = validity_loss + lambda_param * cost
        loss.backward()
        optimizer.step()

        if (
            previous_loss is not None
            and torch.abs(previous_loss - loss.detach()) <= loss_threshold
        ):
            converged = True
            break
        previous_loss = loss.detach().clone()

    return (
        candidate.detach().cpu().numpy(),
        iterations,
        converged,
        float(validity_loss.detach()),
        float(cost.detach()),
    )


class RoarLimeAdapter:
    """Apply published ROAR optimization to a local logistic LIME surrogate."""

    def __init__(
        self,
        task: Any,
        *,
        seed: int,
        num_samples: int = 20_000,
        delta: float = 0.1,
        lambda_param: float = 0.01,
        learning_rate: float = 0.001,
        loss_threshold: float = 0.0001,
        maximum_iterations: int = 20_000,
    ) -> None:
        self.task = task
        self.seed = seed
        self.num_samples = num_samples
        self.delta = delta
        self.lambda_param = lambda_param
        self.learning_rate = learning_rate
        self.loss_threshold = loss_threshold
        self.maximum_iterations = maximum_iterations
        self.last_metadata: dict[str, object] = {}
        X_train = task.training_data.X
        self.explainer = LimeTabularExplainer(
            X_train.to_numpy(),
            feature_names=X_train.columns.tolist(),
            class_names=["adverse", "favorable"],
            mode="classification",
            discretize_continuous=False,
            feature_selection="none",
            random_state=seed,
        )

    def _predict_hard_proba(self, values: np.ndarray) -> np.ndarray:
        return hard_label_probabilities(
            self.task.model.predict_proba(values).to_numpy()
        )

    def generate_for_instance(
        self, instance: pd.Series, *, neg_value: int = 0, **_kwargs: object
    ) -> pd.DataFrame:
        target = 1 - neg_value
        explanation = self.explainer.explain_instance(
            instance.to_numpy(dtype=float),
            self._predict_hard_proba,
            labels=(target,),
            num_features=len(instance),
            num_samples=self.num_samples,
            model_regressor=LogisticRegression(
                random_state=self.seed,
                max_iter=1_000,
            ),
        )
        scaled_coefficients, scaled_intercept = logistic_lime_parameters(
            explanation, len(instance)
        )
        scaled_coefficients = np.round(
            scaled_coefficients, decimals=LIME_PARAMETER_DECIMALS
        )
        scaled_intercept = round(scaled_intercept, LIME_PARAMETER_DECIMALS)

        scale = np.asarray(self.explainer.scaler.scale_, dtype=float)
        mean = np.asarray(self.explainer.scaler.mean_, dtype=float)
        coefficients, intercept = input_space_parameters(
            scaled_coefficients,
            scaled_intercept,
            scale,
            mean,
        )
        factual_values = instance.to_numpy(dtype=float)
        ce, iterations, converged, validity_loss, optimization_cost = optimize_roar(
            factual_values,
            coefficients,
            intercept,
            target=target,
            delta=self.delta,
            lambda_param=self.lambda_param,
            learning_rate=self.learning_rate,
            loss_threshold=self.loss_threshold,
            maximum_iterations=self.maximum_iterations,
        )
        result = pd.DataFrame([ce], columns=instance.index)
        factual_surrogate_probability = float(
            torch.sigmoid(
                torch.tensor(coefficients @ factual_values + intercept)
            ).item()
        )
        ce_surrogate_probability = float(
            torch.sigmoid(torch.tensor(coefficients @ ce + intercept)).item()
        )
        base_factual_probability = float(self.task.model.predict(instance).iloc[0, 0])
        base_ce_probability = float(self.task.model.predict(result).iloc[0, 0])
        self.last_metadata = {
            "roar_implementation": ROAR_LIME_IMPLEMENTATION,
            "surrogate": "lime_logistic_hard_labels",
            "lime_num_samples": self.num_samples,
            "lime_parameter_rounding_decimals": LIME_PARAMETER_DECIMALS,
            "lime_local_accuracy": float(explanation.score),
            "surrogate_factual_probability": factual_surrogate_probability,
            "base_factual_probability": base_factual_probability,
            "roar_delta": self.delta,
            "roar_lambda": self.lambda_param,
            "roar_learning_rate": self.learning_rate,
            "roar_loss": "binary_cross_entropy",
            "roar_uncertainty_set": "linf_box_weights_and_intercept",
            "roar_iterations": iterations,
            "roar_converged": converged,
            "roar_final_worst_case_bce": validity_loss,
            "roar_final_l1": optimization_cost,
            "surrogate_ce_probability": ce_surrogate_probability,
            "base_ce_probability": base_ce_probability,
            "surrogate_base_ce_agreement": bool(
                (ce_surrogate_probability >= 0.5) == (base_ce_probability >= 0.5)
            ),
        }
        return result
