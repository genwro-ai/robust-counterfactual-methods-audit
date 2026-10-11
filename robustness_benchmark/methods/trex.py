"""T:Rex (Hamman et al., ICML 2023), gradient version T-Rex:I.

Follows ``batch_trex_search`` in github.com/FaisalHamman/TReX-Counterfactuals:
from a Wachter base counterfactual, take normalized gradient-ascent steps on
the stability R(x) until R reaches tau or the budget ends, staying in an L2
ball around the base counterfactual and in the data box, and return the last
valid iterate. Steps, ball and box use per-feature min-max coordinates, the
[0, 1] scale of the reference code.

Deviations, all for comparability with RobX:
- R is RobX's estimator (mean minus population std of P(y=1) under
  N(x, sigma^2 I)), not the paper's relaxed mean_i(m(x_i) - |m(x) - m(x_i)|).
- tau, sigma^2 = 0.01 and N = 1,000 match ``robx_balanced``; tau is the median
  training-target stability, and sigma is in standardized units.
- R is checked at the current iterate, so a returned point that met tau is the
  one tested; the reference tests the point before its update. Acceptance is
  R >= tau as in the reference (RobX uses R > tau).
- The Wachter base uses the fixed default settings, not the per-dataset tuned
  Wachter configuration, so T:Rex has no tuned hyperparameters.
- 200 steps, the largest budget in Appendix D.
- Immutable columns get zero gradient (the Wachter base also keeps them).

As in the paper (App. D.1) and ``batch_trex_search``, one-hot and integer
columns are searched as continuous and only clamped to the data box; the
registry's final projection makes the returned point a valid encoding.
"""

import numpy as np
import pandas as pd
import torch

from robustness_benchmark.core.protocol import STABILITY_TAU_SEED_OFFSET
from robustness_benchmark.methods.robx import (
    tau_from_stability_quantile,
    training_target_stability_scores,
)
from robustness_benchmark.methods.wachter import WACHTER_DEFAULTS, Wachter

TREX_IMPLEMENTATION = "hamman2023_trex_i_robx_stability_v2"


def target_stability(model, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    """Differentiable RobX stability of P(y=1) at ``x`` for given noise."""

    probabilities = model.predict_proba_tensor(x + noise).reshape(-1)
    return probabilities.mean() - probabilities.std(correction=0)


def trex_search(
    model,
    base_counterfactual: np.ndarray,
    *,
    tau: float,
    variance: float,
    samples: int,
    step_size: float,
    max_steps: int,
    trust_radius: float,
    feature_min: np.ndarray,
    feature_max: np.ndarray,
    rng: np.random.Generator,
    frozen: list[int] | None = None,
) -> tuple[np.ndarray, float, int]:
    """Return the last valid iterate, its stability and the steps taken.

    Steps, the trust region and the box live in min-max coordinates
    z = (x - min) / (max - min), the [0, 1] scale the reference code uses;
    stability is measured in the model's input coordinates.
    """

    span = feature_max - feature_min
    offset = torch.as_tensor(feature_min, dtype=torch.float32)
    scale = torch.as_tensor(np.where(span > 0, span, 1.0), dtype=torch.float32)
    upper = torch.as_tensor(span > 0, dtype=torch.float32)
    anchor = (
        torch.as_tensor(base_counterfactual, dtype=torch.float32) - offset
    ) / scale
    z = anchor.clone()
    best, best_stability = np.asarray(base_counterfactual, dtype=float), -np.inf
    for step in range(max_steps + 1):
        z.requires_grad_(True)
        x = offset + z * scale
        noise = torch.as_tensor(
            rng.standard_normal((samples, len(z))) * np.sqrt(variance),
            dtype=torch.float32,
        )
        stability = target_stability(model, x, noise)
        with torch.no_grad():
            if float(model.predict_proba_tensor(x[None])) >= 0.5:
                best = x.detach().numpy().astype(float)
                best_stability = float(stability.detach())
                if best_stability >= tau:
                    return best, best_stability, step
        if step == max_steps:
            break
        (gradient,) = torch.autograd.grad(stability, z)
        if frozen:
            gradient[frozen] = 0.0
        with torch.no_grad():
            z = z + step_size * gradient / (gradient.norm() + 1e-7)
            shift = z - anchor
            z = anchor + shift * min(1.0, trust_radius / (float(shift.norm()) + 1e-7))
            z = torch.clamp(z, torch.zeros_like(upper), upper)
    return best, best_stability, max_steps


class TRexAdapter:
    def __init__(
        self,
        task,
        *,
        seed: int,
        variance: float = 0.01,
        stability_samples: int = 1_000,
        tau_quantile: float = 0.5,
        step_size: float = 0.01,
        max_steps: int = 200,
        trust_radius: float = 1.0,
    ) -> None:
        self.model = task.model
        self.frozen = task.schema.immutable_columns()
        self.variance = variance
        self.stability_samples = stability_samples
        self.step_size = step_size
        self.max_steps = max_steps
        self.trust_radius = trust_radius
        X_train = task.training_data.X.to_numpy()
        self.feature_min = X_train.min(axis=0)
        self.feature_max = X_train.max(axis=0)
        # Same scores, seed and quantile rule as full_benchmark's RobX
        # calibration, so tau equals robx_balanced's on the same base model.
        scores, target_points = training_target_stability_scores(
            X_train,
            lambda values: self.model.predict_proba(values).iloc[:, 1].to_numpy(),
            variance=variance,
            N=stability_samples,
            seed=seed + STABILITY_TAU_SEED_OFFSET,
        )
        self.tau, self.tau_calibration = tau_from_stability_quantile(
            scores, quantile=tau_quantile, target_training_points=target_points
        )
        self.base_generator = Wachter(task)
        self.rng = np.random.default_rng(seed)
        self.last_metadata: dict[str, object] = {}

    def generate_for_instance(
        self, factual: pd.Series, *, neg_value: int = 0, **_kwargs: object
    ) -> pd.DataFrame:
        if neg_value != 0:
            raise ValueError("T:Rex is calibrated for the favorable class 1")
        metadata = {
            "tau": self.tau,
            "raw_tau": self.tau_calibration["raw_tau"],
            "variance": self.variance,
            "stability_samples": self.stability_samples,
            "base_generator": "wachter",
        }
        base = self.base_generator.generate_for_instance(
            factual, neg_value=neg_value, **WACHTER_DEFAULTS
        )
        if self.model.predict_single(base) != 1:
            self.last_metadata = {**metadata, "status": "base_counterfactual_invalid"}
            return pd.DataFrame(columns=factual.index)

        counterfactual, stability, steps = trex_search(
            self.model,
            base.iloc[0].to_numpy(dtype=float),
            tau=self.tau,
            variance=self.variance,
            samples=self.stability_samples,
            step_size=self.step_size,
            max_steps=self.max_steps,
            trust_radius=self.trust_radius,
            feature_min=self.feature_min,
            feature_max=self.feature_max,
            rng=self.rng,
            frozen=self.frozen,
        )
        self.last_metadata = {
            **metadata,
            "status": "ok" if stability >= self.tau else "tau_not_reached",
            "trex_stability": stability,
            "trex_steps": steps,
        }
        return pd.DataFrame([counterfactual], columns=factual.index)
