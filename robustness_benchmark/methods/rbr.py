import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist

from robustness_benchmark.methods.rbr_core import RBR as AuthorRBR


class LabelPredictor:
    """Expose the class-label API expected by the authors' implementation."""

    def __init__(self, model) -> None:
        self.model = model

    def predict(self, values: np.ndarray) -> np.ndarray:
        probability = self.model.predict(np.asarray(values)).iloc[:, 0].to_numpy()
        return (probability >= 0.5).astype(int)


class RBRAdapter:
    """Generate model-agnostic recourse under local distributional ambiguity."""

    def __init__(
        self,
        task,
        *,
        seed: int,
        num_counterfactuals: int = 1_000,
        num_samples: int = 200,
        perturb_radius_fraction: float = 0.2,
        delta_plus: float = 0.2,
        sigma: float = 1.0,
        epsilon_op: float = 0.5,
        epsilon_pe: float = 0.5,
        maximum_iterations: int = 500,
        generator=None,
    ) -> None:
        if num_counterfactuals <= 0 or num_samples <= 0 or maximum_iterations <= 0:
            raise ValueError("RBR sampling and iteration counts must be positive")
        if perturb_radius_fraction <= 0 or delta_plus < 0 or sigma <= 0:
            raise ValueError("invalid RBR radius, proximity budget, or bandwidth")
        if epsilon_op < 0 or epsilon_pe < 0:
            raise ValueError("RBR ambiguity radii must be non-negative")

        self.task = task
        self.seed = seed
        self.num_counterfactuals = num_counterfactuals
        self.num_samples = num_samples
        self.perturb_radius_fraction = perturb_radius_fraction
        self.delta_plus = delta_plus
        self.sigma = sigma
        self.epsilon_op = epsilon_op
        self.epsilon_pe = epsilon_pe
        self.maximum_iterations = maximum_iterations

        training = task.training_data.X.to_numpy(dtype=np.float32)
        distances = pdist(training, metric="euclidean")
        self.maximum_training_distance = (
            float(np.max(distances)) if len(distances) else 0.0
        )
        self.perturb_radius = (
            self.perturb_radius_fraction * self.maximum_training_distance
        )
        self.generator = generator or AuthorRBR(
            LabelPredictor(task.model),
            training,
            y_target=1,
            num_cfacts=min(self.num_counterfactuals, len(training)),
            max_iter=self.maximum_iterations,
            random_state=self.seed,
            device="cpu",
        )
        self.last_metadata: dict[str, object] = {}

    def generate_for_instance(
        self, factual: pd.Series, *, neg_value: int = 0, **_kwargs: object
    ) -> pd.DataFrame:
        if neg_value != 0:
            raise ValueError("The RBR adapter currently supports target class 1")

        candidate = self.generator.fit_instance(
            factual.to_numpy(dtype=np.float32),
            self.num_samples,
            self.perturb_radius,
            self.delta_plus,
            self.sigma,
            self.epsilon_op,
            self.epsilon_pe,
            None,
        )
        values = np.asarray(candidate, dtype=float).reshape(-1)
        finite = len(values) == len(factual) and bool(np.isfinite(values).all())
        base_valid = finite and self.task.model.predict_single(values) == 1
        self.last_metadata = {
            "rbr_outcome": "success" if base_valid else "base_invalid",
            "rbr_author_feasible": bool(getattr(self.generator, "feasible", False)),
            "rbr_num_counterfactuals": self.num_counterfactuals,
            "rbr_num_samples": self.num_samples,
            "rbr_perturb_radius_fraction": self.perturb_radius_fraction,
            "rbr_maximum_training_distance": self.maximum_training_distance,
            "rbr_perturb_radius": self.perturb_radius,
            "rbr_delta_plus": self.delta_plus,
            "rbr_sigma": self.sigma,
            "rbr_epsilon_op": self.epsilon_op,
            "rbr_epsilon_pe": self.epsilon_pe,
            "rbr_maximum_iterations": self.maximum_iterations,
            "rbr_parameter_distribution": "wasserstein_gaussian_mixture_ambiguity",
        }
        if not finite:
            return pd.DataFrame(columns=factual.index)
        return pd.DataFrame([values], columns=factual.index)
