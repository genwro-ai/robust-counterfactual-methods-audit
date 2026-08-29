import numpy as np
import pandas as pd

from robustness_benchmark.methods.betarce import BetaRCE
from robustness_benchmark.methods.growing_spheres import growing_spheres_search
from robustness_benchmark.methods.robx import RobX


def _crisp(model, values: np.ndarray) -> np.ndarray:
    probability = model.predict(np.asarray(values)).iloc[:, 0].to_numpy()
    return (probability >= 0.5).astype(int)


def _proba(model, values: np.ndarray) -> np.ndarray:
    return model.predict_proba(np.asarray(values)).to_numpy()


class RobXAdapter:
    def __init__(
        self,
        task,
        *,
        seed: int,
        variance: float = 0.01,
        tau: float = 0.5,
        stability_samples: int = 1_000,
    ) -> None:
        self.task = task
        self.seed = seed
        self.variance = variance
        self.tau = tau
        self.stability_samples = stability_samples
        X_train = task.training_data.X.to_numpy()
        self.generator = RobX(
            X_train,
            lambda values: _crisp(task.model, values),
            lambda values: _proba(task.model, values),
            feature_min=X_train.min(axis=0),
            feature_max=X_train.max(axis=0),
        )
        self.last_metadata: dict[str, object] = {}

    def generate_for_instance(
        self, factual: pd.Series, *, neg_value: int = 0, **_kwargs: object
    ) -> pd.DataFrame:
        result = self.generator.generate(
            factual.to_numpy(),
            target_class=1 - neg_value,
            variance=self.variance,
            tau=self.tau,
            N=self.stability_samples,
            k=10,
            robx_max_iter=800,
            robx_lambda=0.1,
            gs_n_search_samples=1_000,
            gs_p_norm=1,
            gs_step=0.1,
            gs_max_iter=800,
            rng=np.random.default_rng(self.seed),
        )
        self.last_metadata = {
            **result.metadata,
            "variance": self.variance,
            "tau": self.tau,
            "stability_samples": self.stability_samples,
            "robx_k": 10,
        }
        if result.counterfactual is None:
            return pd.DataFrame(columns=factual.index)
        return pd.DataFrame([result.counterfactual], columns=factual.index)


class BetaRCEAdapter:
    def __init__(
        self,
        task,
        generation_models: list,
        *,
        seed: int,
        delta: float = 0.9,
        alpha: float = 0.95,
        generation_ensemble_n: int = 32,
        gs_n_search_samples: int = 100,
        gs_p_norm: int = 2,
        gs_step: float = 0.1,
        base_gs_max_iter: int = 200,
        gs_max_iter: int = 100,
    ) -> None:
        if len(generation_models) != generation_ensemble_n:
            raise ValueError(
                "BetaRCE requires exactly "
                f"{generation_ensemble_n} separate generation-ensemble models"
            )
        self.rng = np.random.default_rng(seed)
        self.delta = delta
        self.alpha = alpha
        self.generation_ensemble_n = generation_ensemble_n
        self.gs_n_search_samples = gs_n_search_samples
        self.gs_p_norm = gs_p_norm
        self.gs_step = gs_step
        self.base_gs_max_iter = base_gs_max_iter
        self.gs_max_iter = gs_max_iter
        self.base_predict = lambda values: _crisp(task.model, values)
        self.generator = BetaRCE(
            self.base_predict,
            [
                lambda values, model=model: _crisp(model, values)
                for model in generation_models
            ],
        )
        self.last_metadata: dict[str, object] = {}

    def generate_for_instance(
        self, factual: pd.Series, *, neg_value: int = 0, **_kwargs: object
    ) -> pd.DataFrame:
        target_class = 1 - neg_value
        base_counterfactual = growing_spheres_search(
            factual.to_numpy(),
            pred_fn_crisp=self.base_predict,
            target_class=target_class,
            n_search_samples=self.gs_n_search_samples,
            p_norm=self.gs_p_norm,
            step=self.gs_step,
            max_iter=self.base_gs_max_iter,
            rng=self.rng,
        )
        if base_counterfactual is None:
            self.last_metadata = {
                "status": "base_counterfactual_not_found",
                "delta": self.delta,
                "alpha": self.alpha,
                "generation_ensemble_n": self.generation_ensemble_n,
            }
            return pd.DataFrame(columns=factual.index)

        result = self.generator.generate(
            base_counterfactual,
            target_class=target_class,
            delta=self.delta,
            alpha=self.alpha,
            gs_n_search_samples=self.gs_n_search_samples,
            gs_p_norm=self.gs_p_norm,
            gs_step=self.gs_step,
            gs_max_iter=self.gs_max_iter,
            rng=self.rng,
        )
        self.last_metadata = {
            **result.metadata,
            "delta": self.delta,
            "alpha": self.alpha,
            "generation_ensemble_n": self.generation_ensemble_n,
            "base_generator": "growing_spheres",
            "gs_n_search_samples": self.gs_n_search_samples,
            "gs_p_norm": self.gs_p_norm,
            "gs_step": self.gs_step,
            "base_gs_max_iter": self.base_gs_max_iter,
            "gs_max_iter": self.gs_max_iter,
        }
        if result.counterfactual is None:
            return pd.DataFrame(columns=factual.index)
        return pd.DataFrame([result.counterfactual], columns=factual.index)
