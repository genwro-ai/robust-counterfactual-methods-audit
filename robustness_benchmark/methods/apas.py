import math
from time import perf_counter
from typing import Protocol

import gurobipy
import numpy as np
import pandas as pd
import torch
from gurobipy import GRB
from torch.func import functional_call, vmap

from robustness_benchmark.methods.mce import MCE

APAS_IMPLEMENTATION = "marzari2024_algorithm2_calibrated_radius_holdout_v3"
_CERTIFICATION_SEED_OFFSET = 1_000_003


class _MCEProtocol(Protocol):
    def generate_for_instance(
        self, instance: object, /, **kwargs: object
    ) -> object: ...


def _solver_status(component: object) -> int | None:
    opt = getattr(component, "opt", None)
    model = getattr(opt, "gurobiModel", None)
    status = getattr(model, "status", None)
    return int(status) if isinstance(status, (int, np.integer)) else None


def _time_limit_status() -> int:
    return int(GRB.TIME_LIMIT)


def _set_default_solver_limit(seconds: float) -> None:
    gurobipy.setParam("TimeLimit", max(float(seconds), 0.01))
    gurobipy.setParam("OutputFlag", 0)


def _reset_default_solver_limit() -> None:
    gurobipy.setParam("TimeLimit", GRB.INFINITY)


class APASAdapter:
    """Generate CEs with APDeltaS search and an independent final certificate.

    This implements the paper's robust-CFE generation pattern in Algorithm 2
    at a fixed, externally calibrated parameter radius. It starts with a
    minimum-cost CE, increases its output margin, and accepts the first
    candidate that remains valid on every sampled parameter concretization.
    The selected candidate is checked once more on a separate sample that never
    affects candidate selection. It does not implement Algorithm 1's search for
    the maximum radius of an already fixed CFE.
    """

    def __init__(
        self,
        task,
        *,
        seed: int,
        delta: float,
        confidence: float = 0.999,
        robustness_probability: float = 0.995,
        initial_margin: float = 0.01,
        increment: float = 0.2,
        maximum_margin: float = 20.0,
        wall_time_seconds: float = 30.0,
        mce: _MCEProtocol | None = None,
    ) -> None:
        if delta <= 0:
            raise ValueError("delta must be positive")
        if not 0 < confidence < 1 or not 0 < robustness_probability < 1:
            raise ValueError("APDeltaS confidence and probability must be in (0, 1)")
        if increment <= 0 or maximum_margin < initial_margin:
            raise ValueError("invalid APDeltaS margin schedule")
        if wall_time_seconds <= 0:
            raise ValueError("wall_time_seconds must be positive")

        if mce is None:
            mce = MCE(task)

        self.task = task
        self.seed = seed
        self.delta = delta
        self.confidence = confidence
        self.robustness_probability = robustness_probability
        self.number_of_samples = math.ceil(
            math.log(1 - confidence) / math.log(robustness_probability)
        )
        self.initial_margin = initial_margin
        self.increment = increment
        self.maximum_margin = maximum_margin
        self.wall_time_seconds = wall_time_seconds
        self.mce = mce
        self._search_parameters: dict[str, torch.Tensor] | None = None
        self._instance_index = 0
        self.last_metadata: dict[str, object] = {}

    def _sample_parameter_concretizations(self, seed: int) -> dict[str, torch.Tensor]:
        network = self.task.model.get_torch_model()
        generator = torch.Generator().manual_seed(seed)
        return {
            name: parameter.detach().unsqueeze(0)
            + torch.empty(
                (self.number_of_samples, *parameter.shape),
                dtype=parameter.dtype,
                device=parameter.device,
            ).uniform_(-self.delta, self.delta, generator=generator)
            for name, parameter in network.named_parameters()
        }

    def _parameter_concretizations(self) -> dict[str, torch.Tensor]:
        if self._search_parameters is None:
            self._search_parameters = self._sample_parameter_concretizations(self.seed)
        return self._search_parameters

    def _passes_parameter_sample(
        self,
        candidate: pd.Series,
        desired_output: int,
        parameters: dict[str, torch.Tensor],
    ) -> bool:
        network = self.task.model.get_torch_model()
        buffers = dict(network.named_buffers())
        value = torch.as_tensor(candidate.to_numpy(), dtype=torch.float32).reshape(
            1, -1
        )

        def predict(parameters: dict[str, torch.Tensor]) -> torch.Tensor:
            return functional_call(network, (parameters, buffers), (value,)).reshape(())

        with torch.no_grad():
            probability = vmap(predict)(parameters)
        valid = probability >= 0.5 if desired_output == 1 else probability < 0.5
        return bool(valid.all())

    def _is_probabilistically_robust(
        self, candidate: pd.Series, desired_output: int
    ) -> bool:
        """Apply the author-faithful, reused sample during candidate search."""

        return self._passes_parameter_sample(
            candidate,
            desired_output,
            self._parameter_concretizations(),
        )

    def certify_candidate(
        self,
        candidate: pd.Series,
        desired_output: int,
        *,
        instance_index: int,
    ) -> tuple[bool, int]:
        """Certify a selected CFE on an independent, instance-specific sample."""

        certification_seed = self.seed + _CERTIFICATION_SEED_OFFSET + instance_index
        parameters = self._sample_parameter_concretizations(certification_seed)
        return (
            self._passes_parameter_sample(candidate, desired_output, parameters),
            certification_seed,
        )

    def _metadata(
        self,
        *,
        outcome: str,
        certified: bool,
        iterations: int,
        margin: float | None,
        author_sample_accepted: bool = False,
        confirmation_passed: bool | None = None,
        confirmation_seed: int | None = None,
    ) -> dict[str, object]:
        return {
            "apas_implementation": APAS_IMPLEMENTATION,
            "method_certified": certified,
            "apas_outcome": outcome,
            "apas_delta": self.delta,
            "apas_confidence": self.confidence,
            "apas_robustness_probability": self.robustness_probability,
            "apas_samples": self.number_of_samples,
            "apas_iterations": iterations,
            "apas_final_margin": margin,
            "apas_initial_margin": self.initial_margin,
            "apas_increment": self.increment,
            "apas_maximum_margin": self.maximum_margin,
            "apas_parameter_distribution": "independent_uniform_linf_box",
            "apas_base_method": "mce",
            "apas_author_sample_accepted": author_sample_accepted,
            "apas_search_sample_reused": True,
            "apas_confirmation_independent": True,
            "apas_confirmation_passed": confirmation_passed,
            "apas_confirmation_samples": (
                self.number_of_samples if confirmation_passed is not None else 0
            ),
            "apas_confirmation_seed": confirmation_seed,
        }

    def generate_for_instance(
        self, factual: pd.Series, *, neg_value: int = 0, **_kwargs: object
    ) -> pd.DataFrame:
        if neg_value != 0:
            raise ValueError("The APDeltaS adapter currently supports target class 1")
        desired_output = 1
        instance_index = self._instance_index
        self._instance_index += 1
        deadline = perf_counter() + self.wall_time_seconds
        margin = self.initial_margin
        iterations = 0

        try:
            while margin <= self.maximum_margin + 1e-12:
                remaining = deadline - perf_counter()
                if remaining <= 0:
                    self.last_metadata = self._metadata(
                        outcome="timeout",
                        certified=False,
                        iterations=iterations,
                        margin=margin,
                    )
                    return pd.DataFrame(columns=factual.index)

                _set_default_solver_limit(remaining)
                result = self.mce.generate_for_instance(
                    factual,
                    neg_value=neg_value,
                    minimum_distance=margin,
                    M=100_000,
                    epsilon=0.0001,
                )
                iterations += 1
                if (
                    _solver_status(self.mce) == _time_limit_status()
                    or perf_counter() >= deadline
                ):
                    self.last_metadata = self._metadata(
                        outcome="timeout",
                        certified=False,
                        iterations=iterations,
                        margin=margin,
                    )
                    return pd.DataFrame(columns=factual.index)
                if not isinstance(result, pd.DataFrame) or result.empty:
                    self.last_metadata = self._metadata(
                        outcome="base_method_failed",
                        certified=False,
                        iterations=iterations,
                        margin=margin,
                    )
                    return pd.DataFrame(columns=factual.index)

                candidate = result.iloc[0].astype(float)
                candidate.index = factual.index
                if self.task.model.predict_single(
                    candidate
                ) == desired_output and self._is_probabilistically_robust(
                    candidate, desired_output
                ):
                    confirmation_started = perf_counter()
                    confirmation_passed, confirmation_seed = self.certify_candidate(
                        candidate,
                        desired_output,
                        instance_index=instance_index,
                    )
                    self.last_metadata = self._metadata(
                        outcome=(
                            "probabilistically_robust"
                            if confirmation_passed
                            else "author_sample_accepted_holdout_rejected"
                        ),
                        certified=confirmation_passed,
                        iterations=iterations,
                        margin=margin,
                        author_sample_accepted=True,
                        confirmation_passed=confirmation_passed,
                        confirmation_seed=confirmation_seed,
                    )
                    self.last_metadata["apas_confirmation_runtime_seconds"] = (
                        perf_counter() - confirmation_started
                    )
                    return pd.DataFrame([candidate.to_numpy()], columns=factual.index)
                margin += self.increment

            self.last_metadata = self._metadata(
                outcome="margin_limit",
                certified=False,
                iterations=iterations,
                margin=margin - self.increment,
            )
            return pd.DataFrame(columns=factual.index)
        finally:
            _reset_default_solver_limit()
