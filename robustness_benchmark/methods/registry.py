from time import perf_counter

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from robustness_benchmark.core.schema import FeatureSchema
from robustness_benchmark.core.task import ClassificationTask, FrameDataset
from robustness_benchmark.evaluation.plausibility import (
    SPARSITY_TOLERANCE,
    GowerDistance,
)
from robustness_benchmark.methods.apas import APAS_IMPLEMENTATION, APASAdapter
from robustness_benchmark.methods.betarce import BETARCE_IMPLEMENTATION, BetaRCEAdapter
from robustness_benchmark.methods.configuration import (
    GROWING_SPHERES_TARGETS,
    unsupported_methods,
)
from robustness_benchmark.methods.growing_spheres import GrowingSpheresAdapter
from robustness_benchmark.methods.kdtree import KDTreeNNCE
from robustness_benchmark.methods.rbr import RBR_IMPLEMENTATION, RBRAdapter
from robustness_benchmark.methods.rnce import RNCE, RNCE_IMPLEMENTATION
from robustness_benchmark.methods.roar import ROAR_IMPLEMENTATION, RoarAdapter
from robustness_benchmark.methods.roar_lime import (
    LIME_NUM_SAMPLES,
    ROAR_LIME_IMPLEMENTATION,
    RoarLimeAdapter,
)
from robustness_benchmark.methods.robx import ROBX_IMPLEMENTATION, RobXAdapter
from robustness_benchmark.methods.trex import TREX_IMPLEMENTATION, TRexAdapter
from robustness_benchmark.methods.wachter import WACHTER_DEFAULTS, Wachter


class NoCounterfactualFound(Exception):
    """A generator completed normally but found no qualifying counterfactual."""


class UnsupportedModelError(ValueError):
    """A method cannot run on the model class it was asked to explain."""


def _float_config(config: dict[str, object], key: str, default: float) -> float:
    value = config.pop(key, default)
    if not isinstance(value, (int, float)):
        raise TypeError(f"{key} must be numeric")
    return float(value)


def _required_float_config(config: dict[str, object], key: str) -> float:
    if key not in config:
        raise ValueError(f"{key} must be provided explicitly")
    return _float_config(config, key, 0.0)


def _int_config(config: dict[str, object], key: str, default: int) -> int:
    value = config.pop(key, default)
    if not isinstance(value, int):
        raise TypeError(f"{key} must be an integer")
    return value


def _check_implementation(config: dict[str, object], name: str, expected: str) -> None:
    implementation = config.pop("implementation", expected)
    if implementation != expected:
        raise ValueError(
            f"Unsupported {name} implementation {implementation!r}; "
            f"expected {expected!r}"
        )


def make_task(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    seed: int,
    schema: FeatureSchema | None = None,
):
    return ClassificationTask(
        model, FrameDataset(X_train, y_train, seed=seed, schema=schema)
    )


def create_generator(
    name: str,
    task,
    generation_models: list | None = None,
    method_config: dict[str, object] | None = None,
):
    config = dict(method_config or {})
    unsupported = unsupported_methods([name], task.model)
    if unsupported:
        raise UnsupportedModelError(
            f"{name} cannot explain a {getattr(task.model, 'model_type', '?')} "
            f"model: it {unsupported[name]}"
        )
    if name == "kdtree":
        return KDTreeNNCE(task), config
    if name == "wachter":
        return Wachter(task), {**WACHTER_DEFAULTS, **config}
    if name == "rnce":
        _check_implementation(config, "RNCE", RNCE_IMPLEMENTATION)
        return RNCE(task), config
    if name == "apas":
        _check_implementation(config, "APAS", APAS_IMPLEMENTATION)
        return (
            APASAdapter(
                task,
                seed=int(task.training_data.seed or 0),
                delta=_required_float_config(config, "delta"),
                confidence=_float_config(config, "confidence", 0.999),
                robustness_probability=_float_config(
                    config, "robustness_probability", 0.995
                ),
                initial_margin=_float_config(config, "initial_margin", 0.01),
                increment=_float_config(config, "increment", 0.2),
                maximum_margin=_float_config(config, "maximum_margin", 20.0),
                wall_time_seconds=_float_config(config, "wall_time_seconds", 30.0),
            ),
            config,
        )
    if name == "rbr":
        _check_implementation(config, "RBR", RBR_IMPLEMENTATION)
        return (
            RBRAdapter(
                task,
                seed=int(task.training_data.seed or 0),
                num_counterfactuals=_int_config(config, "num_counterfactuals", 1_000),
                num_samples=_int_config(config, "num_samples", 200),
                perturb_radius_fraction=_float_config(
                    config, "perturb_radius_fraction", 0.2
                ),
                delta_plus=_float_config(config, "delta_plus", 0.2),
                sigma=_float_config(config, "sigma", 1.0),
                epsilon_op=_float_config(config, "epsilon_op", 0.5),
                epsilon_pe=_float_config(config, "epsilon_pe", 0.5),
                maximum_iterations=_int_config(config, "maximum_iterations", 500),
            ),
            config,
        )
    if name == "roar_lime":
        _check_implementation(config, "ROAR-LIME", ROAR_LIME_IMPLEMENTATION)
        return (
            RoarLimeAdapter(
                task,
                seed=int(task.training_data.seed or 0),
                num_samples=_int_config(config, "num_samples", LIME_NUM_SAMPLES),
                delta=_float_config(config, "delta", 0.1),
                lambda_param=_float_config(config, "lambda_param", 0.01),
                learning_rate=_float_config(config, "lr", 0.001),
                loss_threshold=_float_config(config, "loss_threshold", 0.0001),
                maximum_iterations=_int_config(config, "maximum_iterations", 20_000),
            ),
            config,
        )
    if name == "roar":
        _check_implementation(config, "ROAR", ROAR_IMPLEMENTATION)
        return (
            RoarAdapter(
                task,
                delta=_float_config(config, "delta", 0.1),
                lambda_param=_float_config(config, "lambda_param", 0.01),
                learning_rate=_float_config(config, "lr", 0.001),
                loss_threshold=_float_config(config, "loss_threshold", 0.0001),
                maximum_iterations=_int_config(config, "maximum_iterations", 20_000),
            ),
            config,
        )
    if name == "trex":
        _check_implementation(config, "T:Rex", TREX_IMPLEMENTATION)
        config.pop("tau_calibration", None)
        return (
            TRexAdapter(
                task,
                seed=int(task.training_data.seed or 0),
                variance=_float_config(config, "variance", 0.01),
                stability_samples=_int_config(config, "stability_samples", 1_000),
                tau_quantile=_float_config(config, "tau_quantile", 0.5),
                step_size=_float_config(config, "step_size", 0.01),
                max_steps=_int_config(config, "max_steps", 200),
                trust_radius=_float_config(config, "trust_radius", 1.0),
            ),
            config,
        )
    if name in GROWING_SPHERES_TARGETS:
        return (
            GrowingSpheresAdapter(
                task,
                seed=int(task.training_data.seed or 0),
                target_probability=_float_config(
                    config, "target_probability", GROWING_SPHERES_TARGETS[name]
                ),
                n_search_samples=_int_config(config, "n_search_samples", 100),
                p_norm=_int_config(config, "p_norm", 2),
                step=_float_config(config, "step", 0.1),
                max_iter=_int_config(config, "max_iter", 100),
            ),
            config,
        )
    if name in {"robx", "robx_balanced", "robx_robust"}:
        default_tau = 0.5
        if name != "robx" and "tau" not in config:
            raise ValueError(f"{name} requires a calibrated tau")
        _check_implementation(config, "RobX", ROBX_IMPLEMENTATION)
        config.pop("tau_quantile", None)
        config.pop("tau_calibration", None)
        config.pop("tau_calibration_metadata", None)

        return (
            RobXAdapter(
                task,
                seed=int(task.training_data.seed or 0),
                variance=_float_config(config, "variance", 0.01),
                tau=_float_config(config, "tau", default_tau),
                stability_samples=_int_config(config, "stability_samples", 1_000),
            ),
            config,
        )
    if name == "betarce":
        if generation_models is None:
            raise ValueError("betarce requires a separate generation_models ensemble")
        _check_implementation(config, "BetaRCE", BETARCE_IMPLEMENTATION)
        model_space = config.pop(
            "generation_model_space", "bootstrap_fixed_initialization_v2"
        )
        if model_space != "bootstrap_fixed_initialization_v2":
            raise ValueError(f"Unsupported BetaRCE model space {model_space!r}")
        return (
            BetaRCEAdapter(
                task,
                generation_models,
                seed=int(task.training_data.seed or 0),
                delta=_float_config(config, "delta", 0.9),
                alpha=_float_config(config, "alpha", 0.95),
                generation_ensemble_n=_int_config(config, "generation_ensemble_n", 32),
                gs_n_search_samples=_int_config(config, "gs_n_search_samples", 100),
                gs_p_norm=_int_config(config, "gs_p_norm", 2),
                gs_step=_float_config(config, "gs_step", 0.1),
                base_gs_max_iter=_int_config(config, "base_gs_max_iter", 100),
                gs_max_iter=_int_config(config, "gs_max_iter", 100),
            ),
            config,
        )
    raise ValueError(f"Unknown method {name!r}")


def generate_counterfactuals(
    method: str,
    task,
    factuals: pd.DataFrame,
    factual_ids: list[int] | None = None,
    dataset_row_ids: list[int | str] | None = None,
    generation_models: list | None = None,
    method_config: dict[str, object] | None = None,
    progress_description: str | None = None,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    generator, kwargs = create_generator(method, task, generation_models, method_config)
    counterfactuals: list[pd.Series] = []
    records: list[dict[str, object]] = []
    resolved_factual_ids: list[int] = (
        list(factual_ids) if factual_ids is not None else list(range(len(factuals)))
    )
    resolved_dataset_row_ids: list[int | str] = (
        list(dataset_row_ids)
        if dataset_row_ids is not None
        else list(resolved_factual_ids)
    )
    distance = GowerDistance(task.training_data.X, task.schema)

    factual_iterator = factuals.iterrows()
    if progress_description is not None:
        factual_iterator = tqdm(
            factual_iterator,
            total=len(factuals),
            desc=progress_description,
            unit="CE",
            dynamic_ncols=True,
            leave=False,
        )
    for position, (_, factual) in enumerate(factual_iterator):
        factual_id = resolved_factual_ids[position]
        dataset_row_id = resolved_dataset_row_ids[position]
        started = perf_counter()
        error: str | None = None
        status = "exception"
        method_certified: bool | None = None
        projection_changed: bool | None = None
        valid_before_projection: bool | None = None
        try:
            result = generator.generate_for_instance(factual, neg_value=0, **kwargs)
            if not isinstance(result, pd.DataFrame):
                result = pd.DataFrame(np.asarray(result).reshape(1, -1))
            if result.empty:
                raise NoCounterfactualFound
            if result.shape[1] != len(factual):
                raise ValueError(
                    f"Expected a non-empty CE with {len(factual)} features, got shape {result.shape}"
                )
            result = result.iloc[[0]].copy()
            result.columns = factual.index
            ce = result.iloc[0].astype(float)
            if np.array_equal(ce.to_numpy(), factual.to_numpy(dtype=float)):
                # RNCE falls back to returning the factual unchanged when no
                # robust candidate exists; that is a missing CE, not a CE.
                raise NoCounterfactualFound
            finite = bool(np.isfinite(ce.to_numpy()).all())
            if finite:
                # Original papers score the raw output; it is kept for comparison.
                valid_before_projection = task.model.predict_single(ce) == 1
                # Methods search in a relaxed encoding; only a valid one is a CE.
                projected = task.schema.project(ce.to_numpy(), factual.to_numpy())
                projection_changed = bool(
                    np.abs(projected - ce.to_numpy()).max(initial=0.0)
                    > SPARSITY_TOLERANCE
                )
                ce = pd.Series(projected, index=ce.index, name=ce.name)
            base_valid = finite and task.model.predict_single(ce) == 1
            status = (
                "success"
                if base_valid
                else ("base_invalid" if finite else "non_finite")
            )
            method_metadata = getattr(generator, "last_metadata", {})
        except NoCounterfactualFound:
            ce = pd.Series(np.nan, index=factual.index, dtype=float)
            base_valid = False
            method_metadata = getattr(generator, "last_metadata", {})
            status = "no_counterfactual"
        except Exception as exception:  # noqa: BLE001 - generator failure is a benchmark outcome
            error = f"{type(exception).__name__}: {exception}"
            ce = pd.Series(np.nan, index=factual.index, dtype=float)
            base_valid = False
            method_metadata = {}
        elapsed = perf_counter() - started
        counterfactuals.append(ce)
        difference = np.abs(ce.to_numpy() - factual.to_numpy())
        records.append(
            {
                "method": method,
                "factual_id": factual_id,
                "dataset_row_id": dataset_row_id,
                "runtime_seconds": elapsed,
                "generation_status": status,
                "base_valid": bool(base_valid),
                "method_certified": method_certified,
                "projection_changed": projection_changed,
                "base_valid_before_projection": valid_before_projection,
                "generation_error": error,
                **method_metadata,
                "l1_scaled_sum": float(np.nansum(difference)) if base_valid else None,
                "l1_scaled_mean": float(np.nanmean(difference)) if base_valid else None,
                "gower_distance": (
                    float(distance(ce, factual)) if base_valid else None
                ),
            }
        )

    return pd.DataFrame(counterfactuals).reset_index(drop=True), records
