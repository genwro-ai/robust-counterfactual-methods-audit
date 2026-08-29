from time import perf_counter

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from robustness_benchmark.core.task import ClassificationTask, FrameDataset
from robustness_benchmark.methods.apas import APAS_IMPLEMENTATION, APASAdapter
from robustness_benchmark.methods.kdtree import KDTreeNNCE
from robustness_benchmark.methods.rbr import RBRAdapter
from robustness_benchmark.methods.rnce import RNCE
from robustness_benchmark.methods.roar_lime import ROAR_LIME_IMPLEMENTATION, RoarLimeAdapter
from robustness_benchmark.methods.robust_adapters import BetaRCEAdapter, RobXAdapter
from robustness_benchmark.methods.wachter import Wachter


class NoCounterfactualFound(Exception):
    """A generator completed normally but found no qualifying counterfactual."""


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


def make_task(model, X_train: pd.DataFrame, y_train: pd.Series, seed: int):
    return ClassificationTask(model, FrameDataset(X_train, y_train, seed=seed))


def create_generator(
    name: str,
    task,
    generation_models: list | None = None,
    method_config: dict[str, object] | None = None,
):
    config = dict(method_config or {})
    if name == "kdtree":
        return KDTreeNNCE(task), config
    if name == "wachter":
        defaults = {
            "max_allowed_minutes": 0.05,
            "max_iter": 20_000,
            "lr": 0.02,
            "lamb": 0.1,
        }
        defaults.update(config)
        return Wachter(task), defaults
    if name == "rnce":
        return RNCE(task), config
    if name == "apas":
        implementation = config.pop("implementation", APAS_IMPLEMENTATION)
        if implementation != APAS_IMPLEMENTATION:
            raise ValueError(
                f"Unsupported APAS implementation {implementation!r}; "
                f"expected {APAS_IMPLEMENTATION!r}"
            )
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
        implementation = config.pop("implementation", ROAR_LIME_IMPLEMENTATION)
        if implementation != ROAR_LIME_IMPLEMENTATION:
            raise ValueError(
                f"Unsupported ROAR-LIME implementation {implementation!r}; "
                f"expected {ROAR_LIME_IMPLEMENTATION!r}"
            )
        return (
            RoarLimeAdapter(
                task,
                seed=int(task.training_data.seed or 0),
                num_samples=_int_config(config, "num_samples", 20_000),
                delta=_float_config(config, "delta", 0.1),
                lambda_param=_float_config(config, "lambda_param", 0.01),
                learning_rate=_float_config(config, "lr", 0.001),
                loss_threshold=_float_config(config, "loss_threshold", 0.0001),
                maximum_iterations=_int_config(config, "maximum_iterations", 20_000),
            ),
            config,
        )
    if name in {"robx", "robx_balanced", "robx_robust"}:
        default_tau = 0.5
        if name != "robx" and "tau" not in config:
            raise ValueError(f"{name} requires a calibrated tau")
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
        implementation = config.pop(
            "implementation", "paper_two_stage_growing_spheres_base200_v3"
        )
        if implementation != "paper_two_stage_growing_spheres_base200_v3":
            raise ValueError(f"Unsupported BetaRCE implementation {implementation!r}")
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
                base_gs_max_iter=_int_config(config, "base_gs_max_iter", 200),
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
    training_X = task.training_data.X
    median = training_X.median(axis=0)
    mad = (training_X - median).abs().median(axis=0)
    standard_deviation = training_X.std(axis=0, ddof=0)
    robust_scale = mad.where(mad > 1e-8, standard_deviation).where(
        lambda value: value > 1e-8, 1.0
    )

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
                "generation_error": error,
                **method_metadata,
                "l1_scaled_sum": float(np.nansum(difference)) if base_valid else None,
                "l1_scaled_mean": float(np.nanmean(difference)) if base_valid else None,
                "l1_robust_scale_mean": (
                    float(np.nanmean(difference / robust_scale.to_numpy()))
                    if base_valid
                    else None
                ),
            }
        )

    return pd.DataFrame(counterfactuals).reset_index(drop=True), records
