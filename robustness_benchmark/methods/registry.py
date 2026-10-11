"""Build a method's generator from its configuration and run it on factuals."""

from collections.abc import Callable
from time import perf_counter

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from robustness_benchmark.core.schema import CHANGE_TOLERANCE, FeatureSchema
from robustness_benchmark.core.task import ClassificationTask, FrameDataset
from robustness_benchmark.methods.base import Generator
from robustness_benchmark.methods.betarce import BetaRCEAdapter
from robustness_benchmark.methods.configuration import (
    BETARCE_GENERATION_ENSEMBLE_PROTOCOL,
    GROWING_SPHERES_TARGETS,
    IMPLEMENTATIONS,
    ROBX_METHODS,
    unsupported_methods,
)
from robustness_benchmark.methods.growing_spheres import GrowingSpheresAdapter
from robustness_benchmark.methods.kdtree import KDTreeNNCE
from robustness_benchmark.methods.rbr import RBRAdapter
from robustness_benchmark.methods.roar import RoarAdapter
from robustness_benchmark.methods.roar_lime import RoarLimeAdapter
from robustness_benchmark.methods.robx import RobXAdapter
from robustness_benchmark.methods.trex import TRexAdapter
from robustness_benchmark.methods.wachter import Wachter


class NoCounterfactualFound(Exception):
    """A generator completed normally but found no qualifying counterfactual."""


class UnsupportedModelError(ValueError):
    """A method cannot run on the model class it was asked to explain."""


def make_task(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    seed: int,
    schema: FeatureSchema | None = None,
) -> ClassificationTask:
    return ClassificationTask(
        model, FrameDataset(X_train, y_train, seed=seed, schema=schema)
    )


def _settings(config: dict[str, object], *keys: str) -> dict[str, object]:
    """Take the named settings out of ``config``; every one is required."""

    missing = [key for key in keys if key not in config]
    if missing:
        raise ValueError(f"Missing method settings: {missing}")
    return {key: config.pop(key) for key in keys}


def _seed(task: ClassificationTask) -> int:
    return int(task.training_data.seed or 0)


# A factory returns the generator and the keyword arguments of every call.
Factory = Callable[
    [ClassificationTask, dict[str, object], list | None],
    tuple[Generator, dict[str, object]],
]


def _kdtree(task, _config, _models):
    return KDTreeNNCE(task), {}


def _wachter(task, config, _models):
    return Wachter(task), _settings(
        config, "lamb", "lr", "max_iter", "max_allowed_minutes"
    )


def _rnce(task, config, _models):
    # Imported here so that methods without a MILP do not need Gurobi.
    from robustness_benchmark.methods.rnce import RNCE

    return RNCE(task), _settings(
        config, "optimal", "delta", "bias_delta", "interval_encoding"
    )


def _apas(task, config, _models):
    from robustness_benchmark.methods.apas import APASAdapter

    settings = _settings(
        config,
        "delta",
        "confidence",
        "robustness_probability",
        "initial_margin",
        "increment",
        "maximum_margin",
        "wall_time_seconds",
    )
    return APASAdapter(task, seed=_seed(task), **settings), {}


def _rbr(task, config, _models):
    settings = _settings(
        config,
        "num_counterfactuals",
        "num_samples",
        "perturb_radius_fraction",
        "delta_plus",
        "sigma",
        "epsilon_op",
        "epsilon_pe",
        "maximum_iterations",
    )
    return RBRAdapter(task, seed=_seed(task), **settings), {}


ROAR_SETTINGS = (
    "delta",
    "lambda_param",
    "learning_rate",
    "loss_threshold",
    "maximum_iterations",
)


def _roar_lime(task, config, _models):
    settings = _settings(config, "num_samples", *ROAR_SETTINGS)
    return RoarLimeAdapter(task, seed=_seed(task), **settings), {}


def _roar(task, config, _models):
    return RoarAdapter(task, **_settings(config, *ROAR_SETTINGS)), {}


def _trex(task, config, _models):
    settings = _settings(
        config,
        "variance",
        "stability_samples",
        "tau_quantile",
        "step_size",
        "max_steps",
        "trust_radius",
    )
    return TRexAdapter(task, seed=_seed(task), **settings), {}


def _growing_spheres(task, config, _models):
    settings = _settings(
        config, "target_probability", "n_search_samples", "p_norm", "step", "max_iter"
    )
    return GrowingSpheresAdapter(task, seed=_seed(task), **settings), {}


def _robx(task, config, _models):
    if "tau" not in config:
        raise ValueError("RobX requires a calibrated tau")
    settings = _settings(
        config,
        "variance",
        "tau",
        "stability_samples",
        "k",
        "max_iter",
        "robx_lambda",
        "gs_n_search_samples",
        "gs_p_norm",
        "gs_step",
        "gs_max_iter",
    )
    return RobXAdapter(task, seed=_seed(task), **settings), {}


def _betarce(task, config, generation_models):
    if generation_models is None:
        raise ValueError("betarce requires a separate generation_models ensemble")
    model_space = config.pop("generation_model_space")
    if model_space != BETARCE_GENERATION_ENSEMBLE_PROTOCOL:
        raise ValueError(f"Unsupported BetaRCE model space {model_space!r}")
    settings = _settings(
        config,
        "delta",
        "alpha",
        "generation_ensemble_n",
        "gs_n_search_samples",
        "gs_p_norm",
        "gs_step",
        "base_gs_max_iter",
        "gs_max_iter",
    )
    return BetaRCEAdapter(task, generation_models, seed=_seed(task), **settings), {}


FACTORIES: dict[str, Factory] = {
    "kdtree": _kdtree,
    "wachter": _wachter,
    "rnce": _rnce,
    "apas": _apas,
    "rbr": _rbr,
    "roar_lime": _roar_lime,
    "roar": _roar,
    "trex": _trex,
    "betarce": _betarce,
    **{method: _robx for method in ROBX_METHODS},
    **{method: _growing_spheres for method in GROWING_SPHERES_TARGETS},
}


def create_generator(
    name: str,
    task: ClassificationTask,
    generation_models: list | None = None,
    method_config: dict[str, object] | None = None,
) -> tuple[Generator, dict[str, object]]:
    """The generator and per-call arguments for ``method_config``.

    Keys a method does not read (calibration provenance such as
    ``calibration_fingerprint``) stay in the recorded configuration only.
    """

    if name not in FACTORIES:
        raise ValueError(f"Unknown method {name!r}")
    unsupported = unsupported_methods([name], task.model)
    if unsupported:
        raise UnsupportedModelError(
            f"{name} cannot explain a {getattr(task.model, 'model_type', '?')} "
            f"model: it {unsupported[name]}"
        )
    config = dict(method_config or {})
    expected = IMPLEMENTATIONS.get(name)
    implementation = config.pop("implementation", expected)
    if implementation != expected:
        raise ValueError(
            f"Unsupported {name} implementation {implementation!r}; "
            f"expected {expected!r}"
        )
    return FACTORIES[name](task, config, generation_models)


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
    """One counterfactual per factual, projected to a valid encoding.

    Returns the counterfactuals (NaN rows where none was found) and one
    generation row per factual: status, validity before and after the
    projection, runtime and the method's own metadata.
    """

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
                    > CHANGE_TOLERANCE
                )
                ce = pd.Series(projected, index=ce.index, name=ce.name)
            base_valid = finite and task.model.predict_single(ce) == 1
            status = (
                "success"
                if base_valid
                else ("base_invalid" if finite else "non_finite")
            )
            method_metadata = generator.last_metadata
        except NoCounterfactualFound:
            ce = pd.Series(np.nan, index=factual.index, dtype=float)
            base_valid = False
            method_metadata = generator.last_metadata
            status = "no_counterfactual"
        except Exception as exception:  # noqa: BLE001 - generator failure is a benchmark outcome
            error = f"{type(exception).__name__}: {exception}"
            ce = pd.Series(np.nan, index=factual.index, dtype=float)
            base_valid = False
            method_metadata = {}
        elapsed = perf_counter() - started
        counterfactuals.append(ce)
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
            }
        )

    return pd.DataFrame(counterfactuals).reset_index(drop=True), records
