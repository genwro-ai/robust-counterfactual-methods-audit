"""Every method's settings, implementation ID and model-class requirements.

This module is the single source of method settings: the registry passes a
configuration's settings to the method unchanged and adds no defaults of its
own. Fixed methods use ``LOCKED_CONFIGS``; Wachter, ROAR-LIME and ROAR pick one
``TUNING_GRIDS`` entry on validation data.
"""

from collections.abc import Iterable

import numpy as np

from robustness_benchmark.core.classifiers import (
    DIFFERENTIABLE,
    LINEAR,
    TORCH_NETWORK,
    missing_capabilities,
)
from robustness_benchmark.core.model_changes import (
    BETARCE_GENERATION_ENSEMBLE_PROTOCOL,
    BETARCE_GENERATION_ENSEMBLE_SIZE,
)

# Versioned identifiers of each implementation, recorded with every run.
APAS_IMPLEMENTATION = "marzari2024_algorithm2_calibrated_radius_holdout_v4"
BETARCE_IMPLEMENTATION = "paper_two_stage_growing_spheres_base100_v4"
RBR_IMPLEMENTATION = "nguyen2022_authors_final_iterate_v2"
RNCE_IMPLEMENTATION = "jiang2024_algorithm4_l1_line_search_v1"
RNCE_INTERVAL_ENCODING = "sign_aware_first_layer_v2"
ROAR_IMPLEMENTATION = "upadhyay2021_box_logistic_exact_v2"
ROAR_LIME_IMPLEMENTATION = "upadhyay2021_box_logistic_lime_v3"
ROBX_IMPLEMENTATION = "dutta2022_algorithm1_closest_of_c_k3_v2"
TREX_IMPLEMENTATION = "hamman2023_trex_i_robx_stability_v2"

# The default of LIME's explain_instance, which AI4LIFE's lime_explanation uses.
LIME_NUM_SAMPLES = 5_000
# Fixed Wachter settings; T:Rex's base counterfactual uses them as they are.
WACHTER_DEFAULTS: dict[str, object] = {
    "max_allowed_minutes": 0.05,
    "max_iter": 20_000,
    "lr": 0.02,
    "lamb": 0.1,
}
STABILITY_TAU_CALIBRATION = "training_target_stability_quantile_v1"

# Growing spheres counts a candidate once P(y=1) reaches the target; the
# margin variants trace the robustness-distance frontier.
GROWING_SPHERES_TARGETS: dict[str, float] = {
    "growing_spheres": 0.5,
    **{f"gs_margin_{percent}": percent / 100 for percent in (60, 70, 80, 90)},
}

METHODS = (
    "kdtree",
    "wachter",
    "apas",
    "rnce",
    "rbr",
    "roar_lime",
    "robx_balanced",
    "robx_robust",
    "betarce",
    "trex",
    "roar",
    *GROWING_SPHERES_TARGETS,
)
# RobX thresholds are calibrated per base model by full_benchmark.
ROBX_METHODS = ("robx_balanced", "robx_robust")

# Methods not listed only query predictions and run on every model class.
METHOD_REQUIRED_CAPABILITIES: dict[str, frozenset[str]] = {
    "wachter": frozenset({DIFFERENTIABLE}),
    "rnce": frozenset({TORCH_NETWORK}),
    "apas": frozenset({TORCH_NETWORK}),
    "trex": frozenset({DIFFERENTIABLE}),
    "roar": frozenset({LINEAR}),
}


def unsupported_methods(
    methods: Iterable[str], model_or_capabilities: object
) -> dict[str, str]:
    """Map each method the model cannot serve to the capabilities it lacks."""

    unsupported = {}
    for method in methods:
        missing = missing_capabilities(
            model_or_capabilities, METHOD_REQUIRED_CAPABILITIES.get(method, ())
        )
        if missing:
            unsupported[method] = "requires " + ", ".join(sorted(missing))
    return unsupported


def _robx(tau_quantile: float) -> dict[str, object]:
    return {
        "implementation": ROBX_IMPLEMENTATION,
        "variance": 0.01,
        "tau_quantile": tau_quantile,
        "tau_calibration": STABILITY_TAU_CALIBRATION,
        "stability_samples": 1_000,
        "k": 3,
        "max_iter": 100,
        "robx_lambda": 0.1,
        "gs_n_search_samples": 100,
        "gs_p_norm": 2,
        "gs_step": 0.1,
        "gs_max_iter": 100,
    }


LOCKED_CONFIGS: dict[str, dict[str, object]] = {
    "kdtree": {},
    "apas": {
        "implementation": APAS_IMPLEMENTATION,
        "confidence": 0.999,
        "robustness_probability": 0.995,
        "initial_margin": 0.01,
        "increment": 0.2,
        "maximum_margin": 20.0,
        "wall_time_seconds": 30.0,
    },
    "rnce": {
        "implementation": RNCE_IMPLEMENTATION,
        "optimal": True,
        "delta": 0.005,
        "bias_delta": 0.005,
        "interval_encoding": RNCE_INTERVAL_ENCODING,
    },
    "rbr": {
        "implementation": RBR_IMPLEMENTATION,
        "num_counterfactuals": 1_000,
        "num_samples": 200,
        "perturb_radius_fraction": 0.2,
        "delta_plus": 0.2,
        "sigma": 1.0,
        "epsilon_op": 0.5,
        "epsilon_pe": 0.5,
        "maximum_iterations": 500,
    },
    "betarce": {
        "implementation": BETARCE_IMPLEMENTATION,
        "delta": 0.9,
        "alpha": 0.95,
        "generation_ensemble_n": BETARCE_GENERATION_ENSEMBLE_SIZE,
        "generation_model_space": BETARCE_GENERATION_ENSEMBLE_PROTOCOL,
        "gs_n_search_samples": 100,
        "gs_p_norm": 2,
        "gs_step": 0.1,
        "base_gs_max_iter": 100,
        "gs_max_iter": 100,
    },
    "robx_balanced": _robx(tau_quantile=0.5),
    "robx_robust": _robx(tau_quantile=0.9),
    "trex": {
        "implementation": TREX_IMPLEMENTATION,
        "variance": 0.01,
        "stability_samples": 1_000,
        "tau_quantile": 0.5,
        "tau_calibration": STABILITY_TAU_CALIBRATION,
        "step_size": 0.01,
        "max_steps": 200,
        "trust_radius": 1.0,
    },
    **{
        method: {
            "target_probability": target,
            "n_search_samples": 100,
            "p_norm": 2,
            "step": 0.1,
            "max_iter": 100,
        }
        for method, target in GROWING_SPHERES_TARGETS.items()
    },
}

# A deliberate deviation from the authors' settings. BetaRCE's config_exp.yaml
# stops every growing-spheres search after 100 shells of width 0.1, so no
# candidate lies further than 10 from the factual in standardised units, and
# in 30 dimensions random directions rarely reach a boundary beyond about half
# that radius. Where this left RobX without a start counterfactual (Breast
# Cancer on every model class, HELOC on the MLP), every growing-spheres search
# runs up to CARLA's 1,000 shells instead. Shell width and samples per shell
# are unchanged, so each search is the same up to shell 100.
EXTENDED_GS_MAX_ITER = 1_000
EXTENDED_GS_SEARCH: frozenset[tuple[str, str]] = frozenset(
    {
        ("breast_cancer", "logistic"),
        ("breast_cancer", "mlp"),
        ("breast_cancer", "xgboost"),
        ("heloc", "mlp"),
    }
)
# The settings that cap a method's growing-spheres shells. RobX's own
# ``max_iter`` counts interpolation steps and stays as it is.
GS_MAX_ITER_KEYS: dict[str, tuple[str, ...]] = {
    **{method: ("max_iter",) for method in GROWING_SPHERES_TARGETS},
    "betarce": ("base_gs_max_iter", "gs_max_iter"),
    **{method: ("gs_max_iter",) for method in ROBX_METHODS},
}


def locked_config(
    method: str, dataset: str | None = None, model_type: str | None = None
) -> dict[str, object]:
    """A fixed method's settings for one dataset and model class."""

    config = dict(LOCKED_CONFIGS.get(method, {}))
    if (dataset, model_type) in EXTENDED_GS_SEARCH:
        for key in GS_MAX_ITER_KEYS.get(method, ()):
            config[key] = EXTENDED_GS_MAX_ITER
    return config


ROAR_LAMBDAS = (0.0005, 0.001, 0.005, *np.arange(0.01, 1.1, 0.05))

TUNING_GRIDS: dict[str, list[dict[str, object]]] = {
    "wachter": [
        {**WACHTER_DEFAULTS, "lamb": lamb, "lr": lr}
        for lamb in (0.01, 0.1, 1.0)
        for lr in (0.01, 0.02)
    ],
    "roar_lime": [
        {
            "implementation": ROAR_LIME_IMPLEMENTATION,
            "delta": 0.1,
            "lambda_param": float(lamb),
            "learning_rate": 0.001,
            "num_samples": LIME_NUM_SAMPLES,
            "loss_threshold": 0.0001,
            "maximum_iterations": 20_000,
        }
        for lamb in ROAR_LAMBDAS
    ],
    "roar": [
        {
            "implementation": ROAR_IMPLEMENTATION,
            "delta": 0.1,
            "lambda_param": float(lamb),
            "learning_rate": 0.001,
            "loss_threshold": 0.0001,
            "maximum_iterations": 20_000,
        }
        for lamb in ROAR_LAMBDAS
    ],
}

IMPLEMENTATIONS: dict[str, str] = {
    method: str(config["implementation"])
    for method, config in {
        **LOCKED_CONFIGS,
        **{method: grid[0] for method, grid in TUNING_GRIDS.items()},
    }.items()
    if "implementation" in config
}


def default_config(method: str) -> dict[str, object]:
    """The locked configuration, or the first grid entry of a tuned method.

    Benchmark runs tune instead; this serves the smoke test and unit tests.
    """

    if method in TUNING_GRIDS:
        return dict(TUNING_GRIDS[method][0])
    return dict(LOCKED_CONFIGS[method])
