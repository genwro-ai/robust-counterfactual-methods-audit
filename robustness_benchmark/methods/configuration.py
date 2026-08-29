import numpy as np

from robustness_benchmark.methods.apas import APAS_IMPLEMENTATION
from robustness_benchmark.methods.rnce import RNCE_INTERVAL_ENCODING
from robustness_benchmark.methods.roar_lime import ROAR_LIME_IMPLEMENTATION

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
)
METHOD_CHOICES = (*METHODS, "robx")

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
        "delta": 0.005,
        "bias_delta": 0.005,
        "interval_encoding": RNCE_INTERVAL_ENCODING,
    },
    "rbr": {
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
        "implementation": "paper_two_stage_growing_spheres_base200_v3",
        "delta": 0.9,
        "alpha": 0.95,
        "generation_ensemble_n": 32,
        "generation_model_space": "bootstrap_fixed_initialization_v2",
        "gs_n_search_samples": 100,
        "gs_p_norm": 2,
        "gs_step": 0.1,
        "base_gs_max_iter": 200,
        "gs_max_iter": 100,
    },
    "robx_balanced": {
        "variance": 0.01,
        "tau_quantile": 0.5,
        "tau_calibration": "training_target_stability_quantile_v1",
        "stability_samples": 1_000,
    },
    "robx_robust": {
        "variance": 0.01,
        "tau_quantile": 0.9,
        "tau_calibration": "training_target_stability_quantile_v1",
        "stability_samples": 1_000,
    },
    "robx": {"variance": 0.01, "tau": 0.5, "stability_samples": 1_000},
}

TUNING_GRIDS: dict[str, list[dict[str, object]]] = {
    "wachter": [
        {"lamb": lamb, "lr": lr} for lamb in (0.01, 0.1, 1.0) for lr in (0.01, 0.02)
    ],
    "roar_lime": [
        {
            "implementation": ROAR_LIME_IMPLEMENTATION,
            "delta": 0.1,
            "lambda_param": float(lamb),
            "lr": 0.001,
            "num_samples": 20_000,
            "loss_threshold": 0.0001,
            "maximum_iterations": 20_000,
        }
        for lamb in (*[0.0005, 0.001, 0.005], *np.arange(0.01, 1.1, 0.05))
    ],
}
