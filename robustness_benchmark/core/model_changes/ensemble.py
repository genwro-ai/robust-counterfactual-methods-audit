"""BetaRCE's generation ensemble: bootstrap models with the base initialization."""

import numpy as np

from robustness_benchmark.core.classifiers import Classifier
from robustness_benchmark.core.data import DatasetSplit
from robustness_benchmark.core.fitting import fit_spec
from robustness_benchmark.core.model_specs import ModelSpec

BETARCE_GENERATION_ENSEMBLE_SIZE = 32
BETARCE_GENERATION_ENSEMBLE_PROTOCOL = "bootstrap_fixed_initialization_v2"
BETARCE_SAMPLING_SEED_OFFSET = 100_000


def make_betarce_generation_ensemble(
    data: DatasetSplit,
    *,
    base_seed: int,
    spec: ModelSpec,
    size: int = BETARCE_GENERATION_ENSEMBLE_SIZE,
) -> list[Classifier]:
    """Train BetaRCE's separate bootstrap ensemble with fixed initialization."""

    return [
        make_betarce_generation_model(
            data,
            base_seed=base_seed,
            member=member,
            spec=spec,
        )
        for member in range(size)
    ]


def make_betarce_generation_model(
    data: DatasetSplit,
    *,
    base_seed: int,
    member: int,
    spec: ModelSpec,
) -> Classifier:
    """Train one resumable member of BetaRCE's bootstrap model space."""

    if member < 0:
        raise ValueError("member must be nonnegative")
    n = len(data.X_train)
    sampling_seed = base_seed + BETARCE_SAMPLING_SEED_OFFSET + member
    index = np.random.default_rng(sampling_seed).integers(0, n, size=n)
    return fit_spec(
        spec,
        data.X_train.iloc[index].reset_index(drop=True),
        data.y_train.iloc[index].reset_index(drop=True),
        data.X_val,
        data.y_val,
        seed=base_seed,
    )
