"""Reproducible selection of the adverse factuals a method explains."""

import pandas as pd

from robustness_benchmark.core.provenance import row_id_value


def select_adverse_indices(
    probabilities: pd.Series,
    requested: int,
    seed: int,
    threshold: float = 0.5,
) -> pd.Index:
    """Select a seeded, capped sample of model-adverse row identifiers."""

    if requested < 1:
        raise ValueError("requested must be positive")
    available = probabilities.index[probabilities < threshold].to_numpy()
    size = min(requested, len(available))
    selected = pd.Index(
        pd.Series(available).sample(n=size, random_state=seed).to_numpy()
    )
    selected.name = probabilities.index.name
    return selected


def select_factuals(
    model, X: pd.DataFrame, requested: int, seed: int
) -> tuple[pd.DataFrame, list[int], list[int | str]]:
    """Select a reproducible sample of adverse factual instances."""

    probability = model.predict(X).iloc[:, 0]
    probability.index = X.index
    indices = select_adverse_indices(probability, requested=requested, seed=seed)
    frame = X.loc[indices].reset_index(drop=True)
    factual_ids = list(range(len(frame)))
    dataset_row_ids = [row_id_value(value) for value in indices]
    return frame, factual_ids, dataset_row_ids
