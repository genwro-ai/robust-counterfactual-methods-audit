"""Search-time helpers that keep candidates inside a FeatureSchema's valid set."""

import numpy as np

from robustness_benchmark.core.schema import FeatureSchema

# Inputs are stored as float32, so equal raw values can differ in the last bits.
IMMUTABLE_MATCH_TOLERANCE = 1e-5


def is_constrained(schema: FeatureSchema | None) -> bool:
    """False for all-numeric, all-mutable schemas, on which methods run unchanged."""

    return schema is not None and bool(
        schema.categorical or schema.integer or schema.immutable
    )


def matches_immutable(rows, factual, schema: FeatureSchema) -> np.ndarray:
    """Mask of the rows that share every immutable value with ``factual``."""

    columns = schema.immutable_columns()
    rows = np.asarray(rows, dtype=float).reshape(-1, len(schema.columns))
    reference = np.asarray(factual, dtype=float).reshape(-1)[columns]
    return np.isclose(
        rows[:, columns], reference, rtol=0.0, atol=IMMUTABLE_MATCH_TOLERANCE
    ).all(axis=1)


def mutable_groups(schema: FeatureSchema) -> list[list[int]]:
    """Column indices of each one-hot group a counterfactual may change."""

    return [
        group
        for feature, group in schema.one_hot_groups().items()
        if feature not in schema.immutable and len(group) > 1
    ]


def resample_categories(
    candidates: np.ndarray, groups: list[list[int]], rng: np.random.Generator
) -> None:
    """Move each group to another category with probability 1 / len(groups).

    Every row keeps a valid one-hot encoding and changes one category on
    average, the categorical analogue of a small numeric step.
    """

    for group in groups:
        rows = np.flatnonzero(rng.random(len(candidates)) < 1.0 / len(groups))
        current = candidates[np.ix_(rows, group)].argmax(axis=1)
        chosen = (current + rng.integers(1, len(group), size=len(rows))) % len(group)
        candidates[np.ix_(rows, group)] = 0.0
        candidates[rows, np.asarray(group)[chosen]] = 1.0
