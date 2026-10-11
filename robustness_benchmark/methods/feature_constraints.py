"""Search-time helpers that keep candidates inside a FeatureSchema's valid set."""

from collections.abc import Callable

import numpy as np

from robustness_benchmark.core.schema import FeatureSchema

# Inputs are stored as float32, so equal raw values can differ in the last bits.
IMMUTABLE_MATCH_TOLERANCE = 1e-5


def is_constrained(schema: FeatureSchema | None) -> bool:
    """False for all-numeric, all-mutable schemas, on which methods run unchanged."""

    return schema is not None and bool(
        schema.categorical or schema.integer or schema.immutable
    )


def search_projection(
    schema: FeatureSchema | None, factual
) -> Callable[[np.ndarray], np.ndarray] | None:
    """Map a relaxed search iterate to the encoding the registry will score.

    Gradient methods judge validity on this projected point, not on the
    relaxed one, which projection can push back across the boundary. None on
    unconstrained schemas, where methods run unchanged.
    """

    if not is_constrained(schema):
        return None
    reference = np.asarray(factual, dtype=float).reshape(-1)
    return lambda candidate: schema.project(candidate, reference)


def valid_projection(
    model, project: Callable[[np.ndarray], np.ndarray], candidate, target: int = 1
) -> np.ndarray | None:
    """The projected candidate if ``model`` assigns it ``target``, else None."""

    point = project(candidate)
    return point if model.predict_single(point) == target else None


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
    """Move each group to another category with probability 1 / (len(groups) + 1).

    Every row keeps a valid one-hot encoding and changes fewer than one
    category on average, the categorical analogue of a small numeric step;
    between a third and a half of the rows keep every category.
    """

    probability = 1.0 / (len(groups) + 1)
    for group in groups:
        rows = np.flatnonzero(rng.random(len(candidates)) < probability)
        current = candidates[np.ix_(rows, group)].argmax(axis=1)
        chosen = (current + rng.integers(1, len(group), size=len(rows))) % len(group)
        candidates[np.ix_(rows, group)] = 0.0
        candidates[rows, np.asarray(group)[chosen]] = 1.0
