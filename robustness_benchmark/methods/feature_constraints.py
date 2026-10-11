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

    Wachter judges validity on this projected point, not on the relaxed one,
    which projection can push back across the boundary. None on unconstrained
    schemas, where it runs unchanged.
    """

    if not is_constrained(schema):
        return None
    reference = np.asarray(factual, dtype=float).reshape(-1)
    return lambda candidate: schema.project(candidate, reference)


def valid_projection(
    model,
    project: Callable[[np.ndarray], np.ndarray],
    candidate,
    target: int = 1,
    margin: float = 0.0,
) -> np.ndarray | None:
    """The projected candidate if ``model`` assigns it ``target``, else None.

    ``margin`` requires the target-class probability to clear 0.5 by that much.
    """

    point = project(candidate)
    probability = float(model.predict(point).iloc[0, 0])
    valid = probability - margin >= 0.5 if target == 1 else probability + margin < 0.5
    return point if valid else None


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


def fit_category_changes_in_radius(
    candidates: np.ndarray,
    instance: np.ndarray,
    groups: list[list[int]],
    numeric: np.ndarray,
    radii: np.ndarray,
    p_norm: int,
) -> None:
    """Make category changes count toward each candidate's sampled radius.

    A category change moves two one-hot entries by 1, adding 2 to the sum of
    |change|^p. A row that cannot afford its category changes keeps the
    factual's categories; a row that can shrinks its numeric step to the rest
    of its budget, so its total p-norm change equals its radius.
    """

    reference = instance.reshape(-1)
    changed = sum(
        candidates[:, group].argmax(axis=1) != reference[group].argmax()
        for group in groups
    )
    budget = radii**p_norm - 2.0 * changed
    unaffordable = np.flatnonzero(budget < 0)
    for group in groups:
        candidates[np.ix_(unaffordable, group)] = reference[group]
    shrink = np.flatnonzero((budget >= 0) & (changed > 0))
    if numeric.size and shrink.size:
        scale = budget[shrink] ** (1.0 / p_norm) / radii[shrink]
        rows = np.ix_(shrink, numeric)
        candidates[rows] = (
            reference[numeric]
            + (candidates[rows] - reference[numeric]) * scale[:, None]
        )
