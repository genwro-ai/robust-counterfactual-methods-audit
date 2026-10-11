"""Friedman test with Nemenyi post-hoc comparison of methods over blocks.

Follows Demšar (2006), "Statistical comparisons of classifiers over multiple
data sets", JMLR 7:1-30. Methods are ranked within each block, the Friedman
test asks whether their average ranks differ, and two methods differ when their
average ranks are further apart than the Nemenyi critical difference.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from scipy import stats


class InsufficientData(ValueError):
    """Too few methods or complete blocks for a Friedman test."""


@dataclass(frozen=True)
class RankTestResult:
    """Outcome of one analysis; ``average_ranks`` runs from best (1) to worst."""

    average_ranks: pd.Series
    n_blocks: int
    n_blocks_dropped: int
    statistic: float
    p_value: float
    critical_difference: float
    alpha: float

    @property
    def k_methods(self) -> int:
        return len(self.average_ranks)

    def cliques(self) -> list[tuple[int, int]]:
        """Maximal runs of rank-ordered methods that are not significantly different.

        A rank difference of at least CD is significant (Demšar 2006). Each pair
        is an inclusive index range into ``average_ranks``.
        """

        ranks = self.average_ranks.to_numpy()
        cliques: list[tuple[int, int]] = []
        for start in range(len(ranks)):
            end = (
                int(
                    np.searchsorted(
                        ranks, ranks[start] + self.critical_difference, "left"
                    )
                )
                - 1
            )
            if end > start and (not cliques or end > cliques[-1][1]):
                cliques.append((start, end))
        return cliques


def nemenyi_critical_difference(k: int, n_blocks: int, alpha: float = 0.05) -> float:
    """CD = q_alpha * sqrt(k (k + 1) / (6 N)), with q_alpha from the studentized range.

    Demšar's table lists q_alpha = studentized-range quantile (infinite degrees
    of freedom) divided by sqrt(2); computing it avoids the table's k <= 10 limit.
    """

    q_alpha = stats.studentized_range.ppf(1 - alpha, k, np.inf) / math.sqrt(2)
    return float(q_alpha * math.sqrt(k * (k + 1) / (6 * n_blocks)))


def rank_test(
    scores: pd.DataFrame,
    *,
    block: str | list[str],
    method: str,
    score: str,
    higher_is_better: bool,
    alpha: float = 0.05,
) -> RankTestResult:
    """Friedman test and Nemenyi critical difference over complete blocks.

    ``scores`` is long: one row per (block, method). Ties share their average
    rank. Blocks where any method is missing or non-finite are dropped, so every
    method is compared on the same blocks; the number dropped is reported.
    """

    values = pd.to_numeric(scores[score], errors="coerce").where(np.isfinite)
    wide = scores.assign(**{score: values}).pivot(
        index=block, columns=method, values=score
    )
    complete = wide.dropna()
    k, n_blocks = complete.shape[1], len(complete)
    if k < 3 or n_blocks < 2:
        raise InsufficientData(
            f"Friedman test needs >=3 methods and >=2 complete blocks, "
            f"got {k} and {n_blocks}"
        )
    ranks = complete.rank(axis=1, ascending=not higher_is_better, method="average")
    if (complete.nunique(axis=1) == 1).all():
        statistic, p_value = 0.0, 1.0
    else:
        statistic, p_value = stats.friedmanchisquare(*complete.T.to_numpy())
    return RankTestResult(
        average_ranks=ranks.mean().sort_values(kind="stable"),
        n_blocks=n_blocks,
        n_blocks_dropped=len(wide) - n_blocks,
        statistic=float(statistic),
        p_value=float(p_value),
        critical_difference=nemenyi_critical_difference(k, n_blocks, alpha),
        alpha=alpha,
    )


def plot_critical_difference(
    result: RankTestResult,
    *,
    title: str,
    labels: Mapping[str, str] | None = None,
    colors: Mapping[str, str] | None = None,
    ink: str = "black",
    muted: str = "0.4",
) -> Figure:
    """Demšar's critical-difference diagram: rank 1 at the right, bars join cliques.

    The better half of the methods is labelled on the right, the rest on the
    left; each label hangs from its average rank on the axis.
    """

    labels, colors = labels or {}, colors or {}
    ranks = result.average_ranks
    k = len(ranks)
    cliques = result.cliques()
    row_height, unit_inches = 0.55, 0.3
    first_row = -0.5 - 0.22 * len(cliques)
    rows = math.ceil(k / 2)
    bottom = first_row - row_height * rows

    figure, axis = plt.subplots(
        figsize=(7.2, unit_inches * (1.8 - bottom)), facecolor="white"
    )
    figure.subplots_adjust(left=0.28, right=0.72, bottom=0.01, top=0.88)
    axis.set(xlim=(k + 0.3, 0.7), ylim=(bottom, 1.4))
    axis.axis("off")
    axis.set_title(title, loc="left", fontsize=9.2, fontweight="bold", color=ink)

    def line(x, y, **kwargs):
        axis.plot(x, y, color=kwargs.pop("color", ink), clip_on=False, **kwargs)

    line([1, k], [0, 0], linewidth=1.0)
    for tick in range(1, k + 1):
        line([tick, tick], [0, 0.12], linewidth=0.8)
        axis.text(
            tick, 0.2, str(tick), ha="center", va="bottom", fontsize=6.8, color=muted
        )
    line([k, k - result.critical_difference], [1.0, 1.0], linewidth=1.6)
    axis.text(
        k - result.critical_difference / 2,
        1.1,
        f"CD = {result.critical_difference:.2f}",
        ha="center",
        va="bottom",
        fontsize=6.8,
        color=ink,
    )

    for level, (start, end) in enumerate(cliques):
        y = -0.25 - 0.22 * level
        line([ranks.iloc[start] - 0.04, ranks.iloc[end] + 0.04], [y, y], linewidth=2.4)

    for position, (name, rank) in enumerate(ranks.items()):
        best_half = position < rows
        row = position if best_half else k - 1 - position
        y = first_row - row_height * row
        edge = 0.55 if best_half else k + 0.45
        line([rank, rank, edge], [0, y, y], linewidth=0.6, color=muted)
        axis.scatter(
            rank, 0, s=22, color=colors.get(name, ink), zorder=4, clip_on=False
        )
        axis.text(
            edge + (-0.08 if best_half else 0.08),
            y,
            f"{labels.get(name, name)} ({rank:.2f})",
            ha="left" if best_half else "right",
            va="center",
            fontsize=7,
            color=ink,
        )
    return figure
