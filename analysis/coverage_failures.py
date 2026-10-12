"""Why do some methods return no counterfactual? Read-only audit of a full run.

Reads ``artifacts/full_<dataset>_<model_type>/`` (pooled generation and
survival tables plus the per-run KD-tree factuals and counterfactuals) and
writes CSV tables to ``analysis/output/coverage_failures/``:

- ``failure_causes.csv``: every ``no_counterfactual`` row by method, dataset,
  model type and cause.
- ``gs_failure_by_distance.csv``: growing-spheres failure rate against the L2
  distance (standardised features) from the factual to its KD-tree
  counterfactual, the nearest target-class training point.
- ``gs_hit_probability.csv``: the probability that the benchmark's growing
  spheres search (step 0.1, 100 shells, 100 samples per shell) finds a
  half-space at distance d in D dimensions.
- ``apas_timeouts_by_base_model.csv`` and ``xgboost_margin_failures.csv``:
  per-base-model failure rates.
- ``selection_effect_breast_cancer.csv``: robustness of always-covering
  methods on the factuals plain growing spheres could and could not serve.

Usage: ``.venv/bin/python analysis/coverage_failures.py [artifacts_dir]``
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import beta

ARTIFACTS = Path(sys.argv[1] if len(sys.argv) > 1 else "artifacts")
OUT = Path(__file__).parent / "output" / "coverage_failures"

GENERATION_COLUMNS = [
    "method",
    "factual_id",
    "base_model_id",
    "generation_status",
    "gs_status",
    "robx_status",
    "betarce_status",
    "apas_outcome",
    "apas_iterations",
    "runtime_seconds",
    "rbr_outcome",
]
GS_FAMILY = [
    "growing_spheres",
    "gs_margin_60",
    "gs_margin_70",
    "gs_margin_80",
    "gs_margin_90",
    "betarce",
    "robx_balanced",
    "robx_robust",
]
# The benchmark's growing-spheres settings (BetaRCE's base search).
GS_STEP, GS_SHELLS, GS_SAMPLES = 0.1, 100, 100


def run_dirs():
    for directory in sorted(ARTIFACTS.glob("full_*_*")):
        dataset, model_type = directory.name[len("full_") :].rsplit("_", 1)
        yield directory, dataset, model_type


def load_generation() -> pd.DataFrame:
    frames = []
    for directory, dataset, model_type in run_dirs():
        path = directory / "generation.parquet"
        names = pq.read_schema(path).names
        frame = pd.read_parquet(
            path, columns=[c for c in GENERATION_COLUMNS if c in names]
        )
        frames.append(frame.assign(dataset=dataset, model_type=model_type))
    frame = pd.concat(frames, ignore_index=True)
    frame["seed"] = frame.base_model_id.str.rsplit("_", n=1).str[-1]
    frame["failed"] = frame.generation_status.eq("no_counterfactual")
    return frame


def failure_cause(row: pd.Series, plain_gs_failed: set) -> str:
    method = row.method
    key = (row.dataset, row.model_type, row.seed, row.factual_id)
    if method == "apas":
        return f"apas_{row.apas_outcome}"
    if method == "rbr":
        return "rbr_single_class_neighbourhood"
    if method == "betarce":
        if row.betarce_status == "base_counterfactual_not_found":
            return "gs_search_radius"
        return "betarce_second_stage"
    if method.startswith("robx"):
        return "gs_search_radius"
    if method == "growing_spheres":
        return "gs_search_radius"
    if method.startswith("gs_margin"):
        return "gs_search_radius" if key in plain_gs_failed else "margin_not_reached"
    return "other"


def failure_causes(generation: pd.DataFrame) -> pd.DataFrame:
    failed = generation[generation.failed]
    plain = failed[failed.method.eq("growing_spheres")]
    plain_gs_failed = set(
        zip(plain.dataset, plain.model_type, plain.seed, plain.factual_id)
    )
    causes = failed.apply(failure_cause, axis=1, plain_gs_failed=plain_gs_failed)
    table = (
        failed.assign(cause=causes)
        .groupby(["cause", "method", "model_type", "dataset"])
        .size()
        .rename("n_failed")
        .reset_index()
    )
    requested = (
        generation.groupby(["method", "model_type", "dataset"]).size().rename("n")
    )
    table = table.join(requested, on=["method", "model_type", "dataset"])
    table["failure_rate"] = table.n_failed / table.n
    return table


def kdtree_distances() -> pd.DataFrame:
    frames = []
    for directory, dataset, model_type in run_dirs():
        for run in sorted((directory / "runs").glob("seed_*")):
            kd = run / "kdtree"
            factuals = pd.read_parquet(kd / "factuals.parquet").to_numpy(float)
            counterfactuals = pd.read_parquet(kd / "counterfactuals.parquet")
            ids = pd.read_parquet(kd / "generation.parquet", columns=["factual_id"])
            frames.append(
                pd.DataFrame(
                    {
                        "dataset": dataset,
                        "model_type": model_type,
                        "seed": run.name.split("_")[-1],
                        "factual_id": ids.factual_id.to_numpy(),
                        "kdtree_l2": np.linalg.norm(
                            factuals - counterfactuals.to_numpy(float), axis=1
                        ),
                        "n_features": factuals.shape[1],
                    }
                )
            )
    return pd.concat(frames, ignore_index=True)


def gs_failure_by_distance(generation, distances) -> pd.DataFrame:
    gs = generation[generation.method.isin(GS_FAMILY)].merge(
        distances, on=["dataset", "model_type", "seed", "factual_id"]
    )
    bins = [0, 2, 4, 6, 8, 10, 15, np.inf]
    gs["kdtree_l2_bin"] = pd.cut(gs.kdtree_l2, bins).astype(str)
    return (
        gs.groupby(["dataset", "model_type", "method", "kdtree_l2_bin"])
        .failed.agg(failure_rate="mean", n="size")
        .reset_index()
    )


def gs_hit_probability(distance: float, dimension: int) -> float:
    """Chance that some candidate lands beyond a half-space at ``distance``.

    A uniform direction u in R^D satisfies (u_1 + 1) / 2 ~ Beta((D-1)/2,
    (D-1)/2); shell k samples radii uniformly in (k * step, (k + 1) * step].
    """

    shape = (dimension - 1) / 2
    log_miss = 0.0
    for shell in range(GS_SHELLS):
        radii = np.linspace(shell * GS_STEP, (shell + 1) * GS_STEP, 21)[1:]
        cosine = np.clip(distance / radii, -1, 1)
        hit = beta.sf((cosine + 1) / 2, shape, shape).mean()
        log_miss += GS_SAMPLES * np.log1p(-min(hit, 1 - 1e-15))
    return 1 - np.exp(log_miss)


def hit_probability_table(distances) -> pd.DataFrame:
    dimensions = sorted(set(distances.n_features) | {30})
    rows = [
        {"dimension": d, "distance": r, "hit_probability": gs_hit_probability(r, d)}
        for d in dimensions
        for r in np.arange(1.0, 10.5, 0.5)
    ]
    return pd.DataFrame(rows)


def apas_timeouts(generation) -> pd.DataFrame:
    apas = generation[generation.method.eq("apas")]
    return (
        apas.assign(timeout=apas.apas_outcome.eq("timeout"))
        .groupby(["model_type", "dataset", "base_model_id"])
        .agg(
            n=("timeout", "size"),
            timeouts=("timeout", "sum"),
            timeout_rate=("timeout", "mean"),
            median_runtime_completed=(
                "runtime_seconds",
                lambda s: s[~apas.loc[s.index, "apas_outcome"].eq("timeout")].median(),
            ),
        )
        .reset_index()
    )


def xgboost_margin_failures(generation) -> pd.DataFrame:
    margins = generation[
        generation.model_type.eq("xgboost")
        & generation.method.isin(["growing_spheres"] + GS_FAMILY[1:5])
    ]
    return margins.pivot_table(
        index=["dataset", "base_model_id"],
        columns="method",
        values="failed",
        aggfunc="mean",
    ).reset_index()


def selection_effect(generation) -> pd.DataFrame:
    plain = generation[
        generation.method.eq("growing_spheres") & generation.dataset.eq("breast_cancer")
    ][["model_type", "seed", "factual_id", "failed"]]
    frames = []
    for model_type in ["logistic", "mlp", "xgboost"]:
        survival = pd.read_parquet(
            ARTIFACTS / f"full_breast_cancer_{model_type}" / "survival.parquet",
            columns=[
                "method",
                "base_model_id",
                "factual_id",
                "conditional_eligible",
                "conditional_survival",
            ],
        )
        survival = survival[survival.conditional_eligible]
        survival = survival.assign(
            model_type=model_type,
            seed=survival.base_model_id.str.rsplit("_", n=1).str[-1],
            survived=survival.conditional_survival.astype(float),
        )
        frames.append(
            survival.merge(plain, on=["model_type", "seed", "factual_id"])
            .groupby(["model_type", "method", "failed"])
            .survived.mean()
            .unstack()
            .rename(columns={False: "gs_served", True: "gs_failed"})
            .reset_index()
        )
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    generation = load_generation()
    distances = kdtree_distances()
    tables = {
        "failure_causes": failure_causes(generation),
        "gs_failure_by_distance": gs_failure_by_distance(generation, distances),
        "gs_hit_probability": hit_probability_table(distances),
        "apas_timeouts_by_base_model": apas_timeouts(generation),
        "xgboost_margin_failures": xgboost_margin_failures(generation),
        "selection_effect_breast_cancer": selection_effect(generation),
    }
    for name, table in tables.items():
        table.to_csv(OUT / f"{name}.csv", index=False)
    causes = tables["failure_causes"]
    print(causes.groupby("cause").n_failed.sum().sort_values(ascending=False))
    print("total no_counterfactual:", int(generation.failed.sum()))


if __name__ == "__main__":
    main()
