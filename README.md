# Robust to Which Model Change?

Benchmark code for *Robust to Which Model Change? A Unified Evaluation of Robust Counterfactual Explanations*. The harness generates counterfactual
explanations (CFEs) for a fixed set of factual instances, holds them fixed, and
tests them against 200 held-out changed classifiers per base model, drawn from
eight model-change families. It reports coverage, base validity, empirical
robustness, end-to-end robustness, and proximity separately.

The implementation is self-contained. BetaRCE, RobX, growing spheres, RBR, and
ROAR components were adapted from the original authors' code; provenance and
licences are listed in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## Requirements

- Python 3.11 and [`uv`](https://docs.astral.sh/uv/)
- A Gurobi licence (`gurobipy` 11.0.1). RNCE, AP$\Delta$S, and the MCE search
  solve mixed-integer programs with Gurobi; the other methods do not need it.

```bash
uv sync
uv run pytest
```

All results in the paper were produced on a MacBook Pro (Apple M4 Pro). A full
run for one dataset takes several hours; AP$\Delta$S dominates the generation
time.

## Repository layout

- `robustness_benchmark/core/`: datasets and the frozen split, MLP training, the
  model-change bank, checkpoints and provenance
- `robustness_benchmark/methods/`: CFE implementations, adapters, method
  configuration, and AP$\Delta$S radius calibration
- `robustness_benchmark/evaluation/`: robustness metrics and aggregation
- `robustness_benchmark/cli/`: the three protocol commands (`model_bank`,
  `full_benchmark`, `results_packet`), a smoke test, and an appendix diagnostic
- `data/`: the three tabular datasets not shipped with scikit-learn
  (Breast Cancer is loaded from `sklearn.datasets`)
- `tests/`: unit tests for the data contract, training, metrics, and methods

## Protocol

The protocol has three steps per dataset. `<dataset>` is one of
`breast_cancer`, `diabetes`, `wine_quality`, `heloc`. Every command is
deterministic given its seeds. `full_benchmark` reuses finished runs found in
its output directory, so an interrupted benchmark can be resumed with the same
command.

### 1. Train the base models and the changed-model bank

```bash
uv run python -m robustness_benchmark.cli.model_bank \
  --dataset <dataset> \
  --epochs 300 \
  --base-architecture 32 32 \
  --base-seeds 2026 2027 2028 2029 2030 \
  --include-betarce-ensemble \
  --workers 2 \
  --output artifacts/model_bank_<dataset>
```

This freezes one stratified split (50% training, 10% reserved update pool, 10%
validation, 30% test; standardization fitted on the training partition) and
trains five base MLPs with two hidden layers of 32 ReLU units (full-batch
Adam, learning rate $10^{-3}$, at most 300 epochs, early stopping on
validation cross-entropy with patience 30). The five runs differ only in
initialization seed.

For every base model it then trains and saves 25 held-out variants in each of
eight change families (200 per base model, 1,000 per dataset):

| Family | Construction |
|---|---|
| New initialization | 25 new initialization seeds |
| Bootstrap | bootstrap resample of the training set |
| Data deletion | 1%, 5%, 10% of training rows removed (9/8/8 replicates) |
| Data addition | 25%, 50%, 75%, 100% of the update pool added (8/8/8/1) |
| Label update | 1%, 5%, 10% of training labels flipped (9/8/8) |
| Training configuration | 19 Adam and 6 SGD learning-rate/weight-decay settings |
| Architecture | 25 alternative widths and depths |
| Parameter perturbation | $\ell_\infty$ radii {0.001, 0.005, 0.01, 0.02, 0.05}, five random directions each |

All families except *new initialization* reuse the base model's
initialization seed. Every variant is evaluated; the 3-point
balanced-accuracy flag stored in the catalog is descriptive only. Each variant
is described by its hard-prediction disagreement and mean absolute probability
difference from the base model on the test set.

`--include-betarce-ensemble` additionally trains the 32 bootstrap models that
BetaRCE uses during generation (base initialization, different bootstrap
samples). They are stored separately and never used as evaluation models.

### 2. Generate counterfactuals and evaluate them on the bank

```bash
uv run python -m robustness_benchmark.cli.full_benchmark \
  --bank artifacts/model_bank_<dataset> \
  --output artifacts/full_<dataset> \
  --methods kdtree wachter apas rnce rbr roar_lime robx_balanced robx_robust betarce \
  --n-factuals 250
```

For each base model, up to 250 test instances that it predicts as adverse are
selected (all available instances on Breast Cancer and Diabetes). Every method
generates one CFE per factual, and each CFE is then scored on all 200 variants.

Method settings (`robustness_benchmark/methods/configuration.py`):

- **KD-tree**: nearest favorable training instance. **Wachter**: $\lambda$
  and learning rate grid-tuned by base-model validity, then by distance, on
  up to 25 adverse validation instances of the first base model; frozen for
  the other seeds.
- **ROAR-LIME**: the ROAR min–max objective on a local logistic LIME surrogate
  ($\delta=0.1$, 20,000 LIME samples); its objective weight is tuned like
  Wachter's.
- **RBR**: the authors' implementation with radii and budget fixed at the
  midpoints of their swept ranges; sampling radius 0.2 of the maximum pairwise
  training distance (their convention).
- **RNCE**: interval abstraction with parameter and bias radii 0.005. The
  first-layer interval uses a sign-aware bound so that it is sound for
  standardized (negative) inputs; the published encoding assumes non-negative
  inputs.
- **AP$\Delta$S**: the authors' iterative MCE margin search with
  $\alpha=0.999$, $R=0.995$ (1,378 sampled perturbations) and a 30 s wall-time
  limit per request. Its $\ell_\infty$ radius is calibrated per base model as
  the maximum parameter distance over ten one-epoch incremental updates from
  the base checkpoint on bootstrap samples of the update pool (Adam,
  learning rate $10^{-3}$, batch size 8). Because the search reuses its
  parameter sample, the selected CFE receives one additional independent
  holdout check; certified coverage is recorded separately.
- **RobX**: Gaussian local stability (variance 0.01, 1,000 samples) with a
  per-model threshold: `robx_balanced` uses the median stability of favorable
  training instances, `robx_robust` their 90th percentile.
- **BetaRCE**: growing-spheres base CFE followed by a second growing-spheres
  search that enforces the Beta lower bound ($\delta=0.9$, confidence 0.95)
  over the 32-model ensemble.

Calibration models, the BetaRCE ensemble, and tuning data are never part of
the evaluation bank. Outputs per seed and method are written to
`runs/seed_<seed>/<method>/` as Parquet files (`factuals`, `counterfactuals`,
`generation`, `survival`), with `tuning.json`, `calibration/`, and a
`summary.json` at the top level.

### 3. Aggregate the four datasets

After steps 1–2 have been completed for all four datasets:

```bash
uv run python -m robustness_benchmark.cli.results_packet \
  --artifacts artifacts \
  --output artifacts/results_packet
```

This writes `overall_metrics.csv`, `change_family_metrics.csv`,
`seed_metrics.csv`, `seed_bootstrap_intervals.csv`, the paper figures
(`paper_robustness_by_change_family.pdf`,
`paper_robustness_proximity_tradeoff.pdf`), and `results_brief.md`.

## Metrics

- **Coverage**: fraction of requests for which a method returns a finite
  candidate different from the factual.
- **Base validity**: fraction of returned candidates that the base model
  classifies in the target class.
- **Empirical robustness**: among base-valid CFEs, the fraction still
  classified in the target class by a changed model, averaged over the 25
  variants of a family (or all 200).
- **End-to-end robustness**: coverage × base validity × empirical
  robustness, i.e. the fraction of requests that yield a CFE that is valid for
  the base model and remains valid after the change.
- **Distance**: mean per-feature $\ell_1$ change on standardized features,
  each divided by the feature's training median absolute deviation (standard
  deviation when the MAD is zero).

Each metric is also computed per base model; the results packet reports the
mean and standard deviation over the five base models (`seed_summary.csv`).

## Datasets

Breast Cancer Wisconsin Diagnostic is loaded from scikit-learn. Pima Indians
Diabetes, Wine Quality, and HELOC are read from `data/`. Labels are mapped so
that 0 is the adverse and 1 the favorable outcome. For HELOC, the three
features with extensive special-value missingness are removed first, followed
by rows that still contain a negative sentinel (8,291 rows, 20 features).

## Optional commands

Quick installation check on a small model (minutes, not hours):

```bash
uv run python -m robustness_benchmark.cli.smoke \
  --methods kdtree wachter apas rnce rbr roar_lime robx \
  --n-factuals 5 --epochs 150 --output artifacts/smoke
```

`robustness_benchmark/cli/apas_betarce_pool_diagnostic.py` reproduces the appendix
diagnostic that calibrates the AP$\Delta$S radius from a pool of
*independently initialized* bootstrap models. It expects a directory with
`seed_<seed>/member_*.pt` checkpoints for each base seed; such a pool is not
produced by the protocol above and has to be trained separately.
