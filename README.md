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

- `robustness_benchmark/core/`: datasets and the frozen split, the classifiers
  (`classifiers/`: MLP, logistic regression, XGBoost), per-class training and
  change rules (`model_specs.py`), the model-change bank, checkpoints and
  provenance
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
`breast_cancer`, `diabetes`, `wine_quality`, `heloc`, `german_credit`,
`adult`, `taiwan_default`. Every command is
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
initialization seed. The architecture is prespecified; an audit over five
widths (`base_config_selection.json`) found every candidate within 0.006
validation log-loss of the best.

For every base model it then trains and saves 25 held-out variants in each of
nine change families (225 per base model, 1,125 per dataset):

| Family | Construction |
|---|---|
| New initialization | 25 new initialization seeds |
| Bootstrap | bootstrap resample of the training set |
| Data deletion | 1%, 5%, 10% of training rows removed (9/8/8 replicates) |
| Data addition | 25%, 50%, 75%, 100% of the update pool added (8/8/8/1) |
| Label update | 1%, 5%, 10% of training labels flipped (9/8/8) |
| Training configuration | Adam and SGD learning-rate/weight-decay settings (19 Adam and 6 SGD, with reserves) |
| Architecture | alternative widths and depths |
| Parameter perturbation | $\ell_\infty$ radii {0.05, 0.1, 0.2, 0.3, 0.5} × the base model's mean absolute parameter, five random directions each |
| Incremental update | the deployed model, warm-started and trained by the same procedure on the training set plus the data-addition pool subsets (8/8/8/1); each variant is paired with the data-addition variant trained from scratch on the same data |

Training configurations and architectures are taken in a fixed order from a
candidate list longer than 25. A candidate whose validation probabilities
differ from the base model's by less than 0.001 on average reproduces the base
model and is skipped (recorded as `skipped_candidates`); the first 25 others
form the family. All families except *new initialization* reuse the base
model's initialization seed. Every variant is evaluated; the 3-point
balanced-accuracy flag stored in the catalog is descriptive only. Each variant
is described by its hard-prediction disagreement and mean absolute probability
difference from the base model on the test set.

#### Logistic regression and XGBoost

`--model-type logistic` or `--model-type xgboost` builds the same bank for
another model class (`--epochs` and `--base-architecture` apply to the MLP
only). The base configuration is selected once per dataset by a rule fixed in
advance: the grid setting with the lowest mean validation log-loss over three
development seeds (`base_config_selection.json`). The grids are C ∈ {0.001,
…, 100} for logistic regression, and `max_depth` ∈ {3, 4, 6} × learning rate ∈
{0.03, 0.1} × `n_estimators` ∈ {100, 200, 400} × `min_child_weight` ∈ {1, 5}
for XGBoost (row and column subsampling 0.8). Families adapt to the model
class:

| Family | Logistic regression | XGBoost |
|---|---|---|
| New initialization | not applicable (deterministic fit) | 25 seeds; row and column subsampling 0.8 |
| Training configuration | C from $10^{-3}$ to 100 with L2, L1, and elastic-net penalties, with and without class balancing | learning rate, `reg_lambda`, `min_child_weight`, `gamma`, `reg_alpha`, `subsample` |
| Architecture | not applicable | nearest (`max_depth`, `n_estimators`) alternatives |
| Parameter perturbation | coefficients and intercept | leaf values |
| Incremental update | not applicable (a warm start reaches the same optimum as retraining) | continued boosting on the training set plus pool subset, early-stopped on validation log-loss (at most `n_estimators` new trees) |

Bootstrap, deletion, data addition, and label update are constructed exactly as
for the MLP. `full_benchmark` skips methods the bank's model class cannot serve
(Wachter, RNCE, and AP$\Delta$S on XGBoost) and lists them under
`skipped_methods` in `summary.json`.

`--include-betarce-ensemble` additionally trains the 32 bootstrap models that
BetaRCE uses during generation (base initialization, different bootstrap
samples). They are stored separately and never used as evaluation models.

### 2. Generate counterfactuals and evaluate them on the bank

```bash
uv run python -m robustness_benchmark.cli.full_benchmark \
  --bank artifacts/model_bank_<dataset> \
  --output artifacts/full_<dataset> \
  --n-factuals 250
```

Without `--methods`, every method in `configuration.METHODS` runs; methods the
bank's model class cannot serve are skipped. For each base model, up to 250
test instances that it predicts as adverse are selected (all available
instances on Breast Cancer and Diabetes). Every method generates one CFE per
factual, and each CFE is then scored on every variant in the bank.

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
- **T:Rex** (Hamman et al., 2023; MLP and logistic regression): from a Wachter
  CFE, gradient ascent on RobX's stability score (variance 0.01, 1,000
  samples) until it reaches the `robx_balanced` threshold or 200 steps of size
  0.01 are used, within an $\ell_2$ ball of radius 1 around the start; steps,
  ball and box are in min-max-scaled units as in the reference code. It is the
  gradient-search counterpart of RobX's heuristic search.
- **ROAR** (logistic regression only): the ROAR objective on the model's own
  coefficients and intercept instead of a LIME surrogate, with ROAR-LIME's
  $\delta$ and tuning grid.
- **Growing spheres**: the non-robust baseline on every model class, with
  BetaRCE's base-search settings.
- **Confidence-margin baselines** (`gs_margin_60` to `gs_margin_90`): growing
  spheres that accept a candidate only once the base model's favorable-class
  probability reaches 0.6 to 0.9; they trace the robustness–distance frontier
  a robust method has to beat.

Wachter, RNCE, AP$\Delta$S, and T:Rex need gradients or network weights and do
not run on XGBoost; ROAR needs a linear model.

*Categorical and immutable features.* Each dataset's `FeatureSchema`
(`core/schema.py`) reaches every method through `task.schema`. A
counterfactual changes mutable features only:
- Gradient methods zero the gradients of immutable features.
- Sampling methods (growing spheres, BetaRCE, RobX) never perturb them and
  redraw categories as whole one-hot groups.
- KD-tree and RNCE give each candidate training point the factual's immutable
  values and keep it only if it is still accepted (KD-tree) or still certified
  (RNCE).
- The AP$\Delta$S MILP encodes one-hot groups as binaries and integer features
  as integers.

Every counterfactual is then projected to a valid encoding before validity is
checked: immutable features copied from the factual, one-hot groups snapped,
integers rounded. `projection_changed` records whether that changed it. On
all-numeric datasets every output is identical to earlier runs.

Calibration models, the BetaRCE ensemble, and tuning data are never part of
the evaluation bank. Outputs per seed and method are written to
`runs/seed_<seed>/<method>/` as Parquet files (`factuals`, `counterfactuals`,
`generation`, `survival`), with `tuning.json`, `calibration/`, and a
`summary.json` at the top level.

### 3. Aggregate the results

After steps 1–2 have been completed. MLP results are read from
`artifacts/full_<dataset>`, logistic regression and XGBoost results from
`artifacts/full_<dataset>_<model_type>`; every dataset and model class found
is included, and each must contain all five seeds.

```bash
uv run python -m robustness_benchmark.cli.results_packet \
  --artifacts artifacts \
  --output artifacts/results_packet
```

This writes `overall_metrics.csv`, `change_family_metrics.csv`,
`seed_metrics.csv`, `seed_summary.csv` (all with a `model_type` column),
`method_availability.md` (which methods ran on which model class), the paper
figures per model class (`paper_robustness_by_change_family_<model_type>.pdf`,
`paper_robustness_proximity_tradeoff_<model_type>.pdf`), and
`results_brief.md`. Methods a model class cannot serve appear as n/a; the
confidence-margin baselines are drawn as one dashed curve in the trade-off
plots.

The packet also writes:
- `cross_class_summary.md/csv`: mean empirical robustness, end-to-end
  robustness, and distance per method and model class, with n/a where not run.
- A robustness-versus-plausibility figure per model class
  (`paper_robustness_plausibility_<model_type>`).
- Rank tests: a Friedman test with Nemenyi critical-difference diagrams
  (Demšar 2006) for empirical robustness, end-to-end robustness, and distance,
  per model class and across the methods that run on all three. Blocks in
  which a compared method is missing are dropped and counted, and the
  confidence-margin baselines are excluded. Results are in `rank_tests.csv`
  and `critical_difference_<metric>_<scope>.pdf`.

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
- **Distance**: a Gower-style mean over raw features. A numeric feature
  contributes its standardized change divided by its training median absolute
  deviation (standard deviation when the MAD is zero); a categorical feature
  contributes 1 if its category changes. On all-numeric data this is the
  scaled $\ell_1$ distance of the paper.

- **Sparsity**: number (and fraction) of raw features changed by more than
  $10^{-6}$; a one-hot group counts once.
- **Plausibility**, all computed from the base model and training data only:
  - *kNN distance*: mean distance, scaled as for proximity, to the five
    nearest training points the base model predicts as the target class;
    `knn_distance_relative` divides it by the median leave-one-out kNN
    distance of those points, so values near 1 are as typical as real
    target-class data.
  - *LOF*: local outlier factor (20 neighbours, novelty mode) fitted on the
    same target-class points; reported as a score (about 1 is typical) and an
    inlier rate.
  - *yNN* (Pawelczyk et al., 2021): fraction of the five nearest training
    points, of any class, that the base model labels as the target class.

Sparsity and plausibility are averaged over base-valid CFEs. Each metric is
also computed per base model; the results packet reports the
mean and standard deviation over the five base models (`seed_summary.csv`).

## Datasets

Breast Cancer Wisconsin Diagnostic is loaded from scikit-learn. Pima Indians
Diabetes, Wine Quality, and HELOC are read from `data/`. Labels are mapped so
that 0 is the adverse and 1 the favorable outcome. For HELOC, the three
features with extensive special-value missingness are removed first, followed
by rows that still contain a negative sentinel (8,291 rows, 20 features).

Three datasets with categorical features come from the UCI Machine Learning
Repository (CC BY 4.0; provenance and SHA-256 in `THIRD_PARTY_NOTICES.md`):

| Dataset | Source | Rows | Favorable label |
|---|---|---|---|
| `german_credit` | South German Credit (Grömping 2019), the corrected release of Statlog German Credit | 1,000 | good credit risk |
| `adult` | UCI Adult; train and test files combined, rows with missing values dropped | 10,000 of 45,222 | income >50K |
| `taiwan_default` | Default of Credit Card Clients (Yeh & Lien 2009) | 10,000 of 30,000 | no default next month |

- Adult and Taiwan default are subsampled (stratified, seed 2026) before the
  frozen split, so that runtimes stay comparable with HELOC.
- On Adult, `fnlwgt` and `education` are removed (`education-num` is kept) and
  `native-country` is reduced to United-States vs other. On Taiwan default,
  undocumented EDUCATION and MARRIAGE codes are merged into "other".
- Nominal features are one-hot encoded as `<feature>=<category>` columns and
  are not standardized. Numeric and ordered features are standardized with
  training statistics and treated as integers.
- Immutable features follow CARLA, MACE, and Ustun et al. (2019): age,
  sex/personal status, race, nationality (native country, foreign worker), and
  marital status (marital status and relationship on Adult).

`python -m robustness_benchmark.cli.german_correction_diff` compares Statlog
German Credit with South German Credit. After aligning rows on duration,
amount, and age, 53 of 1,000 rows differ in stored values (amount, age, other
installment plans, housing, foreign worker), and no labels differ.

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
