# Classifier abstraction: MLP, logistic regression, XGBoost

Sub-project 1 of the full-paper extension of *Robust to Which Model Change?*
(arXiv 2609.30918). Later sub-projects (separate specs): categorical/immutable
features with German Credit, Adult and Taiwan Default (plus the ROAR-style
German data-correction shift); T:Rex and a confidence-margin baseline;
plausibility and sparsity metrics; statistical rank tests.

## Goal

Run the existing protocol (frozen split, five base seeds, held-out change bank,
fixed factuals, coverage / validity / robustness / proximity) on three model
classes: the existing MLP, logistic regression (LR), and XGBoost. Methods run
only where they natively apply; inapplicable method x model cells are reported
as n/a rather than adapted.

## Decisions

| Topic | Decision |
|---|---|
| Method gaps | Ragged matrix. Wachter needs gradients; RNCE, AP∆S (and MCE) need a torch network. None of them runs on XGBoost. |
| Base configuration | MLP unchanged (prespecified 32-32). LR and XGBoost are tuned once per dataset on the validation split over a small grid and frozen across the five seeds (best mean validation balanced accuracy over three dev seeds; simplest configuration within 0.01). |
| Change magnitude | Global test-set hard disagreement and probability MAE (unchanged). |
| New family | `incremental_update`: continue training the deployed base model on a subset of the update pool, for all three classes. |

### Change families (25 variants each)

| Family | MLP | LR | XGBoost |
|---|---|---|---|
| seed | new init seed | - (fit is deterministic) | new `random_state`; base uses `subsample = colsample_bytree = 0.8` |
| bootstrap, deletion, data_addition, label_update | shared construction (unchanged) | shared | shared |
| training_config | 19 Adam + 6 SGD lr/wd (unchanged) | C multipliers x penalty (l2, l1, elasticnet 0.5) x class weight | learning-rate multiplier x `reg_lambda` x `min_child_weight`, plus `gamma`, `reg_alpha`, `subsample` |
| architecture | widths/depths (unchanged) | - (no analogue) | 25 nearest (`max_depth`, `n_estimators`) pairs in a 6 x 5 grid |
| bounded_parameter | l_inf on weights (unchanged) | l_inf on coefficients and intercept | l_inf on leaf values (our construction) |
| incremental_update | mini-batch Adam pass from the base checkpoint | same, through the torch view | add trees fitted on the pool subset |

MLP and XGBoost have 225 variants per base model, LR 175. Incremental updates
use pool fractions 25/50/75/100% with 8/8/8/1 replicates and seeds disjoint
from the AP∆S calibration replicas. MLP and LR use the AP∆S calibration update
(1 epoch, batch 8, Adam lr 1e-3); XGBoost adds `max(1, round(0.1 x n_estimators))`
trees. Bounded-parameter radii are the MLP radii {0.001, 0.005, 0.01, 0.02,
0.05} for all classes; realized disagreement is reported, so the radii can be
revisited once the numbers are in.

## Architecture

**Classifier (inference and persistence).** `core/classifiers/base.py` defines
an abstract `Classifier`:

- `model_type`, `capabilities` (class attributes), `input_dim`, `seed`
- `predict(X) -> DataFrame` with one column P(y=1) (the existing contract, so
  every method and metric call site is unchanged)
- `predict_single`, `predict_proba`, `evaluate`, `compute_accuracy`
- `checkpoint_payload()` / `from_checkpoint_payload(payload)`

Capabilities are `DIFFERENTIABLE` (has `predict_proba_tensor`) and
`TORCH_NETWORK` (has `get_torch_model`, `hidden_dim`, `output_dim`).

- `TorchBinaryModel` (MLP) subclasses `Classifier` with both capabilities.
- `LogisticClassifier` subclasses `TorchBinaryModel` with `hidden_dim=()`. It
  fits with scikit-learn's convex solver and copies the coefficients into the
  single linear layer, so it inherits prediction, gradients, the interval
  abstraction layout and parameter perturbation.
- `XGBoostClassifier` wraps an `xgboost.Booster` and has no capabilities.

Checkpoints store `model_type`; payloads without it are legacy MLP checkpoints
and keep loading.

**ModelSpec (training and change construction).** `core/model_specs.py` defines
one spec per class (`MLPSpec`, `LogisticSpec`, `XGBoostSpec`). A spec holds
the hyperparameters and knows how to fit a model, which change families apply,
its training-configuration and architecture variants, its parameter
perturbation, its incremental update, its base-configuration grid, and its
model-id fragment (`mlp_32_32` stays unchanged). `make_changed_models` builds
the shared families once and asks the spec for the rest. The BetaRCE
generation ensemble is the spec's fit on bootstrap samples with the base seed.

**Method gating.** `methods/configuration.py` maps each method to its required
capabilities. `create_generator` raises `UnsupportedModelError` for an
unsupported pair. `full_benchmark` drops unsupported methods up front and
records them with the reason in its summary.

**Bank layout.** `model_bank --model-type {mlp,logistic,xgboost}`. The manifest
adds `model_type`, `model_spec` and the base-configuration selection report.
MLP keeps `selected_architecture`. `full_benchmark` rebuilds the spec from the
manifest and falls back to MLP for old banks.

## Fixes bundled in

The RNCE interval MILP is sign-aware only on the first hidden layer. When there
is no hidden layer (LR), the output layer reads possibly negative inputs with
plain lower and upper weight bounds. The constraints can then be wrong or
infeasible. The output layer now uses the same sign-aware encoding whenever it
is the first layer. MLP encodings are unchanged.

## Testing

- Per-class fit, predict and checkpoint round trip, including a legacy MLP payload.
- LR agrees with scikit-learn's `predict_proba`, and its torch gradients exist.
- XGBoost leaf perturbation reaches the declared l_inf radius on leaf values.
- Incremental updates leave the base model untouched; XGBoost gains trees.
- Per-class family counts, unique ids, and determinism.
- Method gating returns the expected method x model matrix.
- RNCE on LR matches the closed form `w.x + b - delta(||x||_1 + 1)` on inputs
  with negative entries.
- Base-configuration selection.

## Out of scope here

Categorical features, the new datasets, T:Rex, the margin baseline, exact ROAR
on LR, plausibility metrics, the results-packet layout across model classes,
and running the full benchmarks.
