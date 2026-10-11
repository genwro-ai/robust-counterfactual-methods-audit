# Categorical features, new datasets, and rank tests

Sub-project 2 of the full-paper extension (target journal: Data Mining and
Knowledge Discovery). It builds on the classifier abstraction (sub-project 1).

## Shared interface (done before the parallel work)

`core/schema.py::FeatureSchema` describes how model-input columns map to raw
features:
- numeric columns, standardized with the training mean and std
  (`numeric_scale`)
- categorical features as one-hot groups of 0/1 columns, not standardized
- `integer` numeric features, whose raw values are whole numbers
- `immutable` raw features that a counterfactual must not change

`FeatureSchema.project(candidate, factual)` returns the nearest valid encoding:
it copies immutable columns from the factual, snaps each one-hot group to its
largest entry, and rounds integer features in raw units.

The schema reaches the methods through `DatasetSplit.feature_schema` →
`make_task(..., schema=)` → `task.schema`. Datasets without a schema get an
all-numeric, all-mutable one, so existing results are unchanged.

## Datasets (agent 1)

| Name | Source | Notes |
|---|---|---|
| `german_credit` | South German Credit, the corrected release (Grömping 2019, UCI 522, CC BY 4.0) | 1,000 rows |
| `adult` | UCI Adult (UCI 2, CC BY 4.0) | stratified subsample of 10,000 rows (seed 2026) before the split |
| `taiwan_default` | Default of Credit Card Clients (Yeh & Lien 2009, UCI 350, CC BY 4.0) | stratified subsample of 10,000 rows (seed 2026) |

- Subsampling keeps runtimes comparable with HELOC. RNCE certifies every
  training point and AP∆S dominates runtime.
- Files go in `data/` with SHA-256 and licence in `THIRD_PARTY_NOTICES.md`.
- Encoding:
  - Nominal features are one-hot encoded.
  - Ordered categories become integer numeric features.
  - Immutable features follow the recourse literature (CARLA, MACE), with each
    choice documented.
- The split protocol (50/10/10/30) and the label contract (0 = adverse) are
  unchanged.
- Data-correction shift: diff the Statlog release (UCI 144) against South German
  row by row, and report whether stored values differ, not just code meanings.
  Integration into the bank is decided after that report.

## Methods with categorical features (agent 2)

- **All methods:** `generate_counterfactuals` projects every returned
  counterfactual with `schema.project` before checking base validity, and
  records whether projection changed it.
- **Search-time handling:**
  - Gradient methods (Wachter, T:Rex, ROAR, ROAR-LIME, RBR): immutable columns
    are masked from updates. Projection follows the search.
  - Sampling methods (growing spheres, BetaRCE, RobX): immutable columns are
    never perturbed, and categories are sampled as whole groups.
  - KD-tree and RNCE: candidate training points must share the factual's
    immutable values.
  - MILP (RNCE robustness check, MCE inside AP∆S): one-hot groups are binary
    variables summing to 1, and immutable inputs are fixed.
- **Distance:** a Gower-style mean over raw features. Numeric features use
  |Δ|/MAD as now; a categorical feature scores 1 if its category changes. On
  all-numeric data this equals the current `l1_robust_scale_mean`.
- **Sparsity** counts changed raw features, so a one-hot group counts once.
  Plausibility uses the same mixed distance.

## Analysis (agent 3)

- Friedman test with Nemenyi critical-difference diagrams for empirical
  robustness, end-to-end robustness and distance.
- Blocks are (dataset × change family) cells. Analyses run per model class over
  the methods that class supports, plus one cross-class analysis over the
  methods that run on all three.
- Results packet: plausibility and sparsity in tables and figures, a cross-class
  summary table, and labels for the new datasets.
