# Reproducibility protocol

## What this package reproduces

The package implements the manuscript's forward prediction setting: predict the
next higher-temperature log-conductivity for one already observed electrolyte
formulation from its preceding three, four, or five temperatures.  Formula
composition identifiers and target-side information are not input features.

## Locked protocol

- **Development data:** CALiSol-23, filtered and grouped by source DOI.
- **Scoring unit:** held-out DOI, then support count, with equal weighting.
- **Router candidates:** `p in {4, 5, 6}`. A candidate uses RF below `p`
  supports and VFT at `p` or more supports.
- **External collection:** never used to choose a threshold, feature, VFT grid,
  model setting, or interval level.
- **Random forest:** 500 trees, minimum leaf size 2, maximum feature fraction
  0.8, seed 17, one CPU thread.

## Fixed-terminal support-budget sensitivity

This supplementary analysis is intentionally separate from the main
next-temperature benchmark. For each formulation with at least six distinct
temperatures, it retains the final six-point segment once. The K=3, 4, and 5
nested support prefixes all predict the same sixth and final target. Thus the
comparison does not change the target temperature or multiply rolling windows.

For each outer held-out DOI, the threshold `p in {4, 5, 6}` is selected with
three grouped inner DOI splits. For every pre-specified additional-support cost
`c in {0, 0.0025, 0.005, 0.01, 0.02}`, a single budget is selected inside the
outer-training pool by minimizing `inner_MAE(K) + c * (K - 3)`. The held-out
DOI is used only for evaluation. This reports a cost-sensitive
value-of-measurement frontier; it is not a curve-specific stopping rule or an
active-learning claim.

## Included revision-audit artifacts

The checked router--VFT/GPR audit is stored under
`results/router_vft_gpr_audit/`:

1. `doi_macro_errors.csv`: DOI-macro errors for the router, VFT, and the
   independent per-curve GPR;
2. `query_predictions.csv`: query-level predictions and absolute errors;
3. `timing_by_outer_fold.csv`: fit-and-predict timing records for each outer
   DOI fold and support budget;
4. `router_vs_vft_doi_distribution.png`: the paired DOI-level comparison
   figure;
5. `data/development_rows.npy`: the frozen development-row index artifact.

The fixed-terminal support-budget artifacts are stored under
`results/fixed_target_support_budget/`:

1. `fixed_target_predictions.csv`: one terminal-target prediction per
   formulation, support budget, and applicable cost-policy evaluation;
2. `fixed_target_doi_mae.csv`: DOI-level MAE values for the fixed-target
   router and cost-sensitive policy;
3. `fixed_target_router_selection.csv` and `measurement_cost_selection.csv`:
   outer-fold threshold and support-budget selections from training DOI pools;
4. `measurement_cost_frontier.csv` and `fixed_target_cost_frontier.png`: the
   complete pre-specified held-out cost frontier;
5. `protocol.json`: query construction and selection constraints.

The release does not include raw source tables. Their source citations and
licensing conditions remain authoritative; see `data/README.md`.

## Main commands

```bash
python scripts/run_router_vft_gpr_audit.py \
  --calisol-csv /path/to/calisol23_dataset.csv \
  --development-rows data/development_rows.npy \
  --output results/router_vft_gpr_audit
```

```bash
python scripts/run_fixed_target_support_budget_audit.py \
  --calisol-csv /path/to/calisol23_dataset.csv \
  --development-rows data/development_rows.npy \
  --output results/fixed_target_support_budget
```

The audit is CPU-only; no GPU is required. The manuscript reports the hardware
and software environment used for the checked audit.
