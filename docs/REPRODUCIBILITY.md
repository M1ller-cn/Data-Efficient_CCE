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

The release does not include raw source tables. Their source citations and
licensing conditions remain authoritative; see `data/README.md`.

## Main commands

```bash
python scripts/run_router_vft_gpr_audit.py \
  --calisol-csv /path/to/calisol23_dataset.csv \
  --development-rows data/development_rows.npy \
  --output results/router_vft_gpr_audit
```

The audit is CPU-only; no GPU is required. The manuscript reports the hardware
and software environment used for the checked audit.
