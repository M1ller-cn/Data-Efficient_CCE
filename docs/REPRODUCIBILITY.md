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

## Required artifacts for a manuscript release

Before submission, the release must include:

1. the frozen development-row index and DOI split list;
2. each result CSV cited by a table or figure;
3. exact package versions and OS/Python details;
4. scripts used to make manuscript figures;
5. the public-source citation and licence information for every raw dataset.

## Main commands

```bash
python scripts/run_router_stability.py \
  --calisol-csv /path/to/calisol23_dataset.csv \
  --development-rows data/derived/development_rows.npy \
  --output results/router_stability --n-seeds 50

python scripts/run_grouped_split_conformal.py \
  --calisol-csv /path/to/calisol23_dataset.csv \
  --development-rows data/derived/development_rows.npy \
  --output results/conformal_stability --n-calibration-seeds 30
```

Run these commands on the designated AutoDL CPU instance.  They are purposely
CPU-only; no GPU is required for the two sensitivity analyses.
