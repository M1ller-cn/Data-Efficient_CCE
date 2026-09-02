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

The checked audit run is stored under `results/revision_audit/`:

1. `router_stability_20/`: 20 seeds x 19 DOI-held-out folds, including the
   complete inner-split scores, selection frequencies, candidate summaries,
   and machine-readable protocol;
2. `conformal_stability_20/`: the 20-seed grouped calibration audit, including
   DOI--support coverage cells and its summary/protocol;
3. `environment_autodl_cpu.txt`: Python and installed package versions on the
   authorized CPU-only AutoDL instance;
4. `data/derived/development_rows.npy`: the frozen row-index artifact only.

The release does not include raw source tables. Their source citations and
licensing conditions remain authoritative; see `data/README.md`.

## Main commands

```bash
python scripts/run_router_stability.py \
  --calisol-csv /path/to/calisol23_dataset.csv \
  --development-rows data/derived/development_rows.npy \
  --output results/router_stability --n-seeds 20 --n-inner-splits 3

python scripts/run_grouped_split_conformal.py \
  --calisol-csv /path/to/calisol23_dataset.csv \
  --development-rows data/derived/development_rows.npy \
  --output results/conformal_stability --n-calibration-seeds 20
```

Run these commands on the designated AutoDL CPU instance.  They are purposely
CPU-only; no GPU is required for the two sensitivity analyses.
