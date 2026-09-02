# Data-Efficient Support-Budget Routing for Sparse Electrolyte Curves

Reproducibility package for a forward-prediction study of lithium-electrolyte
conductivity curves. The method uses the information available after a small
number of ordered measurements and routes the prediction to a local
rate-normalized random forest at three supports or to a VFT curve with four or
five supports.

## Scope

This is not a composition-to-property screening model. It predicts the next
higher-temperature point of a *previously measured, single formulation*.
Source DOI groups are kept apart during development and the robotic dataset is
reserved for a frozen external score.

## Repository layout

- `src/`: data construction and the two locked prediction branches.
- `scripts/`: grouped-stability and interval-sensitivity experiments.
- `data/`: instructions for obtaining public source data; raw tables are not
  redistributed here.
- `results/`: checked manuscript-result artifacts and their protocol records.
- `docs/REPRODUCIBILITY.md`: protocol, fixed choices, and commands.

## Current status

The checked revision audit is included. It contains the frozen development-row
index, 20-seed DOI-grouped router-selection stability results, 20-seed grouped
calibration-partition sensitivity results, and the AutoDL CPU environment
record. The raw third-party tables are deliberately not redistributed; see
`data/README.md` for source access and citation requirements.

The repeated router audit covers 380 outer-fold decisions (20 seeds across 19
held-out DOI sources): $p=4$ is selected in 376 decisions (98.95%). The
calibration audit is a diagnostic of interval sensitivity, not a calibration
guarantee for a chemistry-disjoint deployment.

`scripts/export_external_decision_examples.py` produces three transparent
illustrative external cases (one per support budget) from already frozen router
predictions. The selection rule is median-error proximity within each K and is
recorded in the resulting CSV; these examples are not used for performance
reporting or model selection.

## Licence and citation

The repository code is intended for scholarly reproducibility. Source data
remain subject to the licences and citation conditions of their original
publishers and repositories; see `data/README.md`.
