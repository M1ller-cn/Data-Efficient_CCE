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
- `results/`: tagged manuscript-release artifacts are placed here.
- `docs/REPRODUCIBILITY.md`: protocol, fixed choices, and commands.

## Current status

The initial release contains the executable protocol and documentation. The
frozen split indices and manuscript-result artifacts will be added only after
they are regenerated on the authorized AutoDL instance and checked against the
paper tables. This prevents an unverified local artifact from being presented
as a published result.

`scripts/export_external_decision_examples.py` produces three transparent
illustrative external cases (one per support budget) from already frozen router
predictions. The selection rule is median-error proximity within each K and is
recorded in the resulting CSV; these examples are not used for performance
reporting or model selection.

## Licence and citation

The repository code is intended for scholarly reproducibility. Source data
remain subject to the licences and citation conditions of their original
publishers and repositories; see `data/README.md`.
