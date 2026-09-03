# Support-Budget Routing for Sparse Electrolyte Conductivity Curves

Reproducibility package for a forward-prediction study of lithium-electrolyte
conductivity curves. The method uses the information available after a small
number of ordered measurements and routes the prediction to a local
rate-normalized random forest at three supports or to a VFT curve with four or
five supports.

## Scope

This is not a composition-to-property screening model. It implements a
process-oriented sequential prediction policy: after several ordered
measurements for one formulation, it estimates the prescribed next
higher-temperature point. Source DOI groups are kept apart during development,
and the robotic dataset is reserved for a frozen external score.

## Repository layout

- `src/`: data construction and the two locked prediction branches.
- `scripts/`: grouped-stability and interval-sensitivity experiments.
- `data/`: instructions for obtaining public source data; raw tables are not
  redistributed here.
- `results/`: checked manuscript-result artifacts and their protocol records.
- `docs/REPRODUCIBILITY.md`: protocol, fixed choices, and commands.

## Current status

The checked revision audit includes the frozen development-row index,
DOI-level router--VFT paired errors, query-level predictions, timing records,
and the paired-difference figure in `results/router_vft_gpr_audit/`. The audit
was run using 16 vCPUs on an Intel(R) Xeon(R) Gold 6430 processor; the manuscript
reports the corresponding software versions and timing records. Raw third-party
tables are deliberately not redistributed; see `data/README.md` for source
access and citation requirements.

The primary router--VFT audit uses leave-one-DOI-out evaluation over 19 common
source DOIs. The router has a favorable point estimate against VFT, but the
paired DOI-level difference is statistically inconclusive after Holm correction.
The interval analysis is a coverage diagnostic and calibration-sensitivity
measurement, not a deployment guarantee for chemistry-disjoint prediction.

`scripts/export_external_decision_examples.py` produces three transparent
illustrative external cases (one per support budget) from already frozen router
predictions. The selection rule is median-error proximity within each K and is
recorded in the resulting CSV; these examples are not used for performance
reporting or model selection.

## Licence and citation

The code is released under the MIT License. Source data remain subject to the
licences and citation conditions of their original publishers and repositories;
see `data/README.md`.

The citable, version-specific archival release is v1.0.1,
https://doi.org/10.5281/zenodo.22266259. The GitHub Release is available at
https://github.com/M1ller-cn/Data-Efficient_CCE/releases/tag/v1.0.1.
