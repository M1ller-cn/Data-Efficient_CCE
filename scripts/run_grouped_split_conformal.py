"""Repeated grouped split-conformal diagnostic with DOI-resampled intervals.

The p=4 router is fixed before this script is executed.  Calibration seeds
change only the 75/25 split inside each outer-training DOI pool; they cannot
change model type, threshold, test DOIs, or nominal interval levels.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.sparse_router import load_calisol, make_query_sets, predict_policy

LEVELS = (0.80, 0.90, 0.95)


def radius(scores, level):
    ordered = np.sort(np.asarray(scores, dtype=float))
    index = min(len(ordered) - 1, int(np.ceil((len(ordered) + 1) * level)) - 1)
    return float(ordered[max(index, 0)])


def percentile_ci(values, rng, n_boot=20_000):
    values = np.asarray(values, dtype=float)
    draws = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calisol-csv", required=True)
    parser.add_argument("--development-rows", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n-calibration-seeds", type=int, default=30)
    parser.add_argument("--base-seed", type=int, default=20260901)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    data = load_calisol(args.calisol_csv, args.development_rows)
    queries = make_query_sets(data)
    sources = np.array(sorted(set.intersection(*(set(q.doi.unique()) for q in queries.values()))))
    rows = []
    for fold, holdout in enumerate(sources):
        train_dois = sources[sources != holdout]
        for seed_index in range(args.n_calibration_seeds):
            random_state = args.base_seed + fold * 10_000 + seed_index
            split = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=random_state)
            fit_idx, cal_idx = next(split.split(train_dois, groups=train_dois))
            fit_dois, cal_dois = train_dois[fit_idx], train_dois[cal_idx]
            for k, query_set in queries.items():
                fit = query_set[query_set.doi.isin(fit_dois)].reset_index(drop=True)
                calibration = query_set[query_set.doi.isin(cal_dois)].reset_index(drop=True)
                test = query_set[query_set.doi.eq(holdout)].reset_index(drop=True)
                cal_pred, branch, t0 = predict_policy(fit, calibration, k, p=4, seed=17)
                test_pred, _, _ = predict_policy(fit, test, k, p=4, seed=17)
                residuals = abs(calibration.target.to_numpy() - cal_pred)
                for level in LEVELS:
                    r = radius(residuals, level)
                    covered = abs(test.target.to_numpy() - test_pred) <= r
                    rows.append({
                        "held_out_doi": holdout, "outer_fold": fold, "calibration_seed": seed_index,
                        "random_state": random_state, "K": k, "branch": branch,
                        "nominal": level, "coverage": float(covered.mean()), "mean_width": 2 * r,
                        "n_fit_dois": len(fit_dois), "n_calibration_dois": len(cal_dois),
                        "n_test_queries": len(test), "vft_t0": t0,
                    })
    cells = pd.DataFrame(rows)
    cells.to_csv(out / "doi_k_seed_coverage.csv", index=False)
    rng = np.random.default_rng(args.base_seed)
    summary = []
    for level in LEVELS:
        subset = cells[cells.nominal.eq(level)]
        per_seed = subset.groupby("calibration_seed", as_index=False).agg(
            coverage=("coverage", "mean"), mean_width=("mean_width", "mean")
        )
        primary = subset[subset.calibration_seed.eq(0)].groupby("held_out_doi", as_index=False).agg(
            coverage=("coverage", "mean"), mean_width=("mean_width", "mean")
        )
        cov_lo, cov_hi = percentile_ci(primary.coverage, rng)
        width_lo, width_hi = percentile_ci(primary.mean_width, rng)
        summary.append({
            "nominal": level,
            "mean_coverage_across_calibration_seeds": float(per_seed.coverage.mean()),
            "coverage_min_across_seeds": float(per_seed.coverage.min()),
            "coverage_max_across_seeds": float(per_seed.coverage.max()),
            "primary_seed_doi_bootstrap_ci_low": cov_lo,
            "primary_seed_doi_bootstrap_ci_high": cov_hi,
            "mean_width_across_calibration_seeds": float(per_seed.mean_width.mean()),
            "primary_seed_width_bootstrap_ci_low": width_lo,
            "primary_seed_width_bootstrap_ci_high": width_hi,
            "n_doi": int(primary.held_out_doi.nunique()), "n_calibration_seeds": args.n_calibration_seeds,
        })
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(out / "coverage_stability_summary.csv", index=False)
    (out / "protocol.json").write_text(json.dumps({
        "purpose": "Calibration sensitivity and DOI-resampled uncertainty for a fixed p=4 router.",
        "policy": "RF constrained rate at K=3; VFT at K=4,5.",
        "outer_protocol": "Leave one common DOI out.",
        "calibration": "One disjoint 75/25 DOI fit/calibration split inside each outer-training pool, repeated only for sensitivity.",
        "levels": list(LEVELS), "n_calibration_seeds": args.n_calibration_seeds,
        "bootstrap_unit": "Held-out DOI, preserving the three K cells before averaging.",
    }, indent=2))
    print(summary_df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
