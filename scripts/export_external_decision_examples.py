"""Export transparent, non-tuned external decision examples.

For each support budget, the script selects the router prediction whose absolute
error is nearest that budget's median router error.  The examples are therefore
illustrative typical cases, not a secondary performance metric or a new model
selection step.  Their full observed prefixes are exported so a reader can see
what information the policy had at the time of prediction.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def load_external(path: str) -> pd.DataFrame:
    raw = pd.read_csv(path, sep=";", skiprows=[1, 2], low_memory=False)
    required = {"experimentID", "temperature", "EIS_conductivity"}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"Unexpected external data schema; missing {sorted(missing)}")
    curve = raw[["experimentID", "temperature", "EIS_conductivity"]].copy()
    curve.columns = ["id", "T", "k"]
    curve["id"] = curve.id.astype(str)
    curve["T"] = pd.to_numeric(curve["T"], errors="coerce") + 273.15
    curve["k"] = pd.to_numeric(curve.k, errors="coerce")
    curve = curve[
        np.isfinite(curve["T"]) & np.isfinite(curve["k"]) & (curve["k"] > 0)
    ].copy()
    curve["log10_sigma"] = np.log10(curve.k)
    return curve


def prefix(curves: pd.DataFrame, ident: str, query_index: int, k: int) -> tuple[str, str]:
    curve = curves[curves.id.eq(str(ident))].groupby("T", as_index=False).log10_sigma.mean().sort_values("T")
    observed = curve.iloc[query_index - k:query_index]
    return (
        "; ".join(f"{value:.2f}" for value in observed["T"].to_numpy(float)),
        "; ".join(f"{value:.4f}" for value in observed.log10_sigma.to_numpy(float)),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--external-csv", required=True)
    parser.add_argument("--router-predictions", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    curves = load_external(args.external_csv)
    predictions = pd.read_csv(args.router_predictions)
    router = predictions[predictions.model.eq("nested_budget_router")].copy()
    if router.empty:
        raise ValueError("No nested_budget_router rows were found.")
    examples = []
    for k, subset in router.groupby("K", sort=True):
        median_error = subset.abs_error.median()
        example = subset.iloc[(subset.abs_error - median_error).abs().argmin()].copy()
        temps, values = prefix(curves, example.id, int(example.query_index), int(k))
        examples.append({
            "selection_rule": "nearest to within-K median router absolute error; illustration only",
            "id": example.id,
            "K": int(k),
            "branch": "RF constrained rate" if int(k) == 3 else "VFT curve",
            "support_temperatures_K": temps,
            "support_log10_sigma": values,
            "query_temperature_K": float(example.query_T),
            "observed_log10_sigma": float(example.target),
            "router_prediction_log10_sigma": float(example.prediction),
            "absolute_error_log10_sigma": float(example.abs_error),
        })
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(examples).to_csv(output, index=False)
    print(pd.DataFrame(examples).to_string(index=False))


if __name__ == "__main__":
    main()
