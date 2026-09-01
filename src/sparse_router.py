"""Data construction and locked branches used by the C&CE manuscript.

The functions are deliberately free of chemistry identifiers at inference.
They only use the observed prefix of one formulation-temperature curve plus
the requested query temperature.  DOI is retained solely for grouped splits.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


T0_GRID = np.arange(50.0, 191.0, 10.0)
FEATURES = [
    "query_T", "K", "anchor_T", "anchor_logk", "arrhenius_pred",
    "linearT_pred", "arrhenius_slope", "arrhenius_intercept", "delta_T",
    "delta_invT_1e4",
]


def formulation_key(row: pd.Series, solvent_columns: Iterable[str]) -> str:
    """Build the documented formulation key used before curve construction."""
    prefix = (
        f"{row.doi}|{row.salt}|{float(row.c):.6g}|{row['c units']}|"
        f"{row['solvent ratio type']}|"
    )
    return prefix + ",".join(f"{float(row[col]):.6g}" for col in solvent_columns)


def load_calisol(path: str, development_rows: str | None = None) -> pd.DataFrame:
    """Load and filter CALiSol-23, optionally using a frozen row-index file."""
    raw = pd.read_csv(path)
    raw["y"] = np.log10(raw.k.clip(lower=1e-12))
    raw = raw[
        np.isfinite(raw.y)
        & np.isfinite(raw["T"])
        & raw.doi.notna()
        & raw.salt.notna()
        & raw.c.notna()
    ].copy()
    solvents = [col for col in raw.columns[7:] if col != "y"]
    raw["group"] = raw.apply(formulation_key, axis=1, solvent_columns=solvents)
    if development_rows:
        rows = np.load(development_rows)
        raw = raw.iloc[rows].copy()
    return raw.reset_index(drop=True)


def make_terminal_queries(data: pd.DataFrame, k: int) -> pd.DataFrame:
    """Make one next-temperature forward query from each eligible curve.

    The terminal held-out point is never available to the features. Replicates
    at equal temperature are averaged before support construction.
    """
    rows: list[dict[str, float | str | int]] = []
    for group, curve in data.groupby("group", sort=False):
        curve = (
            curve.groupby("T", as_index=False).y.mean().sort_values("T")
            .reset_index(drop=True)
        )
        if len(curve) < k + 1:
            continue
        support, query = curve.iloc[-(k + 1):-1], curve.iloc[-1]
        st, sy = support["T"].to_numpy(float), support.y.to_numpy(float)
        arr = np.polyfit(1.0 / st, sy, 1)
        linear = np.polyfit(st, sy, 1)
        record: dict[str, float | str | int] = {
            "doi": data.loc[data.group.eq(group), "doi"].iloc[0],
            "group": group,
            "K": k,
            "target": float(query.y),
            "query_T": float(query["T"]),
            "anchor_T": float(st[-1]),
            "anchor_logk": float(sy[-1]),
            "arrhenius_pred": float(np.polyval(arr, 1.0 / float(query["T"]))),
            "linearT_pred": float(np.polyval(linear, float(query["T"]))),
            "arrhenius_slope": float(arr[0]),
            "arrhenius_intercept": float(arr[1]),
        }
        for t0 in T0_GRID:
            a, b = np.polyfit(1.0 / (st - t0), sy, 1)
            record[f"vft_{int(t0)}"] = float(a / (float(query["T"]) - t0) + b)
        rows.append(record)
    result = pd.DataFrame(rows)
    result["delta_T"] = result.query_T - result.anchor_T
    result["delta_invT_1e4"] = (
        1.0 / result.anchor_T - 1.0 / result.query_T
    ) * 1e4
    return result


def make_query_sets(data: pd.DataFrame, ks: Iterable[int] = (3, 4, 5)) -> dict[int, pd.DataFrame]:
    queries = {int(k): make_terminal_queries(data, int(k)) for k in ks}
    common = set.intersection(*(set(q.doi.unique()) for q in queries.values()))
    return {k: q[q.doi.isin(common)].reset_index(drop=True) for k, q in queries.items()}


def choose_vft_t0(train: pd.DataFrame) -> float:
    maes = [
        np.mean(np.abs(train.target.to_numpy() - train[f"vft_{int(t0)}"].to_numpy()))
        for t0 in T0_GRID
    ]
    return float(T0_GRID[int(np.argmin(maes))])


def predict_constrained_rf(train: pd.DataFrame, test: pd.DataFrame, seed: int = 17) -> np.ndarray:
    rate = (train.target - train.anchor_logk) / train.delta_invT_1e4
    model = RandomForestRegressor(
        n_estimators=500, min_samples_leaf=2, max_features=0.8,
        random_state=seed, n_jobs=1,
    ).fit(train[FEATURES], rate)
    return test.anchor_logk.to_numpy() + model.predict(test[FEATURES]) * test.delta_invT_1e4.to_numpy()


def predict_policy(train: pd.DataFrame, test: pd.DataFrame, k: int, p: int, seed: int = 17) -> tuple[np.ndarray, str, float | None]:
    """Apply policy p: RF below p supports; VFT at p supports or more."""
    if k < p:
        return predict_constrained_rf(train, test, seed), "RF_constrained_rate", None
    t0 = choose_vft_t0(train)
    return test[f"vft_{int(t0)}"].to_numpy(), "VFT", t0


def macro_mae_by_doi(predictions: pd.DataFrame) -> float:
    """Equal DOI then equal-K MAE, matching the manuscript primary metric."""
    doi_k = predictions.assign(abs_error=lambda x: abs(x.target - x.prediction)).groupby(
        ["doi", "K"], as_index=False
    ).abs_error.mean()
    return float(doi_k.groupby("doi").abs_error.mean().mean())
