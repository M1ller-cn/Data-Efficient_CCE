"""Audit a pre-specified fixed-target support-budget allocation policy.

The experiment deliberately differs from ordinary next-temperature prediction.
For every eligible curve, K=3, 4, and 5 use nested prefixes from the same
terminal six-point segment and predict the same final temperature.  Router
thresholds and support budgets are selected only in outer-training DOI pools.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.sparse_router import (
    choose_vft_t0,
    load_calisol,
    make_fixed_terminal_query_sets,
    predict_constrained_rf,
)


KS = (3, 4, 5)
CANDIDATES = (4, 5, 6)
COSTS = (0.0, 0.0025, 0.005, 0.01, 0.02)


def inner_error_tables(query_sets, fit_dois, valid_dois, seed):
    """Return DOI-macro validation errors for each router threshold and K."""
    cells = []
    for k, queries in query_sets.items():
        train = queries[queries.doi.isin(fit_dois)].reset_index(drop=True)
        valid = queries[queries.doi.isin(valid_dois)].reset_index(drop=True)
        rf = predict_constrained_rf(train, valid, seed)
        t0 = choose_vft_t0(train)
        vft = valid[f"vft_{int(t0)}"].to_numpy()
        for p in CANDIDATES:
            prediction = rf if k < p else vft
            cells.append(
                pd.DataFrame(
                    {
                        "doi": valid.doi,
                        "K": k,
                        "p": p,
                        "abs_error": abs(valid.target.to_numpy() - prediction),
                    }
                )
            )
    return (
        pd.concat(cells, ignore_index=True)
        .groupby(["p", "doi", "K"], as_index=False)
        .abs_error.mean()
    )


def select_policy(query_sets, outer_train, fold):
    """Tune p, then cost-sensitive K, using only grouped inner DOI splits."""
    splits = GroupShuffleSplit(
        n_splits=3, test_size=0.25, random_state=10900 + fold
    )
    tables = []
    for inner, (fit_idx, valid_idx) in enumerate(splits.split(outer_train, groups=outer_train)):
        tables.append(
            inner_error_tables(
                query_sets,
                outer_train[fit_idx],
                outer_train[valid_idx],
                10900 + fold + inner,
            )
        )
    errors = pd.concat(tables, ignore_index=True)
    mean_by_p_k = errors.groupby(["p", "K"], as_index=False).abs_error.mean()
    mean_by_p = mean_by_p_k.groupby("p", as_index=False).abs_error.mean()
    selected_p = int(mean_by_p.loc[mean_by_p.abs_error.idxmin(), "p"])
    selected_errors = mean_by_p_k[mean_by_p_k.p.eq(selected_p)].set_index("K").abs_error
    budgets = []
    for cost in COSTS:
        utilities = {k: float(selected_errors.loc[k] + cost * (k - min(KS))) for k in KS}
        selected_k = min(utilities, key=utilities.get)
        budgets.append(
            {
                "measurement_cost": cost,
                "selected_K": selected_k,
                "inner_objective": utilities[selected_k],
                **{f"inner_mae_K{k}": float(selected_errors.loc[k]) for k in KS},
            }
        )
    return selected_p, mean_by_p.set_index("p").abs_error.to_dict(), budgets


def plot_frontier(frontier, output):
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.55), gridspec_kw={"width_ratios": [1.2, 1]})
    axes[0].plot(frontier.measurement_cost, frontier.macro_mae, marker="o", color="#1f4e79", lw=1.6)
    axes[0].set_xlabel("Additional-support cost (log10 conductivity MAE units)")
    axes[0].set_ylabel("Held-out DOI-macro MAE")
    axes[0].set_title("Cost-sensitive policy frontier")
    axes[1].step(frontier.measurement_cost, frontier.mean_selected_K, where="mid", color="#8b1e3f", lw=1.8)
    axes[1].scatter(frontier.measurement_cost, frontier.mean_selected_K, color="#8b1e3f", s=30)
    axes[1].set_xlabel("Additional-support cost")
    axes[1].set_ylabel("Mean selected support budget")
    axes[1].set_yticks(KS)
    axes[1].set_title("Training-selected measurement budget")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output, dpi=320, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calisol-csv", required=True)
    parser.add_argument("--development-rows", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    queries = make_fixed_terminal_query_sets(
        load_calisol(args.calisol_csv, args.development_rows), KS
    )
    groups_by_k = {k: set(q.group) for k, q in queries.items()}
    if len({frozenset(groups) for groups in groups_by_k.values()}) != 1:
        raise RuntimeError("Fixed-terminal groups must be identical across K")
    target_by_k = {
        k: q.set_index("group")[["query_T", "target"]] for k, q in queries.items()
    }
    reference = target_by_k[KS[0]]
    if any(not reference.equals(target_by_k[k]) for k in KS[1:]):
        raise RuntimeError("Fixed-terminal target differs across K")
    sources = np.array(sorted(set.intersection(*(set(q.doi.unique()) for q in queries.values()))))

    predictions, selections, frontier_rows = [], [], []
    for fold, holdout in enumerate(sources):
        outer_train = sources[sources != holdout]
        selected_p, inner_p_mae, budgets = select_policy(queries, outer_train, fold)
        selections.append(
            {
                "held_out_doi": holdout,
                "outer_fold": fold,
                "selected_p": selected_p,
                **{f"inner_mae_p{p}": float(inner_p_mae[p]) for p in CANDIDATES},
            }
        )
        test_cache = {}
        for k, query_set in queries.items():
            train = query_set[query_set.doi.isin(outer_train)].reset_index(drop=True)
            test = query_set[query_set.doi.eq(holdout)].reset_index(drop=True)
            rf = predict_constrained_rf(train, test, seed=17)
            t0 = choose_vft_t0(train)
            vft = test[f"vft_{int(t0)}"].to_numpy()
            router = rf if k < selected_p else vft
            test_cache[k] = (test, router, t0)
            predictions.append(
                pd.DataFrame(
                    {
                        "doi": holdout,
                        "group": test.group,
                        "K": k,
                        "model": "fixed_target_router",
                        "target": test.target.to_numpy(),
                        "prediction": router,
                        "abs_error": abs(test.target.to_numpy() - router),
                        "query_T": test.query_T,
                        "selected_p": selected_p,
                        "vft_t0": t0,
                    }
                )
            )
        for budget in budgets:
            k = budget["selected_K"]
            test, router, t0 = test_cache[k]
            predictions.append(
                pd.DataFrame(
                    {
                        "doi": holdout,
                        "group": test.group,
                        "K": k,
                        "model": "cost_sensitive_budget_policy",
                        "measurement_cost": budget["measurement_cost"],
                        "target": test.target.to_numpy(),
                        "prediction": router,
                        "abs_error": abs(test.target.to_numpy() - router),
                        "query_T": test.query_T,
                        "selected_p": selected_p,
                        "vft_t0": t0,
                    }
                )
            )
            frontier_rows.append(
                {
                    "held_out_doi": holdout,
                    "outer_fold": fold,
                    "selected_p": selected_p,
                    **budget,
                }
            )

    all_predictions = pd.concat(predictions, ignore_index=True)
    all_predictions.to_csv(out / "fixed_target_predictions.csv", index=False)
    pd.DataFrame(selections).to_csv(out / "fixed_target_router_selection.csv", index=False)
    selection_frame = pd.DataFrame(frontier_rows)
    selection_frame.to_csv(out / "measurement_cost_selection.csv", index=False)
    doi_errors = (
        all_predictions.groupby(["doi", "model", "K", "measurement_cost"], dropna=False, as_index=False)
        .abs_error.mean()
    )
    doi_errors.to_csv(out / "fixed_target_doi_mae.csv", index=False)
    policy_errors = doi_errors[doi_errors.model.eq("cost_sensitive_budget_policy")]
    frontier = (
        policy_errors.groupby("measurement_cost", as_index=False).abs_error.mean()
        .rename(columns={"abs_error": "macro_mae"})
        .merge(
            selection_frame.groupby("measurement_cost", as_index=False).selected_K.mean().rename(columns={"selected_K": "mean_selected_K"}),
            on="measurement_cost",
        )
    )
    frontier.to_csv(out / "measurement_cost_frontier.csv", index=False)
    plot_frontier(frontier, out / "fixed_target_cost_frontier.png")
    protocol = {
        "purpose": "Fixed-terminal support-budget allocation sensitivity analysis.",
        "source": "CALiSol-23 development split with frozen row index.",
        "query_construction": "One terminal segment per formulation with at least six distinct temperatures. K=3,4,5 use nested prefixes and the same sixth/final target.",
        "outer_protocol": f"Leave one DOI out over {len(sources)} common DOI sources.",
        "router_selection": "Three DOI-grouped inner splits in each outer-training pool; p in {4,5,6} selected by equally weighted K=3,4,5 DOI-macro MAE.",
        "budget_selection": "For each pre-specified cost c in {0, 0.0025, 0.005, 0.01, 0.02}, select K in {3,4,5} by inner MAE(K)+c*(K-3). No held-out DOI informs p or K.",
        "inference_unit": "One terminal query per eligible formulation, DOI-macro MAE for evaluation.",
        "platform": platform.platform(),
        "python": platform.python_version(),
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print("Eligible groups:", len(groups_by_k[KS[0]]), "Common DOI sources:", len(sources))
    print("Fixed-router MAE by K")
    print(doi_errors[doi_errors.model.eq("fixed_target_router")].groupby("K").abs_error.mean().round(6).to_string())
    print("Cost frontier")
    print(frontier.round(6).to_string(index=False))


if __name__ == "__main__":
    main()
