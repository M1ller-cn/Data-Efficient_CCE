"""Repeated nested DOI-grouped router selection.

This is a post-registration-style stability audit for the predeclared policy
space p in {4, 5, 6}. It never reads external robotic labels and does not alter
the deployed p=4 rule. Its purpose is to quantify how often each candidate is
selected under independent grouped inner partitions.
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
from src.sparse_router import (
    choose_vft_t0,
    load_calisol,
    make_query_sets,
    predict_constrained_rf,
)


def inner_scores(query_sets, fit_dois, valid_dois, candidates, seed):
    """Score every policy using one RF and one VFT prediction per K.

    The branches do not depend on the candidate threshold. Reusing them is
    mathematically identical to refitting them for each p, but avoids a
    three-fold waste of CPU on the small grouped-validation audit.
    """
    rows = []
    for k, queries in query_sets.items():
        train = queries[queries.doi.isin(fit_dois)].reset_index(drop=True)
        valid = queries[queries.doi.isin(valid_dois)].reset_index(drop=True)
        rf = predict_constrained_rf(train, valid, seed)
        t0 = choose_vft_t0(train)
        vft = valid[f"vft_{int(t0)}"].to_numpy()
        by_doi = pd.DataFrame({
            "doi": valid.doi.to_numpy(),
            "rf": abs(valid.target.to_numpy() - rf),
            "vft": abs(valid.target.to_numpy() - vft),
        }).groupby("doi", as_index=False).mean()
        for _, row in by_doi.iterrows():
            rows.append({"doi": row.doi, "K": k, "rf": row.rf, "vft": row.vft})
    errors = pd.DataFrame(rows)
    scores = {}
    for p in candidates:
        selected = np.where(errors.K.to_numpy() < p, errors.rf.to_numpy(), errors.vft.to_numpy())
        policy = errors[["doi", "K"]].assign(abs_error=selected)
        scores[p] = float(policy.groupby("doi").abs_error.mean().mean())
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calisol-csv", required=True)
    parser.add_argument("--development-rows", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n-seeds", type=int, default=50)
    parser.add_argument("--n-inner-splits", type=int, default=3)
    parser.add_argument("--base-seed", type=int, default=20260901)
    args = parser.parse_args()
    if args.n_seeds < 2 or args.n_inner_splits < 2:
        raise ValueError("Use at least two seeds and two grouped inner splits.")

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    data = load_calisol(args.calisol_csv, args.development_rows)
    queries = make_query_sets(data)
    sources = np.array(sorted(set.intersection(*(set(q.doi.unique()) for q in queries.values()))))
    candidates = (4, 5, 6)
    detail, selected = [], []

    for fold, holdout in enumerate(sources):
        outer_train = sources[sources != holdout]
        for repetition in range(args.n_seeds):
            candidate_scores = {p: [] for p in candidates}
            for inner in range(args.n_inner_splits):
                random_state = args.base_seed + fold * 100_000 + repetition * 100 + inner
                splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=random_state)
                fit_idx, valid_idx = next(splitter.split(outer_train, groups=outer_train))
                fit_dois, valid_dois = outer_train[fit_idx], outer_train[valid_idx]
                scores = inner_scores(queries, fit_dois, valid_dois, candidates, random_state)
                for p, score in scores.items():
                    candidate_scores[p].append(score)
                    detail.append({
                        "outer_fold": fold, "held_out_doi": holdout,
                        "repetition": repetition, "inner_split": inner,
                        "random_state": random_state, "p": p, "inner_macro_mae": score,
                        "n_fit_dois": len(fit_dois), "n_valid_dois": len(valid_dois),
                    })
            means = {p: float(np.mean(v)) for p, v in candidate_scores.items()}
            winner = min(means, key=means.get)
            selected.append({
                "outer_fold": fold, "held_out_doi": holdout, "repetition": repetition,
                "selected_p": winner, **{f"mae_p{p}": value for p, value in means.items()},
            })

    detail_df, selected_df = pd.DataFrame(detail), pd.DataFrame(selected)
    detail_df.to_csv(out / "inner_split_scores.csv", index=False)
    selected_df.to_csv(out / "selection_by_outer_fold_and_seed.csv", index=False)
    frequency = selected_df.selected_p.value_counts().reindex(candidates, fill_value=0).rename_axis("p").reset_index(name="selections")
    frequency["selection_rate"] = frequency.selections / len(selected_df)
    frequency.to_csv(out / "selection_frequency.csv", index=False)
    summary = selected_df[[f"mae_p{p}" for p in candidates]].agg(["mean", "std", "min", "max"]).T.reset_index(names="candidate")
    summary.to_csv(out / "candidate_score_summary.csv", index=False)
    (out / "protocol.json").write_text(json.dumps({
        "purpose": "Repeated grouped-inner-split stability audit; no external labels used.",
        "outer_protocol": "Leave one DOI out over the common K=3,4,5 source set.",
        "candidates": list(candidates), "n_seeds": args.n_seeds,
        "n_inner_splits": args.n_inner_splits, "base_seed": args.base_seed,
        "metric": "Equal DOI then equal K macro MAE in log10 conductivity.",
    }, indent=2))
    print(f"common_sources={len(sources)} rows={len(selected_df)}")
    print(frequency.to_string(index=False))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
