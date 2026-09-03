"""LODO router-vs-VFT audit and a small-data Gaussian-process baseline."""
from __future__ import annotations
import argparse, json, platform, sys, time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.sparse_router import FEATURES, choose_vft_t0, load_calisol, make_query_sets, predict_constrained_rf
KS = (3, 4, 5); CANDIDATES = (4, 5, 6)

def gpr_predict(test, seed):
    """Fit an independent fixed-kernel GP to every curve prefix."""
    kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * Matern(length_scale=1.0, length_scale_bounds="fixed", nu=2.5) + WhiteKernel(noise_level=.01, noise_level_bounds="fixed")
    predictions = []
    for _, row in test.iterrows():
        support_t = np.fromstring(row.support_T, sep=";").reshape(-1, 1)
        support_y = np.fromstring(row.support_y, sep=";")
        scaler = StandardScaler().fit(support_t)
        model = GaussianProcessRegressor(kernel=kernel, alpha=1e-8, normalize_y=True, optimizer=None, random_state=seed)
        model.fit(scaler.transform(support_t), support_y)
        predictions.append(model.predict(scaler.transform(np.array([[row.query_T]])))[0])
    return np.asarray(predictions)

def inner_scores(query_sets, fit_dois, valid_dois, seed):
    cells = []
    for k, queries in query_sets.items():
        train = queries[queries.doi.isin(fit_dois)].reset_index(drop=True); valid = queries[queries.doi.isin(valid_dois)].reset_index(drop=True)
        rf = predict_constrained_rf(train, valid, seed); t0 = choose_vft_t0(train); vft = valid[f"vft_{int(t0)}"].to_numpy()
        cells.append(pd.DataFrame({"doi": valid.doi, "K": k, "rf": abs(valid.target.to_numpy()-rf), "vft": abs(valid.target.to_numpy()-vft)}))
    errors = pd.concat(cells, ignore_index=True).groupby(["doi", "K"], as_index=False).mean()
    return {p: float(errors.assign(error=np.where(errors.K < p, errors.rf, errors.vft)).groupby("doi").error.mean().mean()) for p in CANDIDATES}

def select_threshold(query_sets, outer_train, fold):
    aggregate = {p: [] for p in CANDIDATES}; splitter = GroupShuffleSplit(n_splits=3, test_size=.25, random_state=10900+fold)
    for inner, (fit_idx, valid_idx) in enumerate(splitter.split(outer_train, groups=outer_train)):
        scores = inner_scores(query_sets, outer_train[fit_idx], outer_train[valid_idx], 10900+fold+inner)
        for p, score in scores.items(): aggregate[p].append(score)
    means = {p: float(np.mean(v)) for p, v in aggregate.items()}; return min(means, key=means.get), means

def doi_macro(predictions):
    return predictions.groupby(["doi", "K", "model"], as_index=False).abs_error.mean().groupby(["doi", "model"], as_index=False).abs_error.mean()

def paired_statistics(doi_errors, rng):
    wide = doi_errors.pivot(index="doi", columns="model", values="abs_error"); rows=[]; pvalues=[]
    for comparator in ("VFT", "independent_per_curve_GPR"):
        diff=(wide.nested_budget_router-wide[comparator]).to_numpy(); draws=rng.choice(diff,size=(20000,len(diff)),replace=True).mean(axis=1); test=wilcoxon(diff,alternative="two-sided",method="auto"); pvalues.append(test.pvalue)
        rows.append({"comparison":f"nested_budget_router - {comparator}","n_doi":len(diff),"wins_router":int((diff<0).sum()),"mean_mae_difference":float(diff.mean()),"median_mae_difference":float(np.median(diff)),"ci95_low":float(np.quantile(draws,.025)),"ci95_high":float(np.quantile(draws,.975)),"wilcoxon_stat":float(test.statistic),"p_two_sided":float(test.pvalue)})
    order = np.argsort(pvalues)
    adjusted = np.empty(len(pvalues), dtype=float)
    for rank, idx in enumerate(order): adjusted[idx] = min(1.0, (len(pvalues) - rank) * pvalues[idx])
    for row, adjusted_p in zip(rows, adjusted): row["p_holm"] = float(adjusted_p)
    return pd.DataFrame(rows), wide

def plot_distribution(wide, output):
    diff=(wide.nested_budget_router-wide.VFT).sort_values(); fig, axes=plt.subplots(1,2,figsize=(10.2,3.8),gridspec_kw={"width_ratios":[1.05,1.3]})
    axes[0].boxplot(diff.to_numpy(),vert=True,widths=.38,patch_artist=True,boxprops={"facecolor":"#d9e8f5","edgecolor":"#1f4e79"},medianprops={"color":"#8b1e3f","linewidth":1.5}); axes[0].scatter(np.ones(len(diff))+np.linspace(-.09,.09,len(diff)),diff,color="#1f4e79",s=28); axes[0].axhline(0,color="#555",lw=1); axes[0].set_xticks([1],["Router - VFT"]); axes[0].set_ylabel("DOI-macro MAE difference in log10(sigma)"); axes[0].set_title("Paired DOI distribution")
    axes[1].scatter(range(len(diff)),diff,color=np.where(diff<0,"#1f4e79","#b34b5c"),s=36); axes[1].axhline(0,color="#555",lw=1); axes[1].set_xlabel("Held-out DOI, ordered by difference"); axes[1].set_ylabel("Router - VFT DOI-macro MAE"); axes[1].set_title("One point per held-out DOI")
    for ax in axes: ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(output,dpi=320,bbox_inches="tight"); plt.close(fig)

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--calisol-csv",required=True); parser.add_argument("--development-rows",required=True); parser.add_argument("--output",required=True); args=parser.parse_args(); out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    queries=make_query_sets(load_calisol(args.calisol_csv,args.development_rows),KS); sources=np.array(sorted(set.intersection(*(set(q.doi.unique()) for q in queries.values())))); rows=[]; selections=[]; timings=[]
    for fold, holdout in enumerate(sources):
        outer_train=sources[sources!=holdout]; selected_p, inner_mae=select_threshold(queries,outer_train,fold); selections.append({"held_out_doi":holdout,"outer_fold":fold,"selected_p":selected_p,**{f"inner_mae_p{p}":v for p,v in inner_mae.items()}})
        for k, query_set in queries.items():
            train=query_set[query_set.doi.isin(outer_train)].reset_index(drop=True); test=query_set[query_set.doi.eq(holdout)].reset_index(drop=True)
            start=time.perf_counter(); rf=predict_constrained_rf(train,test,17); rf_time=time.perf_counter()-start; start=time.perf_counter(); t0=choose_vft_t0(train); vft=test[f"vft_{int(t0)}"].to_numpy(); vft_time=time.perf_counter()-start; start=time.perf_counter(); gpr=gpr_predict(test,17); gpr_time=time.perf_counter()-start; router=rf if k<selected_p else vft
            for model,prediction in {"nested_budget_router":router,"VFT":vft,"independent_per_curve_GPR":gpr}.items(): rows.append(pd.DataFrame({"doi":holdout,"K":k,"model":model,"target":test.target.to_numpy(),"prediction":prediction,"abs_error":abs(test.target.to_numpy()-prediction),"query_T":test.query_T,"delta_T":test.delta_T,"selected_p":selected_p,"vft_t0":t0}))
            timings.append({"held_out_doi":holdout,"K":k,"n_train":len(train),"n_test":len(test),"rf_fit_and_predict_s":rf_time,"vft_select_and_predict_s":vft_time,"gpr_fit_and_predict_s":gpr_time})
    predictions=pd.concat(rows,ignore_index=True); predictions.to_csv(out/"query_predictions.csv",index=False); pd.DataFrame(selections).to_csv(out/"router_selection.csv",index=False); pd.DataFrame(timings).to_csv(out/"timing_by_outer_fold.csv",index=False); doi_errors=doi_macro(predictions); doi_errors.to_csv(out/"doi_macro_errors.csv",index=False); paired,wide=paired_statistics(doi_errors,np.random.default_rng(20260902)); paired.to_csv(out/"paired_doi_inference.csv",index=False); plot_distribution(wide,out/"router_vs_vft_doi_distribution.png"); doi_errors.groupby("model",as_index=False).abs_error.mean().rename(columns={"abs_error":"macro_mae"}).to_csv(out/"model_summary.csv",index=False)
    (out/"protocol.json").write_text(json.dumps({"purpose":"Router-vs-VFT DOI-paired audit plus GPR.","outer_protocol":"Leave one DOI out over 19 common K=3,4,5 sources.","router_selection":"Three DOI-grouped inner splits; p in {4,5,6}.","inference_unit":"DOI-macro MAE equally over K.","primary_pair":"Router minus VFT; 20,000 DOI bootstrap resamples and Wilcoxon.","gpr":"Independent per-curve StandardScaler plus fixed Matern-5/2 GP on observed supports; no target, held-out DOI, or cross-curve data informs the kernel.","python":platform.python_version()},indent=2),encoding="utf-8"); print(pd.read_csv(out/"model_summary.csv").round(5).to_string(index=False)); print(paired.round(6).to_string(index=False))
if __name__=="__main__": main()
