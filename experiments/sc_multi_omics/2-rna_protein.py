# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: conda
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Calculate EPS in CITE-seq dataset (RNA and surface protein)

# %%
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import patchworklib as pw
import ray
from scipy.stats import linregress, pearsonr
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from expression_copilot import ExpressionCopilotModel


# %%
def adt_preprocessing(adata: sc.AnnData, n_comps:int=20) -> None:
    adata.var["highly_variable"] = True
    adata.layers['log1p'] = adata.X.copy()
    sc.pp.scale(adata, max_value=10)
    sc.pp.pca(adata, n_comps=n_comps)

@ray.remote(num_cpus=4, num_gpus = 0.5, max_retries=1)
def ray_run_batch(gex, other, batch):
    return run_batch(gex, other, batch)

def run_batch(gex, other, batch):
    try:
        gex = gex.copy()
        other = other.copy()

        # print('Preprocessing ADT...')
        adt_preprocessing(other)
        gex.obsm['X_adt_emb'] = other.obsm['X_pca']

        gex.X = gex.layers['counts'].copy()
        model = ExpressionCopilotModel(gex, image_key='X_adt_emb')

        eps = model.calc_metrics_per_gene()

        gene_metrics, _ = model.calc_baseline_metrics(method='mlp')
        model.save_results(f'./results/adt_neurips/{batch}.pkl')
        gene_df = eps.merge(gene_metrics, left_index=True, right_index=True)
        return gene_df
    except Exception as e:
        print(e)
        return None


# %% [markdown]
# # NeurIPS data

# %%
adata = sc.read_h5ad('./data/cite/neurips/GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad')

# %%
len(adata.obs['batch'].unique())

# %%
gex = adata[:, adata.var['feature_types'] == 'GEX'].copy()
other = adata[:, adata.var['feature_types'] != 'GEX'].copy()

gex.X = gex.layers['counts'].copy()
sc.pp.highly_variable_genes(
    gex, layer="counts", n_top_genes=2000, flavor="seurat_v3", batch_key='batch', subset=True
)

# %%
tasks = []
ray.init(ignore_reinit_error=True)
for batch in adata.obs['batch'].unique():
    print(batch)
    gex_batch = gex[gex.obs['batch'] == batch].copy()
    other_batch = other[other.obs['batch'] == batch].copy()
    tasks.append(ray_run_batch.remote(gex_batch, other_batch, batch))
res_l = ray.get(tasks)

res_dict = {}
for batch, res in zip(adata.obs['batch'].unique(), res_l):
    res_dict[batch] = res

ray.shutdown()

# %%
plot_dict = {}
for i, (batch, metrics) in enumerate(res_dict.items()):
    tmp_ax = pw.Brick(figsize=(5,4))
    ax = sns.scatterplot(x=metrics['EPS'], y=metrics['pearson_corr'], s=5, ax=tmp_ax)
    ax.set_title(f'Batch: {batch}')
    ax.set_ylabel('Pearson correlation of each gene')
    ax.set_xlim(0, 1.4)
    plot_dict[i] = tmp_ax


# %%
(plot_dict[0] | plot_dict[1] | plot_dict[2] | plot_dict[3])/ \
    (plot_dict[4] | plot_dict[5] | plot_dict[6] | plot_dict[7])/ \
        (plot_dict[8] | plot_dict[9] | plot_dict[10] | plot_dict[11])

# %%
eps_list, pearson_list = [], []
for batch, metrics in res_dict.items():
    eps_list.append(metrics['EPS'].mean())
    pearson_list.append(metrics['pearson_corr'].mean())
tmp_df = pd.DataFrame({'EPS': eps_list, 'pearson_corr': pearson_list, 'batch': list(res_dict.keys())})
pearson, p_val = pearsonr(tmp_df["EPS"], tmp_df["pearson_corr"])
ax = sns.regplot(tmp_df, x="EPS", y="pearson_corr", scatter_kws={"s": 10})
ax.text(0.1, 0.97, f"Pearson = {pearson:.2f}, p = {p_val:.2e}", transform=ax.transAxes, fontsize=12,
        verticalalignment="top", )
ax.set_xlabel('SPS')
ax.set_ylabel('Pearson correlation per sample')
plt.show()
