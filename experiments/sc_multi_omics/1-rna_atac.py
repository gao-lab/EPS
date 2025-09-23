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
# # Calculate EPS in 10x Multiome dataset (RNA and ATAC)

# %%
import numpy as np
import pandas as pd
import scanpy as sc
import ray
import scanpy as sc
import seaborn as sns
import patchworklib as pw
from scipy.sparse import issparse
from scipy.stats import linregress, pearsonr
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from expression_copilot import ExpressionCopilotModel


# %%
@ray.remote(num_cpus=2, num_gpus = 0.25, max_retries=1)
def ray_run_batch(gex, other, atac_emb_key, batch):
    return run_batch(gex, other, atac_emb_key, batch)

def run_batch(gex, other, atac_emb_key, batch):
    try:
        gex = gex.copy()
        other = other.copy()
        if atac_emb_key == 'ATAC_gene_activity_pca':
            other_emb = sc.pp.pca(other.obsm['ATAC_gene_activity'], n_comps=50)
        elif atac_emb_key == 'X_spectral':
            import snapatac2 as snap
            snap.tl.spectral(other, features=None)
            other_emb = other.obsm['X_spectral']
        else:
            other_emb = other.obsm[atac_emb_key].copy()

        gex.obsm['X_atac_emb'] = other_emb
        model = ExpressionCopilotModel(gex, image_key='X_atac_emb')

        eps = model.calc_metrics_per_gene()
        gene_metrics, _ = model.calc_baseline_metrics(method='mlp')
        model.save_results(f'./results/atac_neurips/{atac_emb_key}-{batch}.pkl')
        gene_df = eps.merge(gene_metrics, left_index=True, right_index=True)
        return gene_df

    except Exception as e:
        print(e)
        return None



# %% [markdown]
# # NeurIPS data

# %%
adata = sc.read_h5ad('./data/multiome/neurips/GSE194122_openproblems_neurips2021_multiome_BMMC_processed.h5ad')

# %%
len(adata.obs['batch'].unique())

# %% [markdown]
# We need to ensure predicting same gene subset in each batch

# %%
gex = adata[:, adata.var['feature_types'] == 'GEX'].copy()
other = adata[:, adata.var['feature_types'] != 'GEX'].copy()

# %%
gex.X = gex.layers['counts'].copy()
sc.pp.highly_variable_genes(
    gex, layer="counts", n_top_genes=2000, flavor="seurat_v3", batch_key='batch', subset=True
)

# %%
gex.X.data

# %%
tasks = []
ray.init(ignore_reinit_error=True)
for batch in adata.obs['batch'].unique():
    print(batch)
    gex_batch = gex[gex.obs['batch'] == batch].copy()
    other_batch = other[other.obs['batch'] == batch].copy()
    for atac_emb_key in ['ATAC_lsi_full', 'X_spectral', 'ATAC_gene_activity_pca']:
        tasks.append(ray_run_batch.remote(gex_batch, other_batch, atac_emb_key, batch))
res_l = ray.get(tasks)
ray.shutdown()

# %% [markdown]
# Load results

# %%
res_dict = {}
for i, batch in enumerate(adata.obs['batch'].unique()):
    res_dict[batch] = {}
    for j, atac_emb_key in enumerate(['ATAC_lsi_full', 'ATAC_lsi_red', 'X_spectral', 'ATAC_gene_activity_pca']):
        res_dict[batch][atac_emb_key] = res_l[i*4 + j]

# %%
res_dict['s1d1']['ATAC_gene_activity_pca']

# %%
plot_dict = {}
for i, (batch, metrics) in enumerate(res_dict.items()):
    tmp_ax = pw.Brick(figsize=(5,4))
    atac_emb_key = 'X_spectral'
    ax = sns.scatterplot(x=metrics[atac_emb_key]['EPS'], y=metrics[atac_emb_key]['pearson_corr'], s=5, ax=tmp_ax)
    ax.set_title(f'Batch: {batch}')
    ax.set_ylabel('Pearson correlation of each gene')
    ax.set_xlim(0, 1.4)
    plot_dict[i] = tmp_ax


# %%
# plot the figures: 4 sub figure per row
len(plot_dict)
black_ax1 = pw.Brick(figsize=(5,4))
black_ax2 = pw.Brick(figsize=(5,4))
black_ax3 = pw.Brick(figsize=(5,4))

(plot_dict[0] | plot_dict[1] | plot_dict[2] | plot_dict[3])/ \
    (plot_dict[4] | plot_dict[5] | plot_dict[6] | plot_dict[7])/ \
        (plot_dict[8] | plot_dict[9] | plot_dict[10] | plot_dict[11])/ \
            (plot_dict[12] | black_ax1 | black_ax2 | black_ax3)


# %%
for atac_emb_key in ['X_spectral']:
    key_res_dict = {batch: metrics[atac_emb_key] for batch, metrics in res_dict.items()}
    eps_list = []
    pearson_list = []
    for batch, metrics in key_res_dict.items():
        eps_list.append(metrics['EPS'].mean())
        pearson_list.append(metrics['pearson_corr'].mean())
    tmp_df = pd.DataFrame({'EPS': eps_list, 'pearson_corr': pearson_list, 'batch': list(key_res_dict.keys())})
    pearson, p_val = pearsonr(tmp_df["EPS"], tmp_df["pearson_corr"])
    ax = sns.regplot(tmp_df, x="EPS", y="pearson_corr", scatter_kws={"s": 10})
    ax.text(0.1, 0.97, f"Pearson = {pearson:.2f}, p = {p_val:.2e}", transform=ax.transAxes, fontsize=12,
            verticalalignment="top", )
    ax.set_xlabel('SPS')
    ax.set_ylabel('Pearson correlation per sample')
    plt.show()

# %%
key_res_dict
