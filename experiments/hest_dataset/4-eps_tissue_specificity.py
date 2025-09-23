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
# # Find tissue specificity of gene EPS

# %%
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc


# %%
res_dict = {}
for file in Path('./per_slice-scale:True-method:mlp').glob('*pkl'):
    with open(file, 'rb') as f:
        tmp_res = pickle.load(f)
    tmp_dict = dict()
    res_dict[file.stem] = tmp_res['var']['pearson_corr']
res_df = pd.DataFrame(res_dict)

# %%
eps_dict = {}
for file in Path('./per_slice-scale:True-method:linear').glob('*pkl'):
    with open(file, 'rb') as f:
        tmp_res = pickle.load(f)
    tmp_dict = dict()
    eps_dict[file.stem] = tmp_res['var']['EPS'] 
eps_df = pd.DataFrame(eps_dict)

# %%
# re-arrange eps_df columns to match res_df
eps_df = eps_df[res_df.columns]

# %%
slice_meta = pd.read_csv('./results/slide_metadata.csv', index_col=0)

# %%
slice_meta['tissue'].value_counts()

# %%
slice_meta.columns

# %%
# combine vars organ and disease_state
slice_meta.groupby(['organ', 'disease_state']).size()

# %%
res_df.fillna(0, inplace=True)
eps_df.fillna(1, inplace=True)

# %%
adata = sc.AnnData(
    X = res_df.T,
    obs = slice_meta.loc[res_df.columns],
    layers = {'eps':eps_df.T}
)

# %% [markdown]
# ## By PCC

# %%
groups = adata.obs['organ'].value_counts()
# only with more than 10 samples
groups = groups[groups>10].index.tolist()
sc.tl.rank_genes_groups(adata, groupby='organ', groups=groups, method='wilcoxon', use_raw=False)
# sc.pl.rank_genes_groups(adata, n_genes=20, sharey=False)

# filter the subadata only contains group
sub_adata = adata[adata.obs['organ'].isin(groups)].copy()
# set the order
sub_adata.obs['organ'] = pd.Categorical(sub_adata.obs['organ'], categories=groups, ordered=True)

sc.pl.rank_genes_groups_heatmap(sub_adata, dendrogram=False, groups = groups, n_genes=10, groupby='organ', standard_scale='False', show_gene_labels=True, cmap='Reds', vmin=0)

# %%
df_eps = sc.get.rank_genes_groups_df(adata, group=groups, log2fc_min=0.1, pval_cutoff=0.01)
print(df_eps.shape)
df_eps.to_csv('./results/hest_organ_deg_by_eps.csv', index=False)

# %% [markdown]
# ## By EPS

# %%
groups = adata.obs['organ'].value_counts()
# only with more than 10 samples
groups = groups[groups>10].index.tolist()
sc.tl.rank_genes_groups(adata, groupby='organ', groups=groups, method='wilcoxon', use_raw=False, layer='eps')
# sc.pl.rank_genes_groups(adata, n_genes=20, sharey=False)

# filter the subadata only contains group
sub_adata = adata[adata.obs['organ'].isin(groups)].copy()
# set the order
sub_adata.obs['organ'] = pd.Categorical(sub_adata.obs['organ'], categories=groups, ordered=True)

sc.pl.rank_genes_groups_heatmap(sub_adata, dendrogram=False, groups = groups, n_genes=-10, groupby='organ', standard_scale=False, show_gene_labels=True, layer='eps', cmap='Reds_r', vmax=1.1)

# %%
sc.pl.rank_genes_groups_heatmap(sub_adata, dendrogram=False, groups = groups, n_genes=-10, groupby='organ', standard_scale=False, show_gene_labels=True, cmap='Reds', vmin=0)

# %%
df_eps = sc.get.rank_genes_groups_df(adata, group=groups, log2fc_min=0.1, pval_cutoff=0.05)
print(df_eps.shape)
df_eps.to_csv('./results/hest_organ_deg_by_eps.csv', index=False)

# %%
