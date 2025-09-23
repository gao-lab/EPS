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
# # Bio-marker analysis

# %%
import pickle
from pathlib import Path

import pandas as pd
import scanpy as sc
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import patchworklib as pw


# %%
res_dict = {}
eps_dict = {}
for file in Path('./gene_marker/').glob('*pkl'):
    with open(file, 'rb') as f:
        tmp_res = pickle.load(f)
    tmp_dict = dict()
    res_dict[file.stem] = tmp_res['var']['pearson_corr']
    eps_dict[file.stem] = tmp_res['var']['EPS']

res_df = pd.DataFrame(res_dict)
eps_df = pd.DataFrame(eps_dict)

eps_mean_df = eps_df.mean(0)


# %%
eps_df

# %%
res_df

# %% [markdown]
# ## Bio-marker

# %%
# res_df_filter = res_df # not filter out any data
res_df_filter = res_df.loc[:, (eps_mean_df < 0.9).values]
eps_df_filter = eps_df.loc[:, (eps_mean_df < 0.9).values]

# %%
slice_meta = pd.read_csv('./results/slide_metadata.csv', index_col=0)
# # rm rows whose disease_state is 'Healthy'
slice_meta = slice_meta[slice_meta['disease_state'] != 'Healthy']
slice_meta = slice_meta[slice_meta.index.isin(res_df_filter.columns)]
slice_meta['organ'].value_counts()


# %%
bio_marker_df = pd.read_csv('./results/cancer_biomarker_targets_with_category.csv', index_col=0)
bio_marker_df.head(2)

# %%
bio_marker_df['Organ'].value_counts()


# %%
biomarker_gene_scores = {}
all_biomarker_genes = []
for i, row in bio_marker_df.iterrows():
    genes = row['Gene']
    genes = genes.split('; ')
    all_biomarker_genes = all_biomarker_genes + genes

bio_res_df = res_df_filter.loc[res_df_filter.index.intersection(all_biomarker_genes)]
bio_eps_df = eps_df_filter.loc[eps_df_filter.index.intersection(all_biomarker_genes)]

bio_res_df_l = []
eps_res_df_l = []
for o in bio_marker_df['Organ'].unique():
    valid_slices = slice_meta.index[slice_meta['organ'] == o].tolist()
    tmp_bio_res_df = bio_res_df[valid_slices]
    tmp_bio_res_df = tmp_bio_res_df.mean(1)
    bio_res_df_l.append(tmp_bio_res_df)

    tmp_eps_res_df = bio_eps_df[valid_slices]
    tmp_eps_res_df = tmp_eps_res_df.mean(1)
    eps_res_df_l.append(tmp_eps_res_df)

bio_res_df_plot = pd.concat(bio_res_df_l, axis=1, keys=bio_marker_df['Organ'].unique())
eps_res_df_plot = pd.concat(eps_res_df_l, axis=1, keys=bio_marker_df['Organ'].unique())
# clip 0
bio_res_df_plot = bio_res_df_plot.clip(lower=0.01)
plt.figure(figsize=(8,8))
sns.heatmap(bio_res_df_plot, annot=True, cmap='Reds')
plt.show()

plt.figure(figsize=(8,8))
sns.heatmap(eps_res_df_plot, annot=True, cmap='Reds_r')
plt.show()

# %%
bio_res_df

# %%
cmap = plt.cm.Reds

plot_l = []
for o in bio_marker_df['Organ'].unique():
    # slice level
    valid_slices = slice_meta.index[slice_meta['organ'] == o].tolist()
    tmp_bio_res_df = res_df[valid_slices]
    tmp_eps_res_df = eps_df[valid_slices]
    
    # gene level
    valid_genes = bio_marker_df[bio_marker_df['Organ'] == o]['Gene'].str.split('; ')
    tmp_bio_res_df = tmp_bio_res_df.loc[tmp_bio_res_df.index.intersection(valid_genes.explode())]
    tmp_eps_res_df = tmp_eps_res_df.loc[tmp_eps_res_df.index.intersection(valid_genes.explode())]
    n_slice = len(valid_slices)
    if n_slice >10:
        tmp_bio_res_df = tmp_bio_res_df.mean(1)
        tmp_bio_res_df = tmp_bio_res_df.clip(lower=0.01)
        tmp_eps_res_df = tmp_eps_res_df.mean(1)

        tmp_df = pd.DataFrame({'bio': tmp_bio_res_df, 'eps': tmp_eps_res_df})
        tmp_df = tmp_df.sort_values(by='bio', ascending=False)

        plt.figure(figsize=(1.2,len(tmp_df)/3 +0.5))
        sns.heatmap(tmp_df['bio'].to_frame().rename(columns={'bio': o}), annot=True, cmap='Reds', vmin=0, vmax=0.6)
        plt.show()
        plt.figure(figsize=(1.2,len(tmp_df)/3 +0.5))
        sns.heatmap(tmp_df['eps'].to_frame().rename(columns={'eps': o}), annot=True, cmap='Reds_r', vmin=0.3, vmax=1)
        plt.show()

# %%
plot_l[1]

# %%
tmp_bio_res_df
