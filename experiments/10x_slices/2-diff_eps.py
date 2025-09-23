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
# # Comparing the EPS across models

# %%
import pickle
from pathlib import Path
from functools import reduce


import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress, pearsonr, gaussian_kde, ttest_rel, wilcoxon

import patchworklib as pw


# %%
def get_bins(name:str, head = False, metric='pearson_corr', palette='magma', bins=10):
    # get the EPS
    slice_file = f'../hest_dataset/per_slice-scale:True-method:linear/{name}.pkl'
    with open(slice_file, 'rb') as f:
        slice_dict = pickle.load(f)
        gene_df = slice_dict['var']
    gene_df = gene_df[['EPS', 'n_cells']]
    
    df_list = []
    for method in ['mlp', 'ensemble', 'ridge', 'linear', 'stnet']:
        if method == 'stnet':
            tmp_gene_df = pd.read_csv(f'../pub_methods/results/stnet/{name}.csv', index_col=0)
        else:
            slice_file = f'../hest_dataset/per_slice-scale:True-method:{method}/{name}.pkl'
            with open(slice_file, 'rb') as f:
                slice_dict = pickle.load(f)
                tmp_gene_df = slice_dict['var']
        tmp_gene_series = tmp_gene_df[metric]
        tmp_gene_series.name = f'{metric}:{method}'
        gene_df = gene_df.merge(tmp_gene_series, left_index=True, right_index=True, how='left')

    return gene_df


# %%
slices_from_10x = ['TENX70', 'TENX72', 'TENX51', 'TENX39', 'TENX40', 'TENX50', 'TENX31']
tissue_of_slices = ['Bowel', 'Lung', 'Ovary', 'Breast', 'Prostate', 'Uterus', 'Brain']

method_plot_dict = {}
for method in ['mlp', 'ensemble','stnet']:
    plot_dict = {}
    for i, slice_ in enumerate(slices_from_10x):
        plot_df = get_bins(slice_, bins=100)
        tmp_ax = pw.Brick(figsize=(5,4))

        plot_df['detla'] = plot_df[f'pearson_corr:{method}'] - plot_df['pearson_corr:linear']
        
        xy = np.vstack([plot_df['EPS'], plot_df['detla']])
        # remove nan
        xy = xy[:, ~np.isnan(xy).any(axis=0)]
        z = gaussian_kde(xy)(xy)
        
        ax = sns.scatterplot(x=xy[0], y=xy[1], s=8, hue=z, alpha=1, palette='magma', ax=tmp_ax, legend=False)
        ax.set_title(tissue_of_slices[i])
        ax.set_xlabel("EPS")
        if method == 'mlp':
            name = "MLP"
        elif method == 'ensemble':
            name = "Ensemble"
        else:
            name = "ST-Net"

        ax.set_ylabel(f"Delta Pearson ({name} - Linear)")
        ax.set_xlim(0, 1.1)
        # add line y=0
        ax.axhline(0, color='grey', linestyle='--')
        # plt.show()
        plot_dict[i] = tmp_ax

    empty_ax = pw.Brick(figsize=(5,4))
    empty_ax.set_title('')
    empty_ax.set_xlim(0, 1.1)
    empty_ax.set_xlabel("EPS")
    empty_ax.set_ylabel(f"Delta Pearson ({name} - Linear)")
    # empty_ax.axis('off')
    method_plot = (plot_dict[0] | plot_dict[1] | plot_dict[2] | plot_dict[3]) / ( plot_dict[4] | plot_dict[5] | plot_dict[6] |empty_ax)
    method_plot_dict[method] = method_plot

# %%
method_plot_dict['mlp']

# %%
method_plot_dict['ensemble']

# %%
method_plot_dict['stnet']

# %% [markdown]
# ## Box plot

# %%
slide_meta = pd.read_csv('../hest_dataset/results/slide_metadata.csv', index_col=0)
# slices_from_10x = ['TENX70', 'TENX72', 'TENX51', 'TENX39', 'TENX40', 'TENX50', 'TENX31']
# subset slices_from_10x
sub_slide_meta = slide_meta.loc[slices_from_10x]
sub_slide_meta

# %%
# wide to long data
sub_slide_meta_long = sub_slide_meta.melt(
    value_vars=['mlp:gene_pearson', "ensemble:gene_pearson", "st-net:gene_pearson", "ridge:gene_pearson", "linear:gene_pearson",],
    var_name='method',
    value_name='score')
sub_slide_meta_long['method'] = sub_slide_meta_long['method'].str.split(':').str[0]

# replace names
sub_slide_meta_long['method'] = sub_slide_meta_long['method'].replace({'mlp': 'MLP', 'ensemble': 'Ensemble', 'st-net': 'ST-Net', 'ridge': 'Ridge', 'linear': 'Linear'})

# %%
plt.figure(figsize=(5,4))
ax = sns.boxplot(sub_slide_meta_long, x='method', y='score', palette='Set1')
sns.stripplot(x='method', y='score', data=sub_slide_meta_long, color='black', jitter=True, ax=ax, alpha=0.5)
ax.set_xlabel('')
ax.set_ylabel('Slice-level Pearson Correlation')
ax.set
