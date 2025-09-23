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
# # Calculate EPS in 10x slices

# %%
import pickle
from pathlib import Path

import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

import patchworklib as pw


# %%
def plot_single_slice(name:str, head = False, metric='pearson_corr', s=5, palette='magma'):
    # get the EPS
    slice_file = f'../hest_dataset/per_slice-scale:True-method:linear/{name}.pkl'
    with open(slice_file, 'rb') as f:
        slice_dict = pickle.load(f)
        gene_df = slice_dict['var']
    
    plot_dict = {}
    for method in ['mlp', 'ensemble', 'ridge', 'linear', 'stnet']:
        if method == 'stnet':
            tmp_gene_df = pd.read_csv(f'../pub_methods/results/stnet/{name}.csv', index_col=0)
            tmp_gene_df = tmp_gene_df.merge(gene_df[['EPS']], left_index=True, right_index=True)
            x = tmp_gene_df['EPS'].values

        else:
            slice_file = f'../hest_dataset/per_slice-scale:True-method:{method}/{name}.pkl'
            with open(slice_file, 'rb') as f:
                slice_dict = pickle.load(f)
                tmp_gene_df = slice_dict['var']
            x = gene_df['EPS'].values

        y = tmp_gene_df[metric].values
        filter_idx = np.isnan(y)
        x = x[~filter_idx]
        y = y[~filter_idx]
        xy = np.vstack([x, y])            
        z = gaussian_kde(xy)(xy)

        tmp_ax = pw.Brick(figsize=(5,4))
        ax = sns.scatterplot(x =xy[0], y=xy[1], hue=z, s=s, ax=tmp_ax, palette=palette, legend=False)
        metric_name = metric.replace('_', ' ')
        if metric == 'rmse':
            metric_name = metric_name.upper()
            ax.set_ylim((np.percentile(y, 3), np.percentile(y, 97)))
            ax.set_ylim((np.percentile(y, 3), np.percentile(y, 97)))
        else:
            metric_name = metric_name.capitalize()
            ax.set_ylim((-0.1, 1.0))
            ax.set_xlim((0, 1.1))

        ax.set_ylabel(f'{metric_name} of each gene')
        ax.set_xlabel('EPS')

        if head:
            if method in ['mlp']:
                ax.set_title(method.upper())
            elif method == 'stnet':
                ax.set_title('ST-NET')
            else:
                ax.set_title(method.capitalize())
        # plt.show()
        plot_dict[method] = ax
    return plot_dict



# %%
slices_from_10x = ['TENX70', 'TENX72', 'TENX51', 'TENX39', 'TENX40', 'TENX50', 'TENX31']

# %%
slide_meta = pd.read_csv('../hest_dataset/results/slide_metadata.csv', index_col=0)
slide_meta.head()

# subset slices_from_10x
sub_slide_meta = slide_meta.loc[slices_from_10x]
sub_slide_meta

# %%
row_l = []
for slice_ in slices_from_10x:
    plot_dict = plot_single_slice(slice_)
    row = plot_dict['ensemble'] | plot_dict['mlp'] | plot_dict['ridge'] | plot_dict['linear'] | plot_dict['stnet']
    row_l.append(row)

row_all = row_l[0] / row_l[1] / row_l[2] / row_l[3] / row_l[4] / row_l[5] / row_l[6]
row_all

# %%
row_l = []
for slice_ in slices_from_10x:
    plot_dict = plot_single_slice(slice_, metric='spearman_corr')
    row = plot_dict['ensemble'] | plot_dict['mlp'] | plot_dict['ridge'] | plot_dict['linear'] | plot_dict['stnet']
    row_l.append(row)

row_all = row_l[0] / row_l[1] / row_l[2] / row_l[3] / row_l[4] / row_l[5] / row_l[6]
row_all

# %%
row_l = []
for slice_ in slices_from_10x:
    plot_dict = plot_single_slice(slice_, metric='rmse')
    row = plot_dict['ensemble'] | plot_dict['mlp'] | plot_dict['ridge'] | plot_dict['linear'] | plot_dict['stnet']
    row_l.append(row)

row_all = row_l[0] / row_l[1] / row_l[2] / row_l[3] / row_l[4] / row_l[5] / row_l[6]
row_all
