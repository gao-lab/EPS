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

# %%
import pickle
from pathlib import Path

import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# %%
eps_dict = {}
for file in Path('./per_slice-scale:True-method:linear').glob('*pkl'):
    with open(file, 'rb') as f:
        tmp_res = pickle.load(f)
    tmp_dict = dict()
    eps_dict[file.stem] = tmp_res['var']['EPS'] 
eps_df = pd.DataFrame(eps_dict)
eps_mean_df = eps_df.mean(axis=1)

# %%
res_dict = {}
for file in Path('./per_slice-scale:True-method:mlp').glob('*pkl'):
    with open(file, 'rb') as f:
        tmp_res = pickle.load(f)
    tmp_dict = dict()
    res_dict[file.stem] = tmp_res['var']['pearson_corr']
res_df = pd.DataFrame(res_dict)
res_mean_df = res_df.mean(axis=1)

# %%
ax = sns.histplot(res_mean_df, alpha=0.5, bins=50, kde=True)
ax.set_xlabel('Pearson Correlation')

# %%
plt.figure(figsize=(6,4))
# sns.kdeplot(eps_mean_df, fill=True)
ax = sns.histplot(eps_mean_df, alpha=0.5, bins=50, kde=True)
ax.set_xlabel('EPS')
# add line x = 0.85
ax.axvline(x=0.83, color='purple', linestyle='--',)
# add the text
ax.text(0.81, 40, 'Top predicted genes', rotation=90, color='purple')

# add line x = 0.85
ax.axvline(x=0.985, color='red', linestyle='--',)
# add the text
ax.text(1.0, 40, 'Top unpredicted genes', rotation=90, color='red')


# %%
eps_mean_df.to_csv('./results/gene_eps_mean.csv')

# %%
