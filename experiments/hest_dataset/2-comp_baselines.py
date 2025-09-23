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
# # Compared results from different baselines

# %%
import pickle
from pathlib import Path
import numpy as np
import seaborn as sns
import pandas as pd
import patchworklib as pw
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel, wilcoxon
from scipy.stats import linregress, pearsonr, gaussian_kde
from matplotlib.ticker import FuncFormatter


# %%
prefix = "./per_slice-scale:True-method:"
methods = ["linear", "mlp", "ensemble", "ridge"]
shared_file_names = []
for method in methods:
    file_names = list(Path(prefix + method).glob("*.pkl"))
    file_names = [file_name.name for file_name in file_names]
    shared_file_names.append(set(file_names))

shared_file_names = set.intersection(*shared_file_names)
len(shared_file_names)

# %%
eps_dict = {}
for file in shared_file_names:
    res_dict = {}
    prefix = "./per_slice-scale:True-method:linear"
    file_name = Path(prefix) / file
    with open(file_name, "rb") as f:
        tmp_res = pickle.load(f)
    rank_median = np.median(tmp_res["image2omics_rank"])
    eps_gene_median = np.median(tmp_res['var']['EPS'])
    eps_gene_mean = np.mean(tmp_res['var']['EPS'])
    n_spot = tmp_res["n_spot"]
    eps_dict[file.split(".")[0]] = {"n_spot":n_spot, "rank_median":rank_median, "eps_gene_mean":eps_gene_mean, "eps_gene_median":eps_gene_median}
eps_df = pd.DataFrame(eps_dict).T
eps_df.head(2)

# %%
prefix = "./per_slice-scale:True-method:"
res_dict_all = {}
for file in shared_file_names:
    res_dict = {}
    for method in methods:
        file_name = Path(prefix + method) / file
        with open(file_name, "rb") as f:
            tmp_res = pickle.load(f)
        res_dict[f"{method}:gene_pearson"] = tmp_res["var"]["pearson_corr"].mean()
        res_dict[f"{method}:gene_spearman"] = tmp_res["var"]["spearman_corr"].mean()
    # print(file)
    # print(res_dict)
    res_dict_all[file.split(".")[0]] = res_dict
    # print("="*50)
res_df = pd.DataFrame(res_dict_all).T
res_df = res_df.merge(eps_df, how="left", left_index=True, right_index=True)
res_df.head(2)

# %%
stnet_metrics_dict = {}
for file in Path("../pub_methods/results/stnet/").rglob("*.csv"):
    name = file.stem.split("_")[0]
    df = pd.read_csv(file, index_col=0)
    pearson_corr = df.mean()['pearson_corr']
    spearman_corr = df.mean()['spearman_corr']
    stnet_metrics_dict[name] = {"st-net:gene_pearson": pearson_corr, 'st-net:gene_spearman':spearman_corr}
stnet_res = pd.DataFrame(stnet_metrics_dict).T
res_df = res_df.merge(stnet_res, how="left", left_index=True, right_index=True)

# %%
res_df.head()

# %% [markdown]
# ## SPS - Pearson correlation

# %%
res_df.columns

# %%
res_df["log_eps_gene_mean"] = np.log10(res_df["eps_gene_mean"])
plot_dict = {}
for method in ["linear", "mlp", "ensemble", "ridge", "st-net"]:
    # sns.scatterplot(res_df, x="log_eps_gene_mean", y=f"{method}:gene_pearson")
    tmp_ax = pw.Brick(figsize=(5,4))
    pearson, p_val = pearsonr(res_df["log_eps_gene_mean"], res_df[f"{method}:gene_pearson"])
    ax = sns.regplot(res_df, x="log_eps_gene_mean", y=f"{method}:gene_pearson", scatter_kws={"s": 10}, ax=tmp_ax)
    slope, intercept, r_value, p_value, std_err = linregress(res_df["log_eps_gene_mean"], res_df[f"{method}:gene_pearson"])
    ax.text(0.25, 0.95, f"Pearson = {pearson:.2f}, p = {p_val:.1e}", transform=ax.transAxes, fontsize=14,
            verticalalignment="top", )
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{10**(float(x)):.2f}"))
    ax.set_xlabel('Slice Predictability Score (SPS)')
    ax.set_ylabel('Pearson')
    ax.set_ylim(-0.01, 0.6)
    if method == "st-net":
        ax.title.set_text("ST-NET")
    elif method == "mlp":
        ax.title.set_text("MLP")
    else:
        ax.title.set_text(f"{method.capitalize()}")
    plot_dict[method] = tmp_ax


# %%
plot_dict['st-net']| plot_dict['ensemble'] | plot_dict['mlp']  | plot_dict['ridge'] | plot_dict['linear'] 

# %% [markdown]
# ## SPS - Spearman correlation

# %%
res_df.columns

# %%
res_df["log_eps_gene_mean"] = np.log10(res_df["eps_gene_mean"])
plot_dict = {}
for method in ["linear", "mlp", "ensemble", "ridge", "st-net"]:
    # sns.scatterplot(res_df, x="log_eps_gene_mean", y=f"{method}:gene_spearman")
    tmp_ax = pw.Brick(figsize=(5,4))
    pearson, p_val = pearsonr(res_df["log_eps_gene_mean"], res_df[f"{method}:gene_spearman"])
    ax = sns.regplot(res_df, x="log_eps_gene_mean", y=f"{method}:gene_spearman", scatter_kws={"s": 10}, ax=tmp_ax)
    slope, intercept, r_value, p_value, std_err = linregress(res_df["log_eps_gene_mean"], res_df[f"{method}:gene_spearman"])
    ax.text(0.25, 0.95, f"Pearson = {pearson:.2f}, p = {p_val:.1e}", transform=ax.transAxes, fontsize=14,
            verticalalignment="top", )
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{10**(float(x)):.2f}"))
    ax.set_xlabel('Slice Predictability Score (SPS)')
    ax.set_ylabel('Spearman')
    ax.set_ylim(-0.01, 0.6)
    if method == "st-net":
        ax.title.set_text("ST-NET")
    elif method == "mlp":
        ax.title.set_text("MLP")
    else:
        ax.title.set_text(f"{method.capitalize()}")
    plot_dict[method] = tmp_ax

# %%
plot_dict['st-net']| plot_dict['ensemble'] | plot_dict['mlp'] | plot_dict['ridge'] | plot_dict['linear'] 

# %% [markdown]
# ## Diff of baselines

# %%
res_df.head()

# %%
plot_df = res_df.copy()
plot_dict = {}
for method in ['mlp', 'ensemble','st-net']:
    plot_df['delta'] = plot_df[f'{method}:gene_pearson'] - plot_df['linear:gene_pearson']
    tmp_ax = pw.Brick(figsize=(5,4))
    xy = np.vstack([plot_df['eps_gene_mean'], plot_df['delta']])
    z = gaussian_kde(xy)(xy)
    ax = sns.scatterplot(x=xy[0], y=xy[1], s=8, alpha=1, ax=tmp_ax, legend=False)
    ax.set_xlabel('Slice Predictability Score (SPS)')
    if method == 'mlp':
        name = "MLP"
    elif method == 'ensemble':
        name = "Ensemble"
    else:
        name = "ST-NET"
    ax.set_ylabel(f'Delta Pearson ({name} - Linear)')
    ax.axhline(0, ls='--', color='grey')
    ax.set_ylim(-0.05, 0.25)
    ax.set_title(f"{name} V.S. Linear")
    plot_dict[method] = tmp_ax

# %%
plot_dict['st-net'] | plot_dict['ensemble'] | plot_dict['mlp']

# %% [markdown]
# ## Violin baselines

# %%
df_long = res_df.melt(
    ignore_index=False,
    value_vars=['st-net:gene_pearson', "ensemble:gene_pearson", "mlp:gene_pearson", "ridge:gene_pearson", "linear:gene_pearson",], 
    var_name="method",
    value_name="score")
df_long["name"] = df_long.index
df_long['method'] = df_long['method'].str.split(':').str[0].str.capitalize()
# rename 'St-net' to 'ST-Net' and 'Mlp' to 'MLP'
df_long['method'] = df_long['method'].replace({'St-net': 'ST-Net', 'Mlp': 'MLP'})
# df_long.head()
df_long = df_long.reset_index(drop=True)
df_long.head()

# plot
plt.figure(figsize=(4.5,3.5))
ax = sns.violinplot(x="method", y="score", data=df_long, palette='Set1', alpha=0.6)
ax.set_ylabel('Pearson of each slice')
ax.set_xlabel(' ')
sns.stripplot(
    x="method", y="score", data=df_long, s=2,
    jitter=True, dodge=False,
    palette='Set1', alpha=0.3
)
sns.despine()


# %%
df_long = res_df.melt(
    ignore_index=False,
    value_vars=['st-net:gene_spearman', "ensemble:gene_spearman", "mlp:gene_spearman", "ridge:gene_spearman", "linear:gene_spearman",], 
    var_name="method",
    value_name="score")
df_long["name"] = df_long.index
df_long['method'] = df_long['method'].str.split(':').str[0].str.capitalize()
df_long['method'] = df_long['method'].replace({'St-net': 'ST-Net', 'Mlp': 'MLP'})

# df_long.head()
df_long = df_long.reset_index(drop=True)
df_long.head()


plt.figure(figsize=(4.5,3.5))
ax = sns.violinplot(x="method", y="score", data=df_long, palette='Set1', alpha=0.6)
ax.set_ylabel('Spearman of each slice')
ax.set_xlabel(' ')
sns.stripplot(
    x="method", y="score", data=df_long, s=2,
    jitter=True, dodge=False,
    palette='Set1', alpha=0.3
)
sns.despine()

# %%
res_df.to_csv('./results/pred_res_per_slice.csv')
