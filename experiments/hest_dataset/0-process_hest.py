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
# # Process [HEST](https://github.com/mahmoodlab/hest) dataset
#
# This dataset includes 1108 slides with 1.5 millon spots, 60M nucleus:
# - ST (552)
# - Visium (515)
# - Visium HD (3)
# - Xenium (38)
#
# > It provides all cell segment results but we just need results for Xenium and Visium HD
#
# For each sample, we provide:,
# ,
# - **wsis/**: H&E-stained whole slide images in pyramidal Generic TIFF (or pyramidal Generic BigTIFF if >4.1GB),
# - **st/**: Spatial transcriptomics expressions in a scanpy .h5ad object,
# - **metadata/**: Metadata,
# - **spatial_plots/**: Overlay of the WSI with the st spots,
# - **thumbnails/**: Downscaled version of the WSI,
# - **tissue_seg/**: Tissue segmentation masks:,
#     - `{id}_mask.jpg`: Downscaled or full resolution greyscale tissue mask,
#     - `{id}_mask.pkl`: Tissue/holes contours in a pickle file,
#     - `{id}_vis.jpg`: Visualization of the tissue mask on the downscaled WSI,
# - **cellvit_seg/**: Cellvit nuclei segmentation,
# - **pixel_size_vis/**: Visualization of the pixel size,
# - **patches/**: 256x256 H&E patches (0.5µm/px) extracted around ST spots in a .h5 object optimized for deep-learning. Each patch is matched to the corresponding ST profile (see **st/**) with a barcode.,
# - **patches_vis/**: Visualization of the mask and patches on a downscaled WSI.
#

# %%
import warnings
import re
from pathlib import Path
from collections import Counter
from tqdm import tqdm

import h5py
import scipy.sparse as sp
import zarr
import scanpy as sc
import pandas as pd
import numpy as np
import ray
import cv2
from PIL import Image
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

# %%
Image.MAX_IMAGE_PIXELS = 3080000000000000000000000000000000000

sc.settings.verbosity = 0
warnings.filterwarnings("ignore")

TARGET_MPP = 0.5 # 20x magnification

# %%
work_dir = Path('../../pipeline/hest')
# result_dir = Path('../../pipeline/results')
index = 'INT1'
count_file = work_dir / 'st' / f'{index}.h5ad'
image_file = work_dir / 'wsis' / f'{index}.tif'
meta_file = work_dir / 'metadata' / f'{index}.json'
assert count_file.exists()
assert image_file.exists()
assert meta_file.exists()

# %%
import cv2

def is_blank_tile(tile, threshold = 0.7) -> bool:
    """
    Remove the tiles whose blank area > threshold
    
    Output:
        (bool): 1: The patch is white, otherwise, 0
    """
    _, white_region = cv2.threshold(tile, 235, 255, cv2.THRESH_BINARY)
    if np.sum(white_region == 255) / (224 * 224) > threshold:
        return True
    else:
        return False



# %%
import re
from collections import Counter

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch
from anndata import AnnData
from loguru import logger
from torch import Tensor


def bulid_gene_map_table(result_dir:str) -> None:
    human_annot = sc.queries.biomart_annotations("hsapiens", 
                                        ["ensembl_gene_id","hgnc_symbol"],
                                        ).set_index("ensembl_gene_id")
    mouse_annot = sc.queries.biomart_annotations("mmusculus", 
                                        ["ensembl_gene_id","mgi_symbol"],
                                        ).set_index("ensembl_gene_id")
    human_annot.to_csv(f'{result_dir}/human_gene_convert.csv')
    mouse_annot.to_csv(f'{result_dir}/mouse_gene_convert.csv')


def gene_freq_from_adatas(adata_list: list[AnnData], use_hvg:bool = False) -> pd.DataFrame:
    r"""
    Get the gene frequency from a list of AnnData
    """
    if use_hvg:
        all_genes = [ad.var_names[ad.var['highly_variable']].tolist() for ad in adata_list]
    else:
        all_genes = [adata.var_names.tolist() for adata in adata_list]
    gene_count = Counter()

    for sublist in all_genes:
        gene_count.update(sublist)
    # print(gene_count)

    gene_count_df = pd.DataFrame.from_dict(gene_count, orient="index").reset_index()
    gene_count_df.columns = ["gene", "count"]
    # sort by count
    gene_count_df = gene_count_df.sort_values("count", ascending=False)
    # gene_count_df.to_csv('./data/processed/crost_gene_count.csv', index=False)
    return gene_count_df


def valid_human_gene_col(genes: list, min_num: int = 2) -> bool:
    r"""
    Check if the gene column is human gene
    """
    human_genes = [
        "CD3D",
        "CCL5",
        "CSF3R",
        "CDK1",
        "JUN",
        "NOTCH1",
        "MYC",
        "B2M",
        "CD14",
        "EGR1",
        "TNFRSF9",
        "MKI67",
        "ISG15"
    ]
    share_num = set(human_genes).intersection(set(genes))
    share_num = len(share_num)
    if share_num > min_num:
        return True
    else:
        return False


def valid_mouse_gene_col(genes: list, verbose: bool = False) -> bool:
    test_genes = np.random.choice(genes, 500, replace=False)
    first_cap_genes = [gene.capitalize() for gene in test_genes]
    mouse_gene_num = 0
    for g1, g2 in zip(test_genes, first_cap_genes):
        if g1 == g2:
            mouse_gene_num += 1
            continue
        if "Rik" in g1:
            mouse_gene_num += 1
            continue
        if "mm10" in g1:
            mouse_gene_num += 1
            continue
        if "ENSM" in g1:
            mouse_gene_num += 1
            continue

    if mouse_gene_num / len(test_genes) > 0.2:
        if verbose:
            print("Found mouse genes")
        return True
    else:
        return False


def valid_ensg_gene_col(genes: list, verbose: bool = False) -> bool:
    test_genes = np.random.choice(genes, 100, replace=False)
    ensg_gene_num = 0
    for gene in test_genes:
        if "ENSG" in gene:
            ensg_gene_num += 1
    if ensg_gene_num / len(test_genes) > 0.5:
        if verbose:
            print("Found ENSG format genes")
        return True
    else:
        return False


def convert_ensg_to_gene_name(
    ensg_genes: list, map_table: str = "./data/human_gene_convert.csv", verbose: bool = False
) -> list:
    if verbose:
        print("Convert ENSG to gene name")

    # replace the "GRCh38______" in all gene (ST data in hest dataset)
    ensg_genes = [re.sub(r"^GRCh38_+|GRCh38_+", "", item) for item in ensg_genes]

    # remove the version number in all gene
    ensg_genes = [re.sub(r"\.\d+$", "", item) for item in ensg_genes]

    human_anno = pd.read_csv(map_table).set_index("ensembl_gene_id")
    # remove the row with nan
    human_anno = human_anno.dropna()
    human_anno = human_anno.to_dict()["hgnc_symbol"]
    return [human_anno.get(gene, gene) for gene in ensg_genes]


def is_valid_gene(gene: str) -> bool:
    pattern = r"\d{4,}|^RP\d{1,}|^RPL\d{1,}|^RPS\d{1,}|^RNU|^RN7S|^__|\.|-AS|-DT"
    if re.search(pattern, gene):
        return False
    else:
        return True

def is_count_data(data: sp.csr_matrix, tol = 1e-4) -> bool:
    if np.allclose(data.data, np.round(data.data), atol=tol):
        return True
    else:
        return False

def to_csr_mat(data) -> sp.csr_matrix:
    if isinstance(data, sp.spmatrix):
        if not isinstance(data, sp.csr_matrix):
            data = data.tocsr()
    elif isinstance(data, np.ndarray):
        data = sp.csr_matrix(data)
    else:
        raise ValueError(f"Unsupport data type {type(data)}")
    return data


def filter_adata(
    adata: sc.AnnData,
    min_genes: int = 200,
    dataset_min_gene: int = 1000,
    dataset_min_cells: int = 100,
    verbose: bool = True,
    remove_non_counts: bool = False,
) -> sc.AnnData:
    r"""
    Filter and check the adata (if not passed, return `None`)
    """
    adata.X = to_csr_mat(adata.X)
    adata.var_names_make_unique()
    adata.obs_names_make_unique()

    # filter cells
    sc.pp.filter_cells(adata, min_genes=min_genes)
    if np.isnan(adata.X.data).any():
        cell_sum = adata.X.sum(1)
        nan_cell = np.isnan(cell_sum).flatten()
        if verbose:
            print(f"Found nan in the adata.X, remove {nan_cell.sum()} cells")
        adata = adata[~nan_cell].copy()

    # filter slices
    if adata.n_obs < dataset_min_cells or adata.n_vars < dataset_min_gene:
        print(f"Remove slice with shape {adata.shape} ")
        return None
    if not np.isclose(adata.X.min(), 0):
        if verbose:
            print("Min value != 0")
        return None
    elif np.signbit(adata.X.data).any():
        if verbose:
            print("Found negative value in the adata.X")
        return None

    # try to recover raw adata
    if adata.X.max() < 10:
        print("Warning: adata.X is logged, because adata.X.max() < 10")
        adata.X.data = np.expm1(adata.X.data)
    found_count = False
    if not is_count_data(adata.X):
        print('Warning: adata.X is not integer')
        for k, v in adata.layers.items():
            v = to_csr_mat(v)
            if is_count_data(v):
                found_count = True
                print(f'Found counts data in layer :{k}')
                adata.X = v.copy()
    else:
        found_count = True
        adata.X = adata.X.astype(np.int32)
    if not found_count and remove_non_counts:
        if verbose:
            print("Remove non-counts data")
        return None

    # try to get valid gene name
    ensg_genes, found_gene_name = None, False
    raw_genes = adata.var.index.tolist()
    if valid_mouse_gene_col(raw_genes, verbose):
        return None  # this is a mouse data
    elif valid_ensg_gene_col(raw_genes, verbose):
        ensg_genes = adata.var.index.tolist()
    elif valid_human_gene_col(raw_genes):
        found_gene_name = True

    all_cols = adata.var.columns.tolist()
    for col_name in all_cols:
        genes = adata.var[col_name].tolist()
        if not isinstance(genes[0], str):
            continue
        elif Counter(genes) == Counter(raw_genes):
            continue
        elif len(set(genes)) < 50:
            continue
        elif valid_mouse_gene_col(genes, verbose):
            return None
        elif valid_ensg_gene_col(genes, verbose):
            ensg_genes = adata.var[col_name].tolist()
        elif valid_human_gene_col(genes):
            adata.var.index = adata.var[col_name]
            found_gene_name = True

    if found_gene_name:
        pass
    elif ensg_genes is not None:
        gene_names = convert_ensg_to_gene_name(ensg_genes, verbose=verbose)
        adata.var.index = gene_names
        # remove the row with ENSG
        adata = adata[:, ~adata.var.index.str.startswith("ENSG")]
    else:
        if verbose:
            print("Can not handle the adata.var :")
            print(adata.var.head())
        return None  # can not find valid genes yet

    # remove invalid gene in adata
    genes = adata.var.index.tolist()
    valid_gene_idx = np.array([is_valid_gene(gene) for gene in genes])
    adata = adata[:, valid_gene_idx]
    adata.var.index = adata.var.index.astype(str)
    adata.var_names_make_unique()

    sc.pp.filter_cells(adata, min_genes=min_genes)
    if adata.n_obs < dataset_min_cells or adata.n_vars < dataset_min_gene:
        print(f"Remove slice with final shape {adata.shape}")
        return None
    else:
        del adata.layers
        return adata.copy()



# %% [markdown]
# ## Filter the meta

# %%
meta = pd.read_csv(work_dir / 'HEST_v1_1_0.csv')
meta_human = meta[meta['species'] == 'Homo sapiens']
meta_mouse = meta[meta['species'] == 'Mus musculus']
print(f'Human: {meta_human.shape[0]}, Mouse: {meta_mouse.shape[0]}')

# %%
print('Mouse meta')
stat = meta_mouse.groupby(['disease_state', 'organ']).size().reset_index().rename(columns={0: 'count'})
# from long to wide
stat = stat.pivot(index='disease_state', columns='organ', values='count').fillna(0).T
print(stat)

print('----------------------------')
print('Human meta')
stat = meta_human.groupby(['disease_state', 'organ']).size().reset_index().rename(columns={0: 'count'})
# from long to wide
stat = stat.pivot(index='disease_state', columns='organ', values='count').fillna(0).T
print(stat)

# %%
# filter our heart data
meta_human_filter = meta_human[~meta_human['organ'].isin(['Heart', 'Bone'])]
print(meta_human_filter.shape)
meta = meta_human_filter

# %%
meta['organ'] = meta['organ'].str.replace(' ', '_')
meta['organ'] = meta['organ'].str.replace('/', '_and_')
meta['st_technology'] = meta['st_technology'].str.replace('Spatial Transcriptomics', 'ST')
meta = meta[~meta['st_technology'].isin(['Xenium', 'Visium HD'])]
meta['st_technology'].value_counts()

# %% [markdown]
# ## Read in data

# %%
adata_files = [work_dir / 'st' / f'{id}.h5ad' for id in meta['id']]

# print(adata_files)
print(len(adata_files))
for file in adata_files:
    assert file.exists()


# %%
@ray.remote(num_cpus=1)
def ray_read_adata_with_images(adata_path, tile_emb_path):
    return read_adata_with_images(adata_path, tile_emb_path)

def read_adata_with_images(name, technology):
    adata_path = work_dir / 'st' / f'{name}.h5ad'
    h5_path = work_dir / 'patches' / f'{name}.h5'
    assert adata_path.exists()
    assert h5_path.exists()
    
    adata = sc.read_h5ad(adata_path)
    raw_cells = adata.n_obs

    with h5py.File(h5_path, "r") as f:
        keys = list(f.keys())
        tiles = f["img"][()]
        # coords = f["coords"][()]
        if 'barcode' in keys:
            barcodes = f["barcode"][()]
        elif 'barcodes' in keys:
            barcodes = f["barcodes"][()]
        else:
            raise ValueError("No barcode found in h5 file")
    
    # some spot is not inside the image
    barcodes = [i[0].decode("utf-8") for i in barcodes.tolist()]
    # some spot is not inside the image
    if tiles.shape[0] != adata.n_obs:
        # here is a trick to remove the suffix
        if all([len(i) == 25 for i in barcodes]):
            barcodes = [i.rstrip('_') for i in barcodes]
        cell_mask = np.isin(adata.obs_names, barcodes)
        adata = adata[cell_mask]

    barcodes = np.array(barcodes)
    if adata.n_obs != tiles.shape[0]:
        tile_mask = np.isin(barcodes, adata.obs_names)
        barcodes = barcodes[tile_mask]
        tiles = tiles[tile_mask]

    assert adata.n_obs == tiles.shape[0]
    adata.obsm["X_image"] = tiles
    
    # filter out blank tiles
    blank_mask = np.array([not is_blank_tile(i) for i in tiles])
    adata = adata[blank_mask, : ].copy()

    adata.obs['tile_barcode'] = barcodes[blank_mask]
    adata.obs['__slide'] = name
    adata.obs['__source'] = 'hest'

    keep_ratio = adata.n_obs / raw_cells * 100
    if keep_ratio < 70:
        print(f'Only {adata.n_obs} cells kept, {keep_ratio:.2f}% in {name}')
    # else:
    #     print(f'{adata.n_obs} cells kept, {keep_ratio:.2f}% ')
    if adata.n_obs > 2000:
        try:
            sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat_v3")
        except:
            print(f'Error find HVG in {name}')
    else:
        adata.var['highly_variable'] = True
    return filter_adata(adata, remove_non_counts = True)

# simple test (NCBI865, NCBI864)
# read_adata_with_images('NCBI864', 'Visium')



# %%
# human_adatas = []
# for i, dic in meta.iterrows():
#     print('=' * 30, i, '=' * 30 )
#     print(dic['id'])
#     adata = read_adata_with_images(dic['id'], dic['st_technology'])
#     if adata is not None:
#         human_adatas.append(adata)
#     else:
#         print(f'Error in {dic["id"]}')

# %%
ray.init()
human_adatas = [ray_read_adata_with_images.remote(dic['id'], dic['st_technology']) for _, dic in meta.iterrows()]
human_adatas = ray.get(human_adatas)
ray.shutdown()

del_idx = []
for i, adata in enumerate(human_adatas):
    if adata is None:
        del_idx.append(i)

# remove failed loads
if len(del_idx) > 0:
    for j in sorted(del_idx, reverse=True):
        del human_adatas[j]

# %%
len(human_adatas)

# %% [markdown]
# # Gene freq analysis 

# %% [markdown]
# ## Gene frequence

# %%
human_count_df = gene_freq_from_adatas(human_adatas)
human_count_df.to_csv('./results/human_gene_counts.csv')
human_count_df.head()

print(human_count_df['count'].describe())

plt.figure(figsize=(4, 3))
_ = plt.hist(human_count_df['count'], bins=50)

# %% [markdown]
# ## HVGs

# %%
hvg_count_df = gene_freq_from_adatas(human_adatas, use_hvg=True)
hvg_count_df.to_csv('./results/human_hvg_counts.csv')
print(hvg_count_df['count'].describe())
print(hvg_count_df.head())
plt.figure(figsize=(4, 3))
hvg_count_df['count'].plot.hist(bins=100, ylim = (0, 8000))

# %% [markdown]
# ## Gene expression level (deprecated)

# %%
# gene_high_expr_list = []
# for adata in human_adatas:
#     high_expr_mask = adata.var['n_cells'] / adata.n_obs > 0.05
#     high_expr_genes = adata.var_names[high_expr_mask].tolist()
#     gene_high_expr_list.append(high_expr_genes)


# high_expr_count = Counter()

# for sublist in gene_high_expr_list:
#     high_expr_count.update(sublist)
# high_expr_count = pd.DataFrame.from_dict(high_expr_count, orient='index').reset_index()
# high_expr_count.columns = ['gene', 'count']
# # sort by count
# high_expr_count = high_expr_count.sort_values('count', ascending=False)
# print(high_expr_count.head())

# high_expr_count['count'].plot.hist(bins=100, ylim = (0, 8000))
# high_expr_count.to_csv('./results/human_high_expr_counts.csv')

# %% [markdown]
# # Merge and save

# %%
human_adata = sc.concat(human_adatas, join='outer', index_unique='-', label='__batch')
print(human_adata)

# %%
hest_valid_gene = human_count_df[human_count_df['count'] > 400]['gene'].values
hest_hvg_gene = hvg_count_df[hvg_count_df['count'] > 200]['gene'].values
hest_valid_gene = set(hest_valid_gene).intersection(set(hest_hvg_gene))
len(hest_valid_gene)

# %%
human_adata = human_adata[:, human_adata.var.index.isin(hest_valid_gene)].copy()
human_adata

# %%
is_count_data(human_adata.X)

# %%
image_tile = human_adata.obsm['X_image']
print(image_tile.shape)

# convert image_tile to zarr format and save
# > about 5 mins for 800k tiles (500GB memory and 100GB disk)
image_tile = zarr.array(image_tile, chunks=(64, 224, 224, 3), dtype='uint8')
zarr.save('./results/image_tile.zarr', image_tile)

del human_adata.obsm['X_image']

# %%
meta = pd.read_csv(work_dir / 'HEST_v1_1_0.csv')
human_adata.obs = human_adata.obs.merge(meta, left_on='__slide', right_on='id', how='left')
human_adata.obs = human_adata.obs.astype(str)
human_adata.obs.to_parquet('./results/human_adata_obs.parquet')

# %%
human_adata.obs = human_adata.obs.astype(str)
human_adata.X = human_adata.X.astype(np.int32)
human_adata.write_h5ad('./results/human_adata.h5ad')

# %%
sc.pp.filter_cells(human_adata, min_genes=0)
sc.pp.filter_cells(human_adata, min_counts=0)

human_adata.var['n_cells'].plot.hist(bins=100)
human_adata.obs['n_genes'].plot.hist(bins=100)

# %% [markdown]
# # Visualization
#
# ## Raw

# %%
human_adata = sc.read_h5ad('./results/human_adata.h5ad')

# %%
# human_adata = gex_embedding(human_adata, n_top_genes=10_000, approx = True)  # do not select hvgs

# %%
sc.pl.umap(human_adata, color=['organ', 'n_genes', 'n_counts'], ncols=1, norm=mcolors.LogNorm())

# %% [markdown]
# ## Subset genes

# %%
adata = sc.read_h5ad('./results/human_adata.h5ad')
adata

# %%
uni_raw_emb = np.load('./results/uni_raw_tile_emb.npy')


# %%
marker_gene_df = pd.read_csv('./results/cell_type_marker_genes_grouped.csv')
all_cell_marker_genes = []
for i, row in marker_gene_df.iterrows():
    genes = row['marker gene list'].split(';')
    genes = [g.strip() for g in genes]
    all_cell_marker_genes += genes

print(set(all_cell_marker_genes)-set(adata.var_names))

# %%
bio_marker_df = pd.read_csv('./results/cancer_biomarker_targets_with_category.csv')
all_bio_marker_genes = []
for i, row in bio_marker_df.iterrows():
    genes = row['Gene'].split(';')
    genes = [g.strip() for g in genes]
    all_bio_marker_genes += genes

print(set(all_bio_marker_genes)-set(adata.var_names))

# %%
marker_genes = all_bio_marker_genes + all_cell_marker_genes
marker_genes = list(set(marker_genes) & set(adata.var_names))
len(marker_genes)

# %%
# 2min
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# %%
adata_sub = adata[:, marker_genes].copy()

# %%
adata_sub.obsm['X_uni'] = uni_raw_emb

# %%
adata_sub.write_h5ad('./results/human_adata_marker_genes_log1p.h5ad')

# %%
adata_sub.X.sum(1).min()
