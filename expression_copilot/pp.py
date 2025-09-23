r"""
Data preprocess.
"""
import scanpy as sc


def process_omics(
    adata: sc.AnnData,
    min_cell: int = 3,
    min_counts: int = 3,
    n_top_genes: int = 2000,
    subset_hvg: bool = True,
) -> None:
    r"""
    Process the omics data.

    Args:
        adata: AnnData object
        min_cell: filter genes expressed in at least min_cell cells, default 3
        min_counts: filter genes with at least min_counts counts, default 3
        n_top_genes: number of highly variable genes to select, default 2000
        subset_hvg: whether to subset highly variable genes, default False
        method: method for omics embedding, currently only support "pca"
    """
    sc.pp.filter_genes(adata, min_cells=min_cell)
    sc.pp.filter_genes(adata, min_counts=min_counts)
    if adata.n_obs > n_top_genes:
        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=n_top_genes,
            flavor="seurat_v3",
            subset=subset_hvg,
        )
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    expr = adata.X.copy()
    expr = expr.toarray() if hasattr(expr, "toarray") else expr
    adata.layers["log1p"] = expr
    sc.pp.scale(adata, max_value=10)
