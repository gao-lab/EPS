r"""
Main `ExpressionCopilotModel` class
"""
import pickle as pkl
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from loguru import logger

from .baseline import Baseline
from .mi import graph_signal
from .pp import process_omics


class ExpressionCopilotModel:
    r"""
    Expression Coplot Model for calculating EPS and baseline metrics

    Args:
        adata: AnnData object with image and omics embeddings in .obsm
        image_key: key for image embeddings in .obsm
    """

    def __init__(self, adata: sc.AnnData, image_key: str, hvgs: int = 3000) -> None:
        process_omics(adata, n_top_genes=hvgs)
        self.adata = adata
        self.image_emb = adata.obsm[image_key]

    def _get_expr(self, genes: list = None, scale: bool = False) -> tuple[np.ndarray, list[str]]:
        expr = self.adata.X.copy() if scale else self.adata.layers["log1p"].copy()
        gene_name = self.adata.var_names
        if genes:
            gene_name, _, gene_idx = np.intersect1d(
                genes, self.adata.var_names, return_indices=True
            )
            assert len(gene_name) > 0
            expr = expr[:, gene_idx]
        return expr, gene_name

    def calc_metrics_per_gene(
        self, k: int = 5, scale: bool = True, genes: list[str] = None, shuffle: bool = False
    ) -> pd.DataFrame:
        r"""
        Calculate EPS (Expression Predictability Score) per gene

        Args:
            k: number of neighbors for k-NN graph
            scale: whether to z-score the expression data, default True
            genes: list of genes to calculate, if None, use all genes
            shuffle: whether to shuffle the image embeddings for null model
        """
        expr, gene_name = self._get_expr(genes, scale=scale)
        image_emb = self.image_emb.copy()
        if shuffle:
            np.random.shuffle(image_emb)
            logger.warning("You are shuffling data!")

        eps = graph_signal(image_emb, expr.T, k=k)
        self.gene_metrics = pd.DataFrame({"EPS": eps}, index=gene_name)
        self.gene_metrics.sort_values(by="EPS", ascending=True, inplace=True)
        logger.info(f"Gene metrics: \n{self.gene_metrics.head(3)}")
        return self.gene_metrics

    def calc_baseline_metrics(
        self,
        method: str = "mlp",
        scale: bool = True,
        genes: list[str] = None,
        shuffle: bool = False,
        seed: int = 0,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        r"""
        Calculate baseline metrics by specified method

        Args:
            method: baseline method, one of ["mlp", "ensemble", "ridge", "linear"], default "mlp"
            scale: whether to z-score the expression data, default True
            genes: list of genes to calculate, if None, use all genes
            shuffle: whether to shuffle the image embeddings for null model
            seed: random seed for baseline model
        """
        image_emb = self.image_emb.copy()
        if shuffle:
            np.random.shuffle(image_emb)
        expr, gene_name = self._get_expr(genes, scale=scale)

        pred_model = Baseline(image_emb, expr, method=method, seed=seed)
        pred_model.fit()
        y_pred = pred_model.transform()
        sample_name = self.adata.obs_names[pred_model.test_idx]

        self.baseline_gene_metrics = pred_model.cal_metrics(pred_model.y_test, y_pred, gene_name)
        logger.info(f"Baseline: {method} gene metrics: \n{self.baseline_gene_metrics.mean()}")
        self.baseline_cell_metrics = pred_model.cal_metrics(
            pred_model.y_test.T, y_pred.T, sample_name
        )
        # logger.info(f"Baseline: {method} cell metrics: \n{self.baseline_cell_metrics.mean()}")
        return self.baseline_gene_metrics, self.baseline_cell_metrics

    def save_results(self, save_path: str) -> None:
        r"""
        Save results to the specified file (.pkl format)
        """
        nan = "unknown"
        merge_kwargs = dict(left_index=True, right_index=True, how="left")
        gene_info_df = self.adata.var
        if hasattr(self, "gene_metrics"):
            gene_info_df = self.adata.var.merge(self.gene_metrics, **merge_kwargs)
        if hasattr(self, "baseline_gene_metrics"):
            gene_info_df = gene_info_df.merge(self.baseline_gene_metrics, **merge_kwargs)

        res = dict(
            # data frame
            obs=self.baseline_cell_metrics if hasattr(self, "baseline_cell_metrics") else nan,
            var=gene_info_df,
        )
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            pkl.dump(res, f)
        logger.info(f"Results saved to {save_path}")
