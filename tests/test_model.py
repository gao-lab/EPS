import random

import numpy as np
import pandas as pd
import pytest
import scanpy as sc

from expression_copilot.model import ExpressionCopilotModel


def simulate_adata(
    N_CELLS: int = 800,
    N_GENES: int = 1000,
    N_CELL_TYPES: int = 8,
    N_IMAGE_EMB_DIM: int = 10,
) -> sc.AnnData:
    r"""
    Simluate an AnnData object for testing.
    """
    np.random.seed(0)
    X = np.random.randint(0, 5, (N_CELLS, N_GENES))
    cell_type = np.random.randint(0, N_CELL_TYPES, N_CELLS)
    cell_type = [f"cell_{i}" for i in cell_type.tolist()]
    genes = [f"gene_{i}" for i in range(N_GENES)]
    random.shuffle(genes)
    image_emb = np.random.rand(N_CELLS, N_IMAGE_EMB_DIM)

    adata = sc.AnnData(
        X=X,
        obs=pd.DataFrame({"cell_type": cell_type}),
        obsm={"X_image_emb": image_emb},
    )
    adata.var_names = genes
    return adata


@pytest.mark.parametrize(
    "shuffle",
    [(True), (False)],
)
def test_model(shuffle):
    adata = simulate_adata()
    model = ExpressionCopilotModel(adata, image_key="X_image_emb")
    genes = ["gene_2"]

    _ = model.calc_metrics_per_gene(shuffle=shuffle)
    _ = model.calc_metrics_per_gene(shuffle=shuffle, genes=genes)
    print("Testing baseline:", "-" * 50)
    _ = model.calc_baseline_metrics(shuffle=shuffle)
    print("Testing with scale data:", "-" * 50)
    _ = model.calc_baseline_metrics(shuffle=shuffle, scale=True)
    print("Testing with specific genes:", "-" * 100)
    _ = model.calc_baseline_metrics(shuffle=shuffle, genes=genes)
    model.save_results("results/test_model_results.pkl")


if __name__ == "__main__":
    test_model(False)
    print("Model test passed.")
