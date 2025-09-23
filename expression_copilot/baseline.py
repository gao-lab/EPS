r"""
Baseline prediction model
"""
import random

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor

from .utils import is_rapids_ready, is_torch_ready


class Baseline:
    r"""
    Baseline model wrapper

    Args:
        image_emb: image embeddings, shape (n_samples, n_features)
        expr: expression data, shape (n_samples, n_genes)
        method: baseline method, one of ["mlp", "ensemble", "ridge", "linear"], default "mlp"
        seed: random seed for baseline model
    """

    def __init__(self, image_emb: np.ndarray, expr: np.ndarray, method: str = "mlp", seed: int = 0):
        random.seed(seed)
        np.random.seed(seed)
        self.seed = seed
        self.image_emb = image_emb
        self.expr = expr
        self.method = method
        # train test split
        indices = np.arange(len(image_emb))
        self.train_idx, self.test_idx = train_test_split(indices, test_size=0.2, random_state=seed)
        self.x_train, self.x_test = image_emb[self.train_idx], image_emb[self.test_idx]
        self.y_train, self.y_test = expr[self.train_idx], expr[self.test_idx]

    def fit(self) -> None:
        r"""
        Model fitting
        """
        if self.method == "linear":  # fast
            if is_rapids_ready():
                from cuml import LinearRegression
            else:
                from sklearn.linear_model import LinearRegression
            self.baseline = LinearRegression(n_jobs=-1)
        elif self.method == "ridge":  # fast
            self.baseline = Ridge(random_state=self.seed)
        elif self.method == "mlp":  # middle (1 min per slice)
            if is_torch_ready():
                from ._torch_regression import TorchMLPRegressor

                self.baseline = TorchMLPRegressor(seed=self.seed)
            else:
                self.baseline = MLPRegressor(
                    max_iter=100, hidden_layer_sizes=(128,), random_state=self.seed
                )
        # elif self.method == 'rf':  # too slow (~10 hours per slice)
        #     self.baseline = RandomForestRegressor(n_estimators=50, n_jobs=-1)
        elif self.method == "ensemble":  # slow: predict gene by gene (10 mins per slice)
            # NOTE: lightGBM/XGBoost only support single target regression
            self.baseline = MultiOutputRegressor(
                lgb.LGBMRegressor(
                    n_estimators=50, learning_rate=0.1, n_jobs=1, random_state=self.seed
                ),
                n_jobs=-1,
            )

        else:
            raise ValueError(f"Unknown method: {self.method}")

        self.baseline.fit(self.x_train, self.y_train)

    def transform(self) -> np.ndarray:
        r"""
        Model prediction
        """
        self.y_pred = self.baseline.predict(self.x_test)
        return self.y_pred

    def cal_metrics(self, y: np.ndarray, y_hat: np.ndarray, names: list = None) -> pd.DataFrame:
        r"""
        Calculate metrics of prediction

        Args:
            y: true values, shape (n_samples, n_genes)
            y_hat: predicted values, shape (n_samples, n_genes)
            names: list of gene names, length n_genes

        Returns:
            DataFrame of metrics per gene, including pearson_corr, spearman_corr, pearson_pval
        """
        if y_hat.ndim == 1:
            y_hat = y_hat.reshape(*y.shape)
        assert y.shape == y_hat.shape
        n_sample = y_hat.shape[0]
        if n_sample < 2:  # no corr
            return pd.DataFrame()
        pearson_corr, pearson_pval = pearsonr(y_hat, y, axis=0)
        spearman_corr, spearman_pval = safe_spearmanr(y_hat, y)
        rmse = root_mean_squared_error(y_hat, y, multioutput="raw_values")

        metrics = pd.DataFrame(
            {
                "pearson_corr": pearson_corr,
                "spearman_corr": spearman_corr,
                "pearson_pval": pearson_pval,
                "spearman_pval": spearman_pval,
                "rmse": rmse,
            },
            index=names,
        )
        metrics.sort_values(by="pearson_corr", ascending=False, inplace=True)
        return metrics


def safe_spearmanr(y_hat: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m, n = y.shape
    corrs = np.full(n, np.nan)
    pvals = np.full(n, np.nan)
    for j in range(n):
        a = y[:, j]
        b = y_hat[:, j]
        res = spearmanr(a, b)
        corrs[j] = res[0]
        pvals[j] = res[1]
    return corrs, pvals
