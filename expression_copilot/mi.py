"""
Mutual Information (MI) related functions.
"""
import numpy as np
from loguru import logger
from numba import njit, prange
from numpy.typing import NDArray

from .utils import knn, knn_to_csr


@njit(cache=True, parallel=False)
def _inner_sparse_x_densevec(
    g_data: np.ndarray,
    g_indices: np.ndarray,
    g_indptr: np.ndarray,
    x: np.ndarray,
    W: np.float64,
) -> float:
    x_bar = x.mean()
    total = 0.0
    n = len(x)
    for i in prange(n):
        s = slice(g_indptr[i], g_indptr[i + 1])
        i_indices = g_indices[s]
        i_data = g_data[s]
        total += np.sum(i_data * ((x[i] - x[i_indices]) ** 2))
    numer = (n - 1) * total
    denom = 2 * W * ((x - x_bar) ** 2).sum()
    return numer / denom


@njit
def _graph_laplacian_nd(
    g_data: np.ndarray,
    g_indices: np.ndarray,
    g_indptr: np.ndarray,
    X: np.ndarray,
) -> np.ndarray:
    m, n = X.shape
    assert n == len(g_indptr) - 1
    W = g_data.sum()
    out = np.zeros(m, dtype=np.float64)
    for k in prange(m):
        x = X[k, :].astype(np.float64)
        out[k] = _inner_sparse_x_densevec(g_data, g_indices, g_indptr, x, W)
    return out


@njit
def _graph_laplacian_1d(
    data: np.ndarray,
    indices: np.ndarray,
    indptr: np.ndarray,
    x: np.ndarray,
    W: np.float64,
):
    n = len(indptr) - 1
    x = x.astype(np.float64)
    x_bar = x.mean()

    total = 0.0
    for i in prange(n):
        s = slice(indptr[i], indptr[i + 1])
        i_indices = indices[s]
        i_data = data[s]
        total += np.sum(i_data * ((x[i] - x[i_indices]) ** 2))

    numer = (n - 1) * total
    denom = 2 * W * ((x - x_bar) ** 2).sum()
    return numer / denom


def _check_vals(
    vals: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.bool_] | slice, NDArray[np.float64]]:
    """
    Check that values wont cause issues in computation.
    """
    from scanpy._utils import is_constant

    full_result = np.empty(vals.shape[0], dtype=np.float64)
    full_result.fill(np.nan)
    idxer = ~is_constant(vals, axis=1)
    if idxer.all():
        idxer = slice(None)
    else:
        logger.warning(
            f"{len(idxer) - idxer.sum()} variables were constant, will return nan for these."
        )
    return vals[idxer], idxer, full_result


def graph_signal(
    x: np.ndarray,
    signal: np.ndarray,
    k: int = 10,
    metric: str = "euclidean",
) -> np.ndarray:
    r"""
    Graph Laplacian of the signal on the k-NN graph built from x.

    Args:
        x: (n_samples, n_features) for building graph
        signal: (n_samples, ) or (n_signals, n_samples)
        k: number of neighbors for k-NN graph
        metric: distance metric for k-NN graph

    Note:
        signal dim in n_signals first.

    Returns:
        eps (graph laplacian quadratic form)
    """
    knn_index, _ = knn(x, k=k, metric=metric)
    g = knn_to_csr(knn_index)

    # graph laplacian
    g_data = g.data.astype(np.float64, copy=False)
    if signal.ndim == 1:
        assert g.shape[0] == signal.shape[0]
        W = g_data.sum()
        eps = _graph_laplacian_1d(g_data, g.indices, g.indptr, signal, W)
    else:
        assert g.shape[0] == signal.shape[1]
        new_vals, idxer, full_result = _check_vals(signal)
        result = _graph_laplacian_nd(g_data, g.indices, g.indptr, new_vals)
        full_result[idxer] = result
        eps = full_result
    return eps
