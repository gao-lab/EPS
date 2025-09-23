r"""
Utility functions
"""
import faiss
import numpy as np
from annoy import AnnoyIndex
from loguru import logger
from scipy.sparse import csr_matrix


def is_torch_ready() -> bool:
    try:
        import torch

        TORCH_FLAG = torch.cuda.is_available()
    except ImportError:
        TORCH_FLAG = False
    return TORCH_FLAG


def is_rapids_ready() -> bool:
    try:
        import cupy
        import rmm

        RSC_FLAG = cupy.cuda.is_available() and rmm.is_initialized()
    except:  # noqa
        RSC_FLAG = False
        logger.warning("Rapids not available, use Scanpy")
    return RSC_FLAG


def knn(
    query: np.ndarray,
    ref: np.ndarray = None,
    k: int = 30,
    metric: str = "euclidean",
    approx: bool = False,
    method: list[str] | str = "auto",
    method_params: dict = {},
) -> tuple[np.ndarray, np.ndarray]:
    r"""
    Build k-NN graph, support multiple backends

    Args:
        query: (n_samples, n_features) for building graph
        ref: (n_ref_samples, n_features) for building graph, if None, use query
        k: number of neighbors for k-NN graph
        metric: distance metric for k-NN graph, one of ["euclidean", "cosine"]
        approx: whether to use approximate nearest neighbor search, default False
        method: method for building k-NN graph, one of ["auto", "cuml", "faiss", "annoy"], default "auto"
        method_params: additional parameters for the method

    Note:
        if ref is None, use self as ref, will ignore the itself in the result.
    """
    CUML_FLAG = is_rapids_ready()

    if method == "auto":
        method = ["cuml", "faiss", "annoy"]
    method = method if isinstance(method, list) else [method]
    if not CUML_FLAG and "cuml" in method:
        method.remove("cuml")
    if not approx and "annoy" in method:
        method.remove("annoy")
    method = method[0]
    # approx_str = " approx " if approx else " "
    # logger.debug(f"Use {method} to compute{approx_str}KNN graph.")

    # if ref is None, use self as ref
    if ref is None:
        self_as_ref = True
        ref = query
        k = k + 1
    else:
        self_as_ref = False

    if "cuml" in method:
        distances, indices = knn_cuml(ref, query, k, metric)
    elif "faiss" in method:
        distances, indices = knn_faiss(ref, query, k, metric, approx)
    elif "annoy" in method:
        distances, indices = knn_annoy(ref, query, k, metric, **method_params)

    if self_as_ref:
        distances, indices = distances[:, 1:], indices[:, 1:]

    return indices, distances


def knn_faiss(
    ref: np.ndarray,
    query: np.ndarray,
    k: int = 30,
    metric: str = "euclidean",
    approx: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    r"""
    Build k-NN graph by faiss
    """
    shape = ref.shape[0]
    ref = np.ascontiguousarray(ref.astype(np.float32))
    query = np.ascontiguousarray(query.astype(np.float32))
    faiss.normalize_L2(ref)
    faiss.normalize_L2(query)
    if metric == "euclidean":
        metric = faiss.METRIC_L2
    elif metric == "cosine":
        metric = faiss.METRIC_INNER_PRODUCT

    if approx:  # 1M
        logger.debug(f"Use HNSW64 index with {shape} elements.")
        index = faiss.index_factory(ref.shape[1], "HNSW64", metric)
        index.train(ref)
    else:  # 500k
        logger.debug(f"Use Flat index with {shape} elements.")
        index = faiss.index_factory(ref.shape[1], "Flat", metric)
    index.add(ref)

    distances, indices = index.search(query, k)
    return distances, indices


def knn_cuml(
    ref: np.ndarray,
    query: np.ndarray,
    k: int = 30,
    metric: str = "euclidean",
) -> tuple[np.ndarray, np.ndarray]:
    r"""
    Build k-NN graph by cuML
    """
    from cuml.neighbors import NearestNeighbors as cuNearestNeighbors

    model = cuNearestNeighbors(n_neighbors=k, metric=metric)
    model.fit(ref)
    distances, indices = model.kneighbors(query)
    return distances, indices


def knn_annoy(
    ref: np.ndarray,
    query: np.ndarray = None,
    k: int = 30,
    metric: str = "euclidean",
    n_trees: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    r"""
    Build k-NN graph by annoy
    """
    metric = "angular" if metric == "cosine" else metric
    if metric == "cosine":
        # normalize the vectors
        ref = ref.astype(np.float32)
        query = query.astype(np.float32)
        faiss.normalize_L2(ref)
        faiss.normalize_L2(query)

    index = AnnoyIndex(ref.shape[1], metric)
    for i in np.arange(ref.shape[0]):
        index.add_item(i, ref[i])
    index.build(n_trees)

    ind_list, dist_list = [], []
    for i in np.arange(query.shape[0]):
        holder = index.get_nns_by_vector(query[i], k, include_distances=True)
        ind_list.append(holder[0])
        dist_list.append(holder[1])
    return np.asarray(dist_list), np.asarray(ind_list)


def knn_to_csr(knn_indices: np.ndarray, knn_dists: np.ndarray = None) -> csr_matrix:
    r"""
    Convert knn indices to csr matrix

    Args:
        knn_indices: (n_samples, k) knn indices
        knn_dists: (n_samples, k) knn distances, if None, will use 1s
    """
    num_nodes = knn_indices.shape[0]
    k = knn_indices.shape[1]

    row = np.repeat(np.arange(num_nodes), k)
    col = knn_indices.flatten()
    if knn_dists is None:
        knn_dists = np.ones_like(knn_indices, dtype=np.float32)
    dists = knn_dists.flatten()

    csr = csr_matrix((dists, (row, col)), shape=(num_nodes, num_nodes))

    return csr
