import numpy as np
import numba as nb
from tools.utils import reshape_wrapper, broadcast_kernel

@nb.jit()
def compute_mmd(X, Y, kernel):
    K_XX = broadcast_kernel(kernel.kernel, X, X)
    K_YY = broadcast_kernel(kernel.kernel, Y, Y)
    K_XY = broadcast_kernel(kernel.kernel, X, Y)

    mmd = np.mean(K_XX) + np.mean(K_YY) - 2 * np.mean(K_XY)
    return mmd

@nb.jit()
def compute_mmd_small_weighted(X, Y, kernel, weights_X=None, weights_Y=None):
    """
    Computes the weighted MMD between two small sets of samples X and Y, which might be weighted.

    Parameters:
    - X: numpy array of shape (n, d), first set of samples.
    - Y: numpy array of shape (m, d), second set of samples.
    - kernel: a kernel function object with a `kernel` method.
    - weights_X: numpy array of shape (n,), weights for samples in X (optional, defaults to uniform).
    - weights_Y: numpy array of shape (m,), weights for samples in Y (optional, defaults to uniform).

    Returns:
    - mmd: the weighted MMD value.
    """
    if weights_X is None:
        weights_X = np.ones(len(X)) / len(X)
    if weights_Y is None:
        weights_Y = np.ones(len(Y)) / len(Y)

    K_XX = broadcast_kernel(kernel, X, X)
    K_YY = broadcast_kernel(kernel, Y, Y)
    K_XY = broadcast_kernel(kernel, X, Y)

    # Weighted means
    mmd = (
        weights_X.T.dot(K_XX).dot(weights_X) +
        weights_Y.T.dot(K_YY).dot(weights_Y) -
        2 * weights_X.T.dot(K_XY).dot(weights_Y)
    )
    return mmd

@nb.jit()
def compute_mmd_large_weighted_Y(X, Y, kernel, weights_Y = None):
    """
    Computes the MMD between two sets of samples X and Y.

    Parameters:
    - X: numpy array of shape (n, d), first set of samples.
    - Y: numpy array of shape (m, d), second set of samples.
    - kernel: a kernel function object with a `kernel` method.
    - weights_Y: numpy array of shape (m,), weights for samples in Y.

    Returns:
    - mmd: the weighted MMD value.
    """
    d = X.shape[1]
    assert d == Y.shape[1], f"Dimension mismatch: {d} != {Y.shape[1]}"
    N, M = len(X), len(Y)
    assert N >= M, f"Dimension mismatch: {N} < {M}"
    # If no weights are provided, use uniform weights
    if weights_Y is None:
        weights_Y = np.ones(M) / M
    else:
        assert M == len(weights_Y), f"Dimension mismatch: {M} != {len(weights_Y)}"
    x_wt = 1./N
    mmd_sq = 0
    # Compute the MMD^2
    # Bottlenecked by the number of samples in X when calculating K_XX
    for idx in nb.prange(N*N):
        i,j = idx // N, idx % N
        add_to_mmd = 0.
        # Add K_XX (noting symmetry): we know i < N
        if j <= i:
            mul = 1 + (i != j)
            add_to_mmd += mul * kernel(X[i], X[j]) * x_wt * x_wt
        # Add K_YY (noting symmetry)
        if i < M and j <= i:
            y_wt_i, y_wt_j = weights_Y[i], weights_Y[j]
            mul = 1 + (i != j)
            add_to_mmd += mul * kernel(Y[i], Y[j]) * y_wt_i * y_wt_j
        # Subtract 2K_XY (no symmetry). We know i < N
        if j < M:
            y_wt_i = weights_Y[j]
            add_to_mmd -= 2 * kernel(X[i], Y[j]) * y_wt_i * x_wt
        mmd_sq += add_to_mmd
    return np.sqrt(mmd_sq)

@nb.jit()
def compute_mmd_weighted(X, Y, kernel, weights_X = None, weights_Y = None):
    """
    Computes the weighted MMD between two sets of samples X and Y.

    Parameters:
    - X: numpy array of shape (n, d), first set of samples.
    - Y: numpy array of shape (m, d), second set of samples.
    - kernel: a kernel function object with a `kernel` method.
    - weights_X: numpy array of shape (n,), weights for samples in X (optional, defaults to uniform).
    - weights_Y: numpy array of shape (m,), weights for samples in Y (optional, defaults to uniform).

    Returns:
    - mmd: the weighted MMD value.
    """
    MAX_SMALL_SIZE = 10000
    # Make sure that everything has compatible sizes
    d = X.shape[1]
    assert d == Y.shape[1], f"Dimension mismatch: {d} != {Y.shape[1]}"
    N, M = len(X), len(Y)

    # Make sure that X is the larger dataset and switch if necessary
    if N < M:
        X, Y = Y, X
        weights_X, weights_Y = weights_Y, weights_X
        N, M = M, N
        # TODO: This should handle N==M more gracefully,
        # but we don't need to worry about that for now

    # If the datasets are small or we have nonzero weights on data X, use the small dataset version
    if N*M < MAX_SMALL_SIZE or (weights_X is not None):
        return compute_mmd_small_weighted(X, Y, kernel, weights_X, weights_Y)
    # Otherwise, use the large dataset version
    else:
        return compute_mmd_large_weighted_Y(X, Y, kernel, weights_Y)

@nb.jit(parallel=True)
def compute_mmd_entropy_large_unweighted(X, kernel):
    """
    Computes the "MMD entropy" of unweighted samples, E_{x,x'}[k(x,x')]

    Parameters:
    - X: numpy array of shape (n, d), first set of samples.
    - kernel: a kernel function object with a `kernel` method.

    Returns:
    - mmds: numpy array of shape (n,), where mmds[i] is the MMD between X and X[i].
    """
    N = len(X)
    mmd = np.float64(0.)
    for idx in nb.prange(N*N):
        i,j = idx // N, idx % N
        add_to_mmd = np.float64(0.)
        # Add K_XX (noting symmetry): we know i < N
        if j <= i:
            mul = 1 + (i != j)
            add_to_mmd += mul * kernel(X[i], X[j])
        mmd += add_to_mmd
    return mmd / (np.float64(N)**2)

@nb.jit()
def compute_cross_mmd_large_weighted_Y(X, Y, kernel, weights_Y = None):
    """
    If MMD^2 = K_XX + K_YY - 2K_XY, this calculates the Y terms K_YY-2K_XY.

    Parameters:
    - X: numpy array of shape (n, d), first set of samples.
    - Y: numpy array of shape (m, d), second set of samples.
    - kernel: a kernel function object with a `kernel` method.
    - weights_Y: numpy array of shape (m,), weights for samples in Y.

    Returns:
    - mmd: the weighted MMD value.
    """
    d = X.shape[1]
    assert d == Y.shape[1], f"Dimension mismatch: {d} != {Y.shape[1]}"
    N, M = len(X), len(Y)
    assert N >= M, f"Dimension mismatch: {N} < {M}"
    # If no weights are provided, use uniform weights
    if weights_Y is None:
        weights_Y = np.ones(M) / M
    else:
        assert M == len(weights_Y), f"Dimension mismatch: {M} != {len(weights_Y)}"
    x_wt = 1 / np.float64(N)
    cross_mmd_sq = 0
    # Bottlenecked by the number of samples in X when calculating K_XY
    for idx in nb.prange(N*M):
        i,j = idx // M, idx % M
        add_to_mmd = 0.
        # Add K_YY (noting symmetry)
        if i < M and j <= i:
            y_wt_i, y_wt_j = np.float64(weights_Y[i]), np.float64(weights_Y[j])
            mul = 1 + (i != j)
            add_to_mmd += mul * np.float64(kernel(Y[i], Y[j])) * y_wt_i * y_wt_j
        # Subtract 2K_XY (no symmetry). We know i < N
        y_wt_i = weights_Y[j]
        add_to_mmd -= 2 * kernel(X[i], Y[j]) * y_wt_i * x_wt
        cross_mmd_sq += add_to_mmd
    return cross_mmd_sq

@nb.jit(parallel=True)
def mmd_cross_entropy_array(all_nodes, all_node_weights, data, kernel, data_mmd):
    mmds = np.zeros(len(all_nodes))
    for i in nb.prange(len(all_nodes)):
        Y, weights_Y = all_nodes[i], all_node_weights[i]
        mmd_sq = data_mmd
        mmd_sq += compute_cross_mmd_large_weighted_Y(data, Y, kernel, weights_Y=weights_Y)
        mmds[i] = np.sqrt(mmd_sq)
    return mmds

def mmd_array_cached(all_nodes, all_node_weights, data, kernel, mmd_self):
    """
    Compute the MMD between data and all_nodes, with weights given by all_node_weights.

    Parameters:
    - data: numpy array of shape (n, d), first set of samples.
    - all_nodes: numpy array of shape (P, m, d), second set of samples.
    - all_node_weights: numpy array of shape (P, m), weights for samples in all_nodes.
    - kernel: a kernel function object with a `kernel` method.

    Returns:
    - mmds: numpy array of shape (P,), where mmds[i] is the MMD between data and all_nodes[i].
    """
    P, m, d = all_nodes.shape
    assert P == len(all_node_weights), f"Dimension mismatch: {P} != {len(all_node_weights)}"
    assert m == all_node_weights.shape[1], f"Dimension mismatch: {m} != {all_node_weights.shape[1]}"
    assert d == data.shape[1], f"Dimension mismatch: {d} != {data.shape[1]}"
    data_mmd = mmd_self[kernel]
    kernel_eval = kernel.kernel
    if data_mmd is None:
        data_mmd = compute_mmd_entropy_large_unweighted(data, kernel_eval)
        mmd_self[kernel] = data_mmd

    mmds = mmd_cross_entropy_array(all_nodes, all_node_weights, data, kernel_eval, data_mmd)
    return mmds

@nb.jit(parallel=True)
def mmd_array_uncached(all_nodes, all_node_weights, data, kernel):
    """
    Compute the MMD between data and all_nodes, with weights given by all_node_weights.

    Parameters:
    - data: numpy array of shape (n, d), first set of samples.
    - all_nodes: numpy array of shape (P, m, d), second set of samples.
    - all_node_weights: numpy array of shape (P, m), weights for samples in all_nodes.
    - kernel: a kernel function object with a `kernel` method.

    Returns:
    - mmds: numpy array of shape (P,), where mmds[i] is the MMD between data and all_nodes[i].
    """
    P, m, d = all_nodes.shape

    # Check dimensions
    assert P == len(all_node_weights), f"Dimension mismatch: {P} != {len(all_node_weights)}"
    assert m == all_node_weights.shape[1], f"Dimension mismatch: {m} != {all_node_weights.shape[1]}"
    assert d == data.shape[1], f"Dimension mismatch: {d} != {data.shape[1]}"

    # Compute MMDs in parallel
    mmds = np.zeros(len(all_nodes))
    for i in nb.prange(len(all_nodes)):
        Y, weights_Y = all_nodes[i], all_node_weights[i]
        mmd_sq = compute_mmd_weighted(data, Y, kernel, weights_Y=weights_Y)
        mmds[i] = np.sqrt(mmd_sq)
    return mmds

def mmd_array(all_nodes, all_node_weights, data, kernel, mmd_self):
    mmds = reshape_wrapper(mmd_array_cached, all_nodes, all_node_weights, data, kernel, mmd_self)
    return mmds
