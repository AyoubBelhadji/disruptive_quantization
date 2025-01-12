import numpy as np
import numba as nb

@nb.jit()
def broadcast_kernel(kernel, x, y):
    """
    Computes the kernel between all pairs of samples in x and y.

    Parameters:
    - kernel: a callable kernel function k(x,y) that is vectorized in both dimensions (see GaussianKernel for example).
    - x: numpy array of shape (n, d), first set of samples.
    - y: numpy array of shape (m, d), second set of samples.

    Returns:
    - K: numpy array of shape (n, m), where K[i, j] = kernel(x[i], y[j]).
    """
    assert x.shape[1] == y.shape[1], f"Dimension mismatch: {x.shape[1]} != {y.shape[1]}"
    x = np.expand_dims(x, axis=1)
    y = np.expand_dims(y, axis=0)
    return kernel(x,y)


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
    Computes the MMD between two sets of samples X and Y. This function is optimized for small datasets.

    See @compute_mmd_weighted for the parameters and return value.
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
    MAX_SMALL_SIZE = 1000*1000
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