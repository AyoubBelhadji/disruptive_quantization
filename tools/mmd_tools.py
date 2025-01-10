import numpy as np
import numba as nb

@nb.jit()
def broadcast_kernel(kernel, x, y):
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
def compute_mmd_weighted(X, Y, kernel, weights_X=None, weights_Y=None):
    """
    Computes the MMD between two sets of samples X and Y, each of which might have weights.

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


@nb.jit(parallel=True)
def calculate_MMD_large_weighted_X(X, Y, kernel, weights_Y):
    """
    Computes the MMD between two sets of samples X and Y.

    Parameters:
    - X: numpy array of shape (n, d), first set of samples.
    - Y: numpy array of shape (m, d), second set of samples.
    - kernel: a kernel function object with a `kernel` method.
    - weights_X: numpy array of shape (n,), weights for samples in X.

    Returns:
    - mmd: the weighted MMD value.
    """
    d = X.shape[1]
    assert d == Y.shape[1], f"Dimension mismatch: {d} != {Y.shape[1]}"
    N, M = len(X), len(Y)
    assert M == len(weights_Y), f"Dimension mismatch: {M} != {len(weights_Y)}"
    x_wt = 1./N
    N_max = max(N,M)
    mmd_sq = 0
    for idx in nb.prange(N_max*N_max):
        i,j = idx // N_max, idx % N_max
        add_to_mmd = 0.
        # Add K_XX
        if i < N and j <= i:
            mul = 1 + (i != j)
            add_to_mmd += mul * kernel(X[i], X[j]) * x_wt * x_wt
        # Add K_YY
        if i < M and j <= i:
            y_wt_i, y_wt_j = weights_Y[i], weights_Y[j]
            mul = 1 + (i != j)
            add_to_mmd += mul * kernel(Y[i], Y[j]) * y_wt_i * y_wt_j
        # Subtract 2K_XY
        if i < N and j < M:
            y_wt_i = weights_Y[i]
            add_to_mmd -= 2 * kernel(X[i], Y[j]) * y_wt_i * x_wt
        mmd_sq += add_to_mmd
    return np.sqrt(mmd_sq)