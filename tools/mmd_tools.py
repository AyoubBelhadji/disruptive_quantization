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
