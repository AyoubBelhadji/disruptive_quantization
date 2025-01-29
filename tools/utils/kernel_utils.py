import numpy as np
import numba as nb

def reshape_wrapper(function: callable, nodes: np.ndarray, node_weights: np.ndarray = None, *args):
    """
        Given array of data (...,M,d) and optional weights (...,M), flatten the first dimensions and apply `function`.

        Expects function f(nodes, node_weights, *args) or f(nodes, *args) if node_weights is None, where if argument has first dimension P, the output will have shape (P,).

        ----------
        function : callable
            The function to be applied to the reshaped nodes and optional node_weights. Signature f(nodes, node_weights, *args).
        nodes : numpy.ndarray
            The array of data with shape (..., M, d) to be reshaped and passed to the function.
        node_weights : numpy.ndarray, optional
            The array of weights with shape (..., M) to be reshaped and passed to the function. Default is None.
        *args : tuple
            Additional arguments to be passed to the function.

        Returns
        -------
        numpy.ndarray
            The result of the function applied to the reshaped nodes (and node_weights if provided), reshaped back to the dimensions (...).
    """
    prev_shape = nodes.shape
    # Flatten the first dimensions
    nodes = nodes.reshape(-1, prev_shape[-2], prev_shape[-1])
    if node_weights is not None:
        node_weights = node_weights.reshape(-1, prev_shape[-2])
        result = function(nodes, node_weights, *args)
    else:
        result = function(nodes, *args)
    return result.reshape(prev_shape[:-2])

@nb.jit()
def broadcast_kernel_serial(kernel, x, y):
    x_expand = np.expand_dims(x, axis=1)
    y_expand = np.expand_dims(y, axis=0)
    return kernel(x_expand,y_expand)

@nb.jit(parallel=True, fastmath=True)
def broadcast_kernel_parallel(kernel, x, y):
    n, m = len(x), len(y)
    K = np.zeros((n, m))
    for i in nb.prange(n):
        K[i] = kernel(x[i], y)
    return K

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
    MIN_PARALLEL = 1001
    assert x.shape[1] == y.shape[1], f"Dimension mismatch: {x.shape[1]} != {y.shape[1]}"
    if len(x) < MIN_PARALLEL and len(y) < MIN_PARALLEL:
        return broadcast_kernel_serial(kernel, x, y)
    # Perform parallelized kernel computation
    transpose = False
    n, m = len(x), len(y)
    # make sure x is larger
    if n < m:
        x, y = y, x
        n, m = m, n
        transpose = True
    # K = broadcast_kernel_parallel(kernel, x, y)
    K = broadcast_kernel_parallel(kernel, x, y)
    if transpose:
        K = K.T
    return K

@nb.jit(parallel=True)
def kernel_avg(kernel, y, avg_pts):
    n, m = len(avg_pts), len(y)
    v0 = np.zeros((m,))
    for i in nb.prange(n):
        kxy = kernel(avg_pts[i], y)
        v0 += kxy
    return v0 / n

@nb.jit(parallel=True)
def kernel_bar_moment(kernel_bar, y, avg_pts):
    n = len(avg_pts)
    v1 = np.zeros_like(y)
    for i in nb.prange(n):
        x = avg_pts[i]
        kxy = kernel_bar(avg_pts[i], y)
        v1 += np.outer(kxy, x)
    return v1 / n

@nb.jit(parallel=True)
def kernel_grady_avg(kernel_grad2, y, avg_pts):
    n = len(avg_pts)
    v1 = np.zeros_like(y)
    for i in nb.prange(n):
        kxy = kernel_grad2(avg_pts[i], y)
        v1 += kxy
    return v1 / n