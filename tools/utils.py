import numpy as np
import numba as nb

def proj_simplex(v):
    r"""Compute the closest point (orthogonal projection) on the
    generalized `(n-1)`-simplex of a vector :math:`\mathbf{v}` wrt. to the Euclidean
    distance. Adapted from Python OT library POT (MIT Licensed), source here:

    https://github.com/PythonOT/POT/blob/1761d0b91fc7551bcb9df15e54f11654b68d844c/ot/utils.py#L86
    """
    assert v.ndim == 1, "Input vector must be 1D"
    n = v.shape[0]
    # sort u in ascending order
    u = np.sort(v)
    # take the descending order
    u = np.flip(u)
    cssv = np.cumsum(u) - 1
    ind = np.arange(n, dtype=v.dtype) + 1
    cond = (u - (cssv / ind)) > 0
    rho = np.sum(cond)
    theta = cssv[rho - 1] / rho
    w = np.maximum(v - theta, np.zeros(v.shape, dtype=v.dtype))
    return w


# Adjugate matrix
def cofactor_matrix(matrix):
    """
    Calculate the cofactor matrix of a given square matrix.
    """
    n = matrix.shape[0]
    cofactor = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            # Minor of element (i, j)
            minor = np.delete(np.delete(matrix, i, axis=0), j, axis=1)
            # Cofactor is the determinant of the minor, times (-1)^(i+j)
            cofactor[i, j] = ((-1) ** (i + j)) * np.linalg.det(minor)

    return cofactor


def adjugate_matrix(matrix):
    """
    Calculate the adjugate (adjoint) of a given square matrix.
    """
    cofactor = cofactor_matrix(matrix)
    adjugate = cofactor.T  # Adjugate is the transpose of the cofactor matrix
    return adjugate

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

@nb.jit(parallel=True)
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
        x_expand = np.expand_dims(x, axis=1)
        y_expand = np.expand_dims(y, axis=0)
        return kernel(x_expand,y_expand)
    # Perform parallelized kernel computation
    transpose = False
    n, m = len(x), len(y)
    # make sure x is larger
    if n < m:
        x, y = y, x
        n, m = m, n
        transpose = True
    K = broadcast_kernel_parallel(kernel, x, y)
    if transpose:
        K = K.T
    return K
