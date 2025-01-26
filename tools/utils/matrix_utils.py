import numpy as np
import numba as nb

@nb.jit()
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
