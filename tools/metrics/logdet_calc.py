import numba as nb
import numpy as np
from tools.utils import reshape_wrapper, broadcast_kernel

@nb.jit(parallel=True)
def compute_all_kernel_logdets(all_nodes, kernel):
    """
    Compute the log-determinant of the kernel matrix for each set of nodes.

    Parameters:
    - all_nodes: numpy array of shape (..., m, d), set of samples.
    - kernel: a kernel function object with a `kernel` method.

    Returns:
    - logdets: numpy array of shape (...), where logdets[i] is the log-determinant of the kernel matrix for all_nodes[i].
    """
    logdets = np.zeros(len(all_nodes))
    for i in nb.prange(len(all_nodes)):
        Y = all_nodes[i]
        K = broadcast_kernel(kernel, Y, Y)
        logdets[i] = np.linalg.slogdet(K)[1]
    return logdets

@nb.jit()
def logdet(Y, kernel):
    """
    Compute the log-determinant of the kernel matrix for a set of nodes.

    Parameters:
    - Y: numpy array of shape (m, d), set of samples.
    - kernel: a kernel function object with a `kernel` method.

    Returns:
    - logdet: the log-determinant of the kernel matrix for Y.
    """
    K = broadcast_kernel(kernel, Y, Y)
    return np.linalg.slogdet(K)[1]

def logdet_array(all_nodes, kernel_obj):
    kernel_eval = kernel_obj.kernel
    logdets = reshape_wrapper(compute_all_kernel_logdets, all_nodes, None, kernel_eval)
    return logdets
