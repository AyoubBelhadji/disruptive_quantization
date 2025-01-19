from algorithms.IKBQ import IterativeKernelBasedQuantization
import numpy as np
from tools.utils import broadcast_kernel

import numba as nb


@nb.jit(parallel=True, fastmath=True)
def kernel_avg(kernel, y, x):
    n, m = len(x), len(y)
    v0 = np.zeros((m,))
    for i in nb.prange(n):
        kxy = kernel(x[i], y)
        v0 += kxy
    return v0 / n


@nb.jit(parallel=True)
def broadcast_kernel_parallel_moments_x(kernel, x, y):
    n, (m, d) = len(x), y.shape
    v0 = np.zeros((m,), dtype=np.float64)
    v1 = np.zeros((m, d), dtype=np.float64)
    for i in nb.prange(n):
        kxy = kernel(x[i], y)  # (m,)
        v0 += kxy
        v1 += np.outer(kxy, x[i])
    return v0, v1


@nb.jit(parallel=True)
def broadcast_kernel_bar_parallel_moments_x(kappa, kappa_bar, x, y):
    n, (m, d) = len(x), y.shape
    v0 = np.zeros((m,), dtype=np.float64)
    v1 = np.zeros((m, d), dtype=np.float64)
    for i in nb.prange(n):
        kxy = kappa(x[i], y)  # (m,)
        v0 += kxy
        kxy_bar = kappa_bar(x[i], y)  # (m,)
        diff = x[i] - y  # (m,d)
        for i_prime in range(m):
            v1[i_prime] += kxy_bar[i_prime] * diff[i_prime]
    return v0, v1


def fast_ms_log_kde(centroids, data, kernel):
    if kernel.kernel_bar_is_scaled_kernel:
        v0, v1 = broadcast_kernel_parallel_moments_x(kernel.kernel, data, centroids)
    else:
        v0, v1 = broadcast_kernel_bar_parallel_moments_x(
            kernel.kernel, kernel.kernel_bar, data, centroids
        )
    return v0, v1


def get_kernel_matrix(kernel, centroids):
    if kernel.kernel_bar_is_scaled_kernel:
        return broadcast_kernel(kernel.kernel, centroids, centroids)
    else:
        return broadcast_kernel(kernel.kernel_bar, centroids, centroids)

@nb.jit()
def msip_mapping(kernel_bar_is_scaled_kernel, c_array, K_matrix, K_bar_matrix, v0, v1_bar):
    wts_msip = np.linalg.solve(K_matrix, v0)
    K_bar_w = K_bar_matrix @ wts_msip
    if kernel_bar_is_scaled_kernel:
        v1_hat = v1_bar
    else:
        v1_hat = v1_bar + c_array * K_bar_w[:, np.newaxis]
    pts_msip = np.linalg.solve(K_bar_matrix, v1_hat) / wts_msip[:, np.newaxis]
    return pts_msip


class CorrectedMultipleMeanShift(IterativeKernelBasedQuantization):
    def __init__(self, params):
        super().__init__(params)
        self.algo_name = "Corrected MMS"
        self.dilation = params.get("dilation", 1.0)

    def calculate_weights(self, c_array, t, w_array):
        x_array = self.data_array

        # Be careful because K means kernel matrix and number of centroids
        kernel = self.kernel_scheduler.GetKernel()

        K_matrix = broadcast_kernel(kernel, c_array, c_array)
        mu_array = kernel_avg(kernel, c_array, x_array)

        weights_array = np.linalg.solve(K_matrix, mu_array)

        return weights_array

    def calculate_centroids(self, c_array, t, w_array):
        x_array = self.data_array

        # Get the kernel and prekernel functions
        kernel = self.kernel_scheduler.GetKernelInstance()

        K_matrix = broadcast_kernel(kernel.kernel, c_array, c_array)
        K_bar_matrix = broadcast_kernel(kernel.kernel_bar, c_array, c_array)
        v0, v1_bar = fast_ms_log_kde(c_array, x_array, kernel)

        pts_msip = msip_mapping(kernel.kernel_bar_is_scaled_kernel, c_array, K_matrix, K_bar_matrix, v0, v1_bar)

        c_tplus1_array = (1 - self.dilation) * c_array + self.dilation * pts_msip

        return c_tplus1_array
