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
    for x_idx in nb.prange(n):
        kxy = kernel(x[x_idx], y)  # (m,)
        v0 += kxy
        for y_idx in range(m):
            v1[y_idx] += kxy[y_idx] * x[x_idx]
    return v0, v1


@nb.jit(parallel=True)
def broadcast_kernel_bar_parallel_moments_x(kappa, kappa_bar, x, y):
    n, (m, d) = len(x), y.shape
    v0 = np.zeros((m,), dtype=np.float64)
    grad_v0 = np.zeros((m, d), dtype=np.float64)
    for x_idx in nb.prange(n):
        kxy = kappa(x[x_idx], y)  # (m,)
        v0 += kxy
        kxy_bar = kappa_bar(x[x_idx], y)  # (m,)
        for y_idx in range(m):
            grad_v0[y_idx] += kxy_bar[y_idx] * (x[x_idx] - y[y_idx])
    return v0/n, grad_v0/n


def rhs_calculation(kernel, centroids, data):
    if False: #kernel.kernel_bar_is_scaled_kernel:
        v0, v1 = broadcast_kernel_parallel_moments_x(kernel.kernel, data, centroids)
    else:
        v0, v1 = broadcast_kernel_bar_parallel_moments_x(
            kernel.kernel, kernel.kernel_bar, data, centroids
        )
    return v0, v1

@nb.jit()
def msip_mapping(kernel, y_array, v0, v1_hat):
    K_matrix = broadcast_kernel(kernel.kernel, y_array, y_array)
    wts_msip = np.linalg.solve(K_matrix, v0)
    K_bar_matrix = broadcast_kernel(kernel.kernel_bar, y_array, y_array)
    K_bar_w = K_bar_matrix @ wts_msip
    for y_idx in range(len(y_array)):
        v1_hat[y_idx] += K_bar_w[y_idx] * y_array[y_idx]
    LHS_matrix = K_bar_matrix #if use_k_bar else K_matrix
    pts_msip = np.linalg.solve(LHS_matrix, v1_hat) / wts_msip[:, np.newaxis]
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

        v0, v1_bar = rhs_calculation(kernel, c_array, x_array)

        pts_msip = msip_mapping(kernel, c_array, v0, v1_bar)

        c_tplus1_array = (1 - self.dilation) * c_array + self.dilation * pts_msip

        return c_tplus1_array
