#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 25/25/2525 25:25:25
Also created on Sun Dec 10 12:26:17 2023
Also also created on Mon Nov 18 6:22:10 2024
@author: ayoubbelhadji
"""

# from .sub_algorithm import SubAlgorithm
from algorithms.IKBQ import IterativeKernelBasedQuantization
import numpy as np
from tools.utils import adjugate_matrix, broadcast_kernel, kernel_avg

import numba as nb


@nb.jit(parallel=True, fastmath=True)
def nb_max_axis0(arr):
    """
    Compute the maximum value along axis 0 of a NumPy array.

    Args:
        arr: A NumPy array.

    Returns:
        The maximum value along axis 0.
    """
    ret = np.zeros(arr.shape[1])
    for i in nb.prange(ret.size):
        ret[i] = np.max(arr[:, i])
    return ret


@nb.jit()
def stable_ms_log_kde_jit(centroids, data, log_kernel):
    """
    Compute the stable mean shift map using pre-kernel values and array operations.

    Args:
        x: The input scalar.
        n_array: A NumPy array of scalars (e.g., n_list).
        log_kernel: A function that computes the pre-kernel values.

    Returns:
        The updated position after applying the mean shift map.
    """

    # Compute pre-kernel values as a NumPy array
    # Assume log_kernel can handle array input
    log_kernel_array = broadcast_kernel(log_kernel, data, centroids)  # (N, M)
    log_kernel_offset = nb_max_axis0(log_kernel_array)  # (M,)

    # Compute the weights (using broadcasting)
    kernel_evals = np.exp(log_kernel_array - log_kernel_offset)  # (N, M)

    # Compute the weighted sum of the differences
    a = kernel_evals.T.dot(data)  # (M, d)
    b = np.sum(kernel_evals, axis=0)  # (M,)

    stable_ms = np.empty_like(a)
    for i in range(a.shape[0]):
        stable_ms[i] = a[i] / b[i]  # (M, d)

    log_kde = log_kernel_offset + np.log(b)  # (M,)

    # Return the mean shift map value
    return stable_ms, log_kde


def stable_ms_log_kde(centroids, data, kernel):
    return stable_ms_log_kde_jit(centroids, data, kernel.log_kernel)


@nb.jit()
def average_x_v(inverse_kernel_mat, log_w, kde_means):
    """
    Compute the weighted mean of vectors using the Log-Sum-Exp trick for stability.
    Handles zero or near-zero weights gracefully.

    Args:
        inverse_kernel_mat: An array of scalars (weights), shape (M,M).
        log_w: An array of log positive scalars (weights), shape (M,).
        kde_means: An array of vectors (M, D), where each row is a vector v_i of dimension D.
        epsilon: Small value to handle near-zero weights.

    Returns:
        A vector representing the weighted mean.
    """
    M = inverse_kernel_mat.shape[0]
    log_w_max = np.max(log_w)

    log_w_adjusted = log_w - log_w_max

    w_adjusted = np.exp(log_w_adjusted)

    KW = inverse_kernel_mat * w_adjusted

    a = KW @ kde_means
    b = np.sum(KW, axis=1)

    # Return the ratio
    valid_b_idxs = b != 0.0
    ret = np.empty_like(a)
    ret[valid_b_idxs] = np.divide(a[valid_b_idxs], b[valid_b_idxs][:, np.newaxis])

    for m in range(M):
        nnzeros = np.where(np.abs(inverse_kernel_mat[m]) > 0.0)[0]
        if len(nnzeros) == M - 1:  # I don't understand this branch of logic?
            zeros = np.where(np.abs(inverse_kernel_mat[m]) == 0.0)[0]
            ret[m] = kde_means[zeros[0], :]
        elif len(nnzeros) == 1:
            ret[m] = kde_means[nnzeros[0], :]

    return ret

def get_kernel_matrix(kernel, centroids):
    return broadcast_kernel(kernel.kernel, centroids, centroids)

class MultipleMeanShift(IterativeKernelBasedQuantization):
    def __init__(self, params):
        super().__init__(params)
        self.algo_name = "Vanilla MMS"
        self.reg_K = params.get("reg_K", 1e-5)
        self.dilation = params.get("dilation", 1.0)
        use_adjugate = params.get("use_adjugate", True)
        self.inv_K_fcn = adjugate_matrix if use_adjugate else np.linalg.inv

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

        K_inv_matrix = self.inv_K_fcn(K_matrix)
        ms_array, log_v_0_array = stable_ms_log_kde(c_array, x_array, kernel)

        c_tplus1_array = (1 - self.dilation) * c_array + self.dilation * average_x_v(
            K_inv_matrix, log_v_0_array, ms_array
        )

        return c_tplus1_array
