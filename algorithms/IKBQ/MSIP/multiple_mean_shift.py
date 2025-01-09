#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 25/25/2525 25:25:25
Also created on Sun Dec 10 12:26:17 2023
Also also created on Mon Nov 18 6:22:10 2024
@author: ayoubbelhadji
"""


# from .sub_algorithm import SubAlgorithm
from algorithms.IKBQ.iterative_kernel_based_quantization import IterativeKernelBasedQuantization
import tools.mmd_tools as mmd_tools
import numpy as np
import numba as nb
from tools.utils import adjugate_matrix

# @nb.jit(cache=True)
def stable_ms_map(centroids, data, pre_kernel):
    """
    Compute the stable mean shift map using pre-kernel values and array operations.

    Args:
        x: The input scalar.
        n_array: A NumPy array of scalars (e.g., n_list).
        pre_kernel: A function that computes the pre-kernel values.

    Returns:
        The updated position after applying the mean shift map.
    """

    # Compute pre-kernel values as a NumPy array
    # Assume pre_kernel can handle array input
    pre_kernel_array = mmd_tools.broadcast_kernel(pre_kernel, data, centroids) # (N, M)

    pre_kernel_offset = np.max(pre_kernel_array, axis=0) # (M,)

    # Compute the weights (using broadcasting)
    weights = np.exp(pre_kernel_array - pre_kernel_offset) # (N, M)

    # Compute the weighted sum of the differences
    a = weights.T.dot(data) # (M, d)
    b = np.sum(weights, axis=0) # (M,)

    # Return the mean shift map value
    return a / b[:, np.newaxis] # (M, d)

# @nb.jit(cache=True)
def stable_log_kde(centroids, data, pre_kernel):
    """
    Compute the stable mean shift map using pre-kernel values and array operations.

    Args:
        x: The input scalar.
        n_array: A NumPy array of scalars (e.g., n_list).
        pre_kernel: A function that computes the pre-kernel values.

    Returns:
        The updated position after applying the mean shift map.
    """

    # Compute pre-kernel values as a NumPy array
    # Assume pre_kernel can handle array input
    pre_kernel_array = mmd_tools.broadcast_kernel(pre_kernel, data, centroids) # (N, M)
    pre_kernel_offset = np.max(pre_kernel_array, axis=0) # (M,)

    # Compute the weights (using broadcasting)
    weights = np.exp(pre_kernel_array - pre_kernel_offset) # (N, M)

    # Compute the weighted sum of the differences
    b = np.sum(weights, axis=0) # (M,)

    # Return the mean shift map value
    return pre_kernel_offset+np.log(b) # (M,)


# @nb.jit(cache=True)
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
    log_w_max = np.max(log_w)

    log_w_adjusted = log_w - log_w_max

    w_adjusted = np.exp(log_w_adjusted)

    KW = inverse_kernel_mat*w_adjusted

    a = KW @ kde_means
    b = np.sum(KW, axis=1)

    # Return the ratio
    ret = a / b[:,np.newaxis]

    for m in range(inverse_kernel_mat.shape[0]):
        nnzeros = np.where(np.abs(inverse_kernel_mat[m]) > 0.0)[0]
        if len(nnzeros) == inverse_kernel_mat.shape[1]-1: # I don't understand this branch of logic?
            zeros = np.where(np.abs(inverse_kernel_mat[m]) == 0.0)[0]
            ret[m] = kde_means[zeros[0], :]
        elif len(nnzeros) == 1:
            ret[m] = kde_means[nnzeros[0], :]

    return ret
class MultipleMeanShift(IterativeKernelBasedQuantization):

    def __init__(self, params):
        super().__init__(params)
        self.algo_name = 'Vanilla MMS'
        self.reg_K = params.get('reg_K', 1e-5)
        self.dilation = params.get('dilation',1.0)
        use_adjugate = params.get('use_adjugate', True)
        self.inv_K_fcn = adjugate_matrix if use_adjugate else np.linalg.inv

    def calculate_weights(self, c_array, t, w_array):
        x_array = self.data_array

        # Be careful because K means kernel matrix and number of centroids
        kernel = self.kernel_scheduler.GetKernel()

        K_matrix = mmd_tools.broadcast_kernel(kernel, c_array, c_array)
        mu_array = mmd_tools.broadcast_kernel(kernel, c_array, x_array).mean(axis=1)

        weights_array = np.linalg.solve(K_matrix, mu_array)

        return weights_array

    def calculate_centroids(self, c_array, t, w_array):
        x_array = self.data_array

        # Get the kernel and prekernel functions
        kernel = self.kernel_scheduler.GetKernel()
        pre_kernel = self.kernel_scheduler.GetPreKernel()

        K_matrix = mmd_tools.broadcast_kernel(kernel, c_array, c_array)

        K_inv_matrix = self.inv_K_fcn(K_matrix)
        ms_array = stable_ms_map(c_array, x_array, pre_kernel)
        log_v_0_array = stable_log_kde(c_array, x_array, pre_kernel)

        c_tplus1_array = (1-self.dilation)*c_array + self.dilation*average_x_v(
            K_inv_matrix, log_v_0_array, ms_array)

        return c_tplus1_array

