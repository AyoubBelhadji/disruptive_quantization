#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 12/02/2024 13:13:00
@author: dannys4
"""

import numpy as np
import numba as nb

from algorithms.IKBQ.iterative_kernel_based_quantization import IterativeKernelBasedQuantization
from tools.utils import proj_simplex, kernel_avg, broadcast_kernel

def grad_MMD_geom(prev_w, prev_nodes, new_nodes, data_array, kernel, MMD_regularization):
    # 2*grad_b eta*MMD_XY + MMD_YYbar
    # = grad_b [(1+eta)(b'*KYY*b) - 2*eta*b'*KYX.mean(axis=1) - 2*b'*KYY_bar*b_bar + eta*KXX.mean() + 2*b_bar'*KYY_bar*b_bar]
    # = (1+eta)2*KYY*b - 2*eta*KYX.mean(axis=1) - 2*KYY_bar*b_bar
    KYY = broadcast_kernel(kernel, new_nodes, new_nodes) #kernel(new_nodes, new_nodes)
    KYY_bar = broadcast_kernel(kernel, new_nodes, prev_nodes)
    KYX_mean = kernel_avg(kernel, new_nodes, data_array)
    ret = (1 + MMD_regularization)*KYY.dot(prev_w) - KYX_mean - MMD_regularization*KYY_bar.dot(prev_w)
    return ret

# @nb.jit()
# def mmd_W2_grad_step(y_t, w_t, kernel_grad2, data_array, step_sz):
    # For each node X_i, computes E_pi[grad_2 K(Y, X_i)] - mean(w_j * grad_2 K(X_j, X_i))
    # Where pi is the distribution of the data points Y = data_array
    # And K is the kernel function
    # And grad K is the gradient of the kernel function, which we assume is grad_2 k(Y,X) = X*k(Y, X)
    # M = len(w_t)
    # for i in range(M):
    #     y_dot_i  = w_t.dot(kernel_grad2(y_t, y_t[i]))
    #     y_dot_i -= np.sum(kernel_grad2(data_array, y_t[i]), axis=0)/len(data_array)
    #     y_dot_i *= -w_t[i]
    #     y_t[i] += step_sz * y_dot_i

# Transpose of above function
@nb.jit(parallel=True)
def mmd_W2_grad_step(y_t, w_t, kernel_grad2, data_array, step_sz):
    M, N = len(w_t), len(data_array)
    dkxy = np.zeros_like(y_t) # (M, d)
    sqrt_N = np.sqrt(N)
    for x_idx in nb.prange(N):
        dkxy_idx = kernel_grad2(data_array[x_idx], y_t) # (M, d)
        dkxy += dkxy_idx/sqrt_N
    for y_idx in range(M):
        y_inc = w_t.dot(kernel_grad2(y_t, y_t[y_idx])) - dkxy[y_idx]/sqrt_N
        y_t[y_idx] -= step_sz * w_t[y_idx] * y_inc

class InteractionForceTransportFlow(IterativeKernelBasedQuantization):

    def __init__(self, params):
        super().__init__(params)
        self.name = "IFTFlow"
        self.step_size = params.get('step_size')
        self.weight_step_size = params.get('weight_step_size', self.step_size)
        self.weight_regularization = params.get('weight_regularization')
        self.use_WFR = params.get('use_WFR', False)
        self.project_simplex = params.get('project_simplex', True)
        self.params = params

        self.kernel_scheduler = params.get('kernel')
        self.y_workspace = np.empty((self.K, self.d))
        self.is_first_iteration = True

    def calculate_weights(self, y_t, t, w_t):
        if self.is_first_iteration:
            self.is_first_iteration = False
            return np.ones(len(y_t)) / self.K

        proposed_step, w_tp1 = None, None
        if self.use_WFR:
            v0 = kernel_avg(self.kernel, y_t, self.data_array)
            Ky = broadcast_kernel(self.kernel, y_t, y_t)
            neg_dF_dmu = v0 - Ky.dot(w_t)
            proposed_step = w_t * np.exp(self.weight_step_size * neg_dF_dmu)
        else:
            # Calculate gradient direction times step size
            c_tplus1 = self.y_workspace
            dF = grad_MMD_geom(w_t, y_t, c_tplus1, self.data_array, self.kernel, self.weight_regularization)

            # Update weights using one step of interior point method
            proposed_step = w_t - self.weight_step_size*dF
        if self.project_simplex:
            w_tp1 = proj_simplex(proposed_step)
        else:
            w_tp1 = proposed_step

        return w_tp1

    def calculate_centroids(self, y_t, t, w_t):
        self.kernel = self.kernel_scheduler.GetKernel()
        self.kernel_grad2 = self.kernel_scheduler.GetKernelGrad2()

        step_sz = self.step_size
        self.y_workspace[:] = y_t
        mmd_W2_grad_step(self.y_workspace, w_t, self.kernel_grad2, self.data_array, step_sz)

        return self.y_workspace

