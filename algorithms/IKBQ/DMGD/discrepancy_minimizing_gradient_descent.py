#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from algorithms.IKBQ.iterative_kernel_based_quantization import IterativeKernelBasedQuantization
from tools.utils import kernel_avg, kernel_grady_avg, broadcast_kernel

def calculate_v1_hat(kernel, w, y, avg_pts):
    v1_hat = kernel_grady_avg(kernel.kernel_grad2, y, avg_pts)
    for i in range(len(y)):
        v1_hat[i] += w.dot(kernel.kernel_bar(y, y[i])) * y[i]
    return v1_hat

class DiscrepancyMinimizingGradientDescent(IterativeKernelBasedQuantization):

    def __init__(self, params):
        super().__init__(params)
        self.name = "DMGD"
        self.step_size = params.get('step_size')
        self.params = params

        self.kernel_scheduler = params.get('kernel')
        self.y_workspace = np.empty((self.K, self.d))
        self.w_workspace = np.empty((self.K,))

        # Workspace for v0, KDE of zeroth moment
        self.v0_workspace = np.empty((self.K,))
        # Workspace for v1, KDE of first moment
        self.v1_workspace = np.empty((self.K, self.d))
        # Workspace for the kernel matrix
        self.kernel_workspace = np.empty((self.K, self.K))

    def calculate_weights(self, y_t, t, w_t):
        # Create kernel matrices with the current centroids
        self.calculate_weights_internal(y_t)
        return self.w_workspace

    def calculate_centroids(self, y_t, t, w_t):
        eta = self.step_size
        kernel = self.kernel_scheduler.GetKernelInstance()
        self.calculate_weights_internal(y_t)
        KbarW = broadcast_kernel(kernel.kernel_bar, y_t, y_t) * self.w_workspace
        v1_hat = calculate_v1_hat(kernel, self.w_workspace, y_t, self.data_array)
        self.y_workspace[:] = y_t - eta * (self.w_workspace[:,np.newaxis] * (KbarW.dot(y_t) - v1_hat))
        return self.y_workspace

    def calculate_weights_internal(self, y_t):
        kernel = self.kernel_scheduler.GetKernel()
        v0 = kernel_avg(kernel, y_t, self.data_array)
        Ky = broadcast_kernel(kernel, y_t, y_t)
        self.w_workspace[:] = np.linalg.solve(Ky, v0)