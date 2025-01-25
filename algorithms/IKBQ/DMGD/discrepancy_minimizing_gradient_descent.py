#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 12/02/2024 13:13:00
@author: dannys4
"""

import numpy as np

from algorithms.IKBQ.iterative_kernel_based_quantization import IterativeKernelBasedQuantization
from tools.utils import proj_simplex

class DiscrepancyMinimizingGradientDescent(IterativeKernelBasedQuantization):

    def __init__(self, params):
        super().__init__(params)
        self.name = "DMGD"
        self.centroid_step_size = params.get('step_size')
        self.params = params

        self.kernel_scheduler = params.get('kernel')
        self.c_workspace = np.empty((self.K, self.d))
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
        eta = self.centroid_step_size
        # Make sure the weights are calculated
        if t == 0:
            self.calculate_weights_internal(y_t)
            w_t = self.w_workspace
        self.kernel_workspace[:] *= w_t
        self.c_workspace[:] = y_t - eta * w_t[:,np.newaxis] * (self.kernel_workspace.dot(y_t) - self.v1_workspace)
        return self.c_workspace

    def calculate_weights_internal(self, y_t):
        self.kernel = self.kernel_scheduler.GetKernel()
        for i in range(self.K):
            self.kernel_workspace[i] = self.kernel(y_t, y_t[i]).flatten()
            Kx = self.kernel(self.data_array, y_t[i]).flatten()
            self.v0_workspace[i] = Kx.mean()
            self.v1_workspace[i] = Kx.dot(self.data_array) / self.data_array.shape[0]
        self.w_workspace[:] = np.linalg.solve(self.kernel_workspace, self.v0_workspace)