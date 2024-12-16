#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 12/02/2024 13:13:00
@author: dannys4
"""

import numpy as np

from algorithms.IKBQ.iterative_kernel_based_quantization import IterativeKernelBasedQuantization
from tools.utils import proj_simplex

class InteractionForceTransportFlow(IterativeKernelBasedQuantization):

    def __init__(self, params):
        super().__init__(params)
        self.name = "IFTFlow"
        self.centroid_step_size = params.get('centroid_step_size')
        self.weight_step_size = params.get('weight_step_size')
        self.eta = params.get('eta', 1.)
        self.params = params

        self.kernel_scheduler = params.get('kernel')
        self.c_workspace = np.empty((self.K, self.d))

    def calculate_weights(self, c_t, t, w_t):
        # Calculate gradient direction
        gradient_dir = np.zeros_like(w_t)
        c_t_plus_1 = self.c_workspace
        for i in range(self.K):
            gradient_dir[i]  = -self.kernel(self.data_array, c_t_plus_1[i]).mean() * self.eta
            gradient_dir[i] += -w_t.dot(self.kernel(c_t_plus_1, c_t_plus_1[i]))

        # Update weights using one step of interior point method
        proposed_step = w_t - gradient_dir
        descent_dir = proj_simplex(proposed_step) - w_t

        tau_w = self.weight_step_size
        w_tp1 = w_t + tau_w * descent_dir
        if (w_tp1 < 0).any():
            w_tp1 = proj_simplex(w_tp1)
        return w_tp1

    def calculate_centroids(self, c_t, t, w_t):
        self.kernel = self.kernel_scheduler.GetKernel()
        self.kernel_grad2 = self.kernel_scheduler.GetKernelGrad2()

        tau_c = self.centroid_step_size
        for i in range(self.K):
            self.c_workspace[i,:] = c_t[i,:]
            # Gradient step for node i
            self.c_workspace[i] -= tau_c * self.kernel_grad2(self.data_array, c_t[i]).mean(axis=0)
            self.c_workspace[i] += tau_c * w_t.dot(self.kernel_grad2(c_t, c_t[i]))
        return self.c_workspace

