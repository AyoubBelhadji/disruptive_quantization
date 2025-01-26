#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 11/26/2024 12:49:00
@author: dannys4
"""

import numpy as np
import numba as nb
import numbers
from tools.utils import kernel_avg, kernel_grad2_avg
from tools.ode import GradientFlowIntegrator

from algorithms.IKBQ.iterative_kernel_based_quantization import IterativeKernelBasedQuantization

@nb.jit()
def WFR_ODE_centroid_diff(y_t, w_t, kernel_grad2, data_array, y_dot):
    M = len(y_t)
    v1 = kernel_grad2_avg(kernel_grad2, y_t, data_array)
    for i in range(M):
        y_dot[i] = v1[i]
        y_dot[i] -= w_t.dot(kernel_grad2(y_t, y_t[i]))
    return y_dot

@nb.jit(parallel=True)
def WFR_ODE_weight_diff_NPMLE(y_t, w_t, kernel, data_array, w_dot):
    N = len(data_array)
    for ell in nb.prange(N):
        k_ell = kernel(data_array[ell], y_t)
        denom = k_ell.dot(w_t)
        w_dot += k_ell/denom
    for i in range(len(w_dot)):
        w_dot[i] = (w_dot[i]/N - 1) * w_t[i]

def WFR_ODE_weight_diff(y_t, w_t, kernel, data_array, w_dot):
    K = len(w_t)
    kde_mass = kernel_avg(kernel, y_t, data_array)
    for i in range(K):
        w_dot[i]  = w_t.dot(kernel(y_t, y_t[i]))
        w_dot[i] -= kde_mass[i]
    # Fisher-Rao adjustment
    w_dot[:] *= -w_t

class WassersteinFisherRao(IterativeKernelBasedQuantization):

    def __init__(self, params):
        super().__init__(params)
        self.algo_name = "WFR"
        self.time_parameterization = params.get('time_parameterization')
        self.time_parameterization.SetLength(self.T)
        self.point_accelerator = params.get('point_accelerator', 1.0)

        # Create the ODE solver function
        self.ODE_solver_str = params.get('ODE_solver', 'RK45')
        self.integrator = GradientFlowIntegrator(self.ODE_solver_str)
        self.steps_per_iteration = params.get('steps_per_iteration', 1)

        # Create workspaces for the centroids and weights
        # Front-facing workspaces
        self.y_workspace = np.empty((self.K, self.d))
        self.w_workspace = np.ones((self.K,))/self.K

        # ODE workspaces
        self.state_0 = np.empty(self.y_workspace.size + self.w_workspace.size)
        diff_workspace = np.empty_like(self.state_0)
        self.diff_workspace = diff_workspace
        self.ydot_workspace = diff_workspace[:-self.K].reshape((self.K, self.d))
        self.wdot_workspace = diff_workspace[-self.K:]
        self.params = params

    def calculate_weights(self, *_):
        return self.w_workspace

    def calculate_centroids(self, y_array, t, w_array):
        self.WFRStep(t, y_array, w_array, self.y_workspace, self.w_workspace)
        return self.y_workspace

    def WFRStep(self, t, y_t, w_t, y_tplus1, w_tplus1):
        """
        Compute the next step in the Wasserstein Fisher-Rao mean-field ODE
        """
        tspan = self.time_parameterization(t)
        self.kernel = self.kernel_scheduler.GetKernel()
        self.kernel_grad2 = self.kernel_scheduler.GetKernelGrad2()

        # Accelerate the points ODE
        self.point_acceleration = self.point_accelerator
        if self.point_acceleration == 'bandwidth':
            self.point_acceleration = self.kernel_scheduler.get_bandwidth()**2
        if not isinstance(self.point_acceleration, numbers.Number):
            raise ValueError(f"WFR: Invalid point_accelerator {self.point_acceleration}, type {type(self.point_acceleration)}")

        # Concatenate the centroids and weights into a single initial condition
        self.state_0[:-self.K] = y_t.reshape(-1)
        self.state_0[-self.K:] = w_t

        # Solve the ODE
        self.integrator(self.state_0, self.WFR_ODE, tspan[1]-tspan[0], self.steps_per_iteration)

        # Copy output into the output arrays
        y_tplus1[:] = self.state_0[:-self.K].reshape((self.K, self.d))
        w_tplus1[:] = self.state_0[-self.K:].reshape(-1)

    def WFR_ODE(self, _, state_t):
        """
        RHS for the Wasserstein Fisher-Rao mean-field ODE
        """
        y_t_vec = state_t[:-self.K]
        y_t = y_t_vec.reshape((self.K, self.d))
        w_t = state_t[-self.K:]
        WFR_ODE_centroid_diff(y_t, w_t, self.kernel_grad2, self.data_array, self.ydot_workspace)
        if self.point_accelerator != 1.0:
            self.ydot_workspace[:] *= self.point_acceleration
        WFR_ODE_weight_diff(  y_t, w_t, self.kernel, self.data_array, self.wdot_workspace)
        return self.diff_workspace