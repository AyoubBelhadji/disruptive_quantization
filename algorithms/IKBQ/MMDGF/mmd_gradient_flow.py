#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 12/02/2024 13:13:00
@author: dannys4
"""

import numpy as np
import numba as nb
from scipy.integrate import solve_ivp
from tools.utils import kernel_grad2_avg
from tools.ode import GradientFlowIntegrator

from algorithms.IKBQ.iterative_kernel_based_quantization import (
    IterativeKernelBasedQuantization,
)


@nb.jit()
def W2_ODE_diff(y_t, kernel_grad2, data_array, y_dot):
    M = len(y_t)
    v1 = kernel_grad2_avg(kernel_grad2, y_t, data_array)
    for i in range(M):
        y_dot[i] = v1[i]
        y_dot[i] -= np.sum(kernel_grad2(y_t, y_t[i]), axis=0) / M
    return y_dot


class MmdGradientFlow(IterativeKernelBasedQuantization):
    def __init__(self, params):
        super().__init__(params)
        self.name = "MMDGF"
        self.step_size = params.get("step_size")
        ODE_solver_str = params.get("ODE_solver", "RK45")
        self.integrator = GradientFlowIntegrator(ODE_solver_str)
        self.time_parameterization = params.get("time_parameterization")
        self.time_parameterization.SetLength(self.T)
        self.params = params
        self.steps_per_iteration = params.get("steps_per_iteration", 1)

        self.kernel_scheduler = params.get("kernel")
        self.y_workspace = np.empty((self.K, self.d))
        self.w_workspace = np.ones((self.K,))/self.K
        self.ydot_workspace = np.empty((self.K, self.d))

    def calculate_weights(self, *_):
        return self.w_workspace

    def calculate_centroids(self, y_array, t, _):
        self.W2Step(t, y_array, self.y_workspace)
        return self.y_workspace

    def W2Step(self, t, y_t, y_tplus1):
        """
        Compute the next step in the Wasserstein mean-field ODE
        """
        tspan = self.time_parameterization(t)
        self.kernel_grad2 = self.kernel_scheduler.GetKernelGrad2()

        # Solve the ODE
        y_tplus1[:] = y_t[:]
        # self.W2_ODE_solver(tspan, y_t, y_tplus1)
        self.integrator(y_tplus1.reshape(-1), self.WFR_ODE, tspan[1]-tspan[0], self.steps_per_iteration)

    def WFR_ODE(self, _, y_t_vec):
        """
        RHS for the Wasserstein mean-field ODE
        """
        y_t = y_t_vec.reshape((self.K, self.d))
        W2_ODE_diff(y_t, self.kernel_grad2, self.data_array, self.ydot_workspace)
        return self.ydot_workspace.reshape(-1)
