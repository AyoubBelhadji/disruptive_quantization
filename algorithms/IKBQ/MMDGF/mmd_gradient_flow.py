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
        self.ODE_solver_str = params.get("ODE_solver", "RK45")
        self.W2_ODE_solver = (
            self.Solve_W2_ODE_Scipy
            if self.ODE_solver_str != "Euler"
            else self.Solve_W2_ODE_Euler
        )
        self.time_parameterization = params.get("time_parameterization")
        self.time_parameterization.SetLength(self.T)
        self.params = params

        self.kernel_scheduler = params.get("kernel")
        self.y_workspace = np.empty((self.K, self.d))
        self.w_workspace = np.ones((self.K,))/self.K
        self.ydot_workspace = np.empty((self.K, self.d))
        self.diff_workspace = self.ydot_workspace.reshape(-1)

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
        self.W2_ODE_solver(tspan, y_t, y_tplus1)

    def Solve_W2_ODE_Scipy(self, tspan, y_t, y_tplus1):
        """
        Solve the Wasserstein mean-field ODE over the given time span
        """
        state_1 = solve_ivp(
            self.WFR_ODE,
            tspan,
            y_t.reshape(-1),
            t_eval=(tspan[1],),
            method=self.ODE_solver_str,
        )
        if len(state_1.y) == 0:
            raise ValueError(f"ODE solver failed to converge for {tspan}, {state_1.y}")
        y_tplus1[:] = state_1.y.reshape((self.K, self.d))

    def Solve_W2_ODE_Euler(self, tspan, y_t, y_tplus1):
        """
        Solve the Wasserstein mean-field ODE over the given time span
        """
        t0, t1 = tspan
        num_steps = self.steps_per_iteration
        dt = (t1 - t0) / num_steps
        y_tplus1[:] = y_t[:]
        y_tplus1_vec = y_tplus1.reshape(-1)
        for t in np.linspace(t0, t1, num_steps, endpoint=False):
            diff = self.WFR_ODE(t, y_tplus1_vec.reshape(-1))
            y_tplus1_vec[:] += dt * diff

    def WFR_ODE(self, _, y_t_vec):
        """
        RHS for the Wasserstein mean-field ODE
        """
        y_t = y_t_vec.reshape((self.K, self.d))
        W2_ODE_diff(y_t, self.kernel_grad2, self.data_array, self.ydot_workspace)
        return self.ydot_workspace.reshape(-1)
