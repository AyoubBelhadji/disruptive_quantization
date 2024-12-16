#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 11/26/2024 12:49:00
@author: dannys4
"""

import numpy as np
from scipy.integrate import solve_ivp

from algorithms.IKBQ.iterative_kernel_based_quantization import IterativeKernelBasedQuantization

class WassersteinFisherRao(IterativeKernelBasedQuantization):

    def __init__(self, params):
        super().__init__(params)
        self.algo_name = "WFR"
        self.time_parameterization = params.get('time_parameterization')
        self.time_parameterization.SetLength(self.T)

        # Create the ODE solver function
        self.ODE_solver_str = params.get('ODE_solver', 'RK45')
        if self.ODE_solver_str == 'Euler':
            self.WFR_ODE_solver = self.Solve_WFR_ODE_Euler
            self.steps_per_iteration = params.get('steps_per_iteration', 1)
        else:
            self.WFR_ODE_solver = self.Solve_WFR_ODE_Scipy

        # Create workspaces for the centroids and weights
        # Front-facing workspaces
        self.c_workspace = np.empty((self.K, self.d))
        self.w_workspace = np.empty((self.K,))

        # ODE workspaces
        self.y0 = np.empty(self.c_workspace.size + self.w_workspace.size)
        diff_workspace = np.empty_like(self.y0)
        self.diff_workspace = diff_workspace
        self.cdot_workspace = diff_workspace[:-self.K].reshape((self.K, self.d))
        self.wdot_workspace = diff_workspace[-self.K:]
        self.params = params

    def calculate_weights(self, *_):
        return self.w_workspace

    def calculate_centroids(self, c_array, t, w_array):
        self.WFRStep(t, c_array, w_array, self.c_workspace, self.w_workspace)
        return self.c_workspace

    def WFRStep(self, t, c_t, w_t, c_tplus1, w_tplus1):
        """
        Compute the next step in the Wasserstein Fisher-Rao mean-field ODE
        """
        tspan = self.time_parameterization(t)
        self.kernel = self.kernel_scheduler.GetKernel()
        self.kernel_grad2 = self.kernel_scheduler.GetKernelGrad2()

        # Concatenate the centroids and weights into a single initial condition
        self.y0[:-self.K] = c_t.reshape(-1)
        self.y0[-self.K:] = w_t

        # Solve the ODE
        y1 = self.WFR_ODE_solver(tspan)

        # Copy output into the output arrays
        c_tplus1[:] = y1[:-self.K].reshape((self.K, self.d))
        w_tplus1[:] = y1[-self.K:].reshape(-1)

    def Solve_WFR_ODE_Scipy(self, tspan):
        """
        Solve the Wasserstein Fisher-Rao mean-field ODE over the given time span
        """
        y1 = solve_ivp(self.WFR_ODE, tspan, self.y0, t_eval=(tspan[1],), method=self.ODE_solver_str)
        return y1.y

    def Solve_WFR_ODE_Euler(self, tspan):
        """
        Solve the Wasserstein Fisher-Rao mean-field ODE over the given time span
        """
        t0, t1 = tspan
        num_steps = self.steps_per_iteration
        dt = (t1-t0)/num_steps
        y1 = self.y0.copy()
        for t in np.linspace(t0, t1, num_steps, endpoint=False):
            self.WFR_ODE(t, y1)
            y1[:] += dt*self.diff_workspace
        return y1

    def WFR_ODE(self, _, y):
        """
        RHS for the Wasserstein Fisher-Rao mean-field ODE
        """
        c_t_vec = y[:-self.K]
        c_t = c_t_vec.reshape((self.K, self.d))
        w_t = y[-self.K:]
        self.WFR_ODE_centroid_diff(c_t, w_t)
        self.WFR_ODE_weight_diff(  c_t, w_t)
        return self.diff_workspace

    def WFR_ODE_centroid_diff(self, c_t, w_t):
        # For each node X_i, computes E_pi[grad_2 K(Y, X_i)] - mean(w_j * grad_2 K(X_j, X_i))
        # Where pi is the distribution of the data points Y = data_array
        # And K is the kernel function
        # And grad K is the gradient of the kernel function, which we assume is grad_2 k(Y,X) = X*k(Y, X)
        c_dot = self.cdot_workspace
        for i in range(self.K):
            c_dot[i] = -self.kernel_grad2(self.data_array, c_t[i]).mean(axis=0)
            c_dot[i] += w_t.dot(self.kernel_grad2(c_t, c_t[i]))
        c_dot[:] *= w_t[:, None]

    def WFR_ODE_weight_diff(self, c_t, w_t):
        w_dot = self.wdot_workspace
        for i in range(self.K):
            expected_xc = self.kernel(self.data_array, c_t[i]).mean()
            w_dot[i] = expected_xc
            w_dot[i] -= w_t.dot(self.kernel(c_t, c_t[i]))
        # Fisher-Rao adjustment
        w_dot[:] *= w_t

if __name__ == '__main__':
    from functions.kernels.gaussian_kernel import GaussianKernel
    T_ = 1
    D_ = 3
    K_ = 11
    kernel = GaussianKernel(1.0)
    class kernel_scheduler:
        def __init__(self):
            self.kernel = kernel
        def GetKernel(self):
            return self.kernel.kernel
        def GetKernelGrad2(self):
            return self.kernel.kernel_grad2
        def IncrementSchedule(self):
            pass

    timesteps = np.array([0.,1e-4])
    time_param = lambda _: (lambda t: timesteps[t:t+2])
    params = {
        'R': 1,
        'K': K_,
        'd': D_,
        'T': T_,
        'time_parameterization': time_param,
        'kernel': kernel_scheduler()
    }

    data_array = np.random.randn(K_, D_)
    c_0 = data_array.copy()
    w_0 = np.ones(K_)/K_

    wfr = WassersteinFisherRao(params)
    wfr.data_array = data_array
    wfr.kernel = kernel.kernel
    wfr.kernel_grad2 = kernel.kernel_grad2
    wfr.WFR_ODE_centroid_diff(c_0, w_0)
    wfr.WFR_ODE_weight_diff(c_0, w_0)
    assert np.linalg.norm(wfr.cdot_workspace/wfr.cdot_workspace.size) < 1e-7

    wfr.WFRStep(0, c_0, w_0, wfr.c_workspace, wfr.w_workspace)
    assert np.linalg.norm(wfr.c_workspace - c_0)/np.linalg.norm(c_0) < 1e-7
    assert np.linalg.norm(wfr.w_workspace - w_0)/np.linalg.norm(w_0) < 1e-7