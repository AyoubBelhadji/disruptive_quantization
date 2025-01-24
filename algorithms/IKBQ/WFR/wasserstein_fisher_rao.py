#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 11/26/2024 12:49:00
@author: dannys4
"""

import numpy as np
import numba as nb
import numbers
from scipy.integrate import solve_ivp
from tools.utils import kernel_avg

from algorithms.IKBQ.iterative_kernel_based_quantization import IterativeKernelBasedQuantization

@nb.jit()
def WFR_ODE_centroid_diff(y_t, w_t, kernel_grad2, data_array, y_dot):
    # For each node X_i, computes E_pi[grad_2 K(Y, X_i)] - mean(w_j * grad_2 K(X_j, X_i))
    # Where pi is the distribution of the data points Y = data_array
    # And K is the kernel function
    # And grad K is the gradient of the kernel function, which we assume is grad_2 k(Y,X) = X*k(Y, X)
    M = len(w_t)
    for i in range(M):
        y_dot[i]  = w_t.dot(kernel_grad2(y_t, y_t[i]))
        y_dot[i] -= np.sum(kernel_grad2(data_array, y_t[i]), axis=0)/len(data_array)
        y_dot[i] *= -w_t[i]

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
        if self.ODE_solver_str == 'Euler':
            self.WFR_ODE_solver = self.Solve_WFR_ODE_Euler
            self.steps_per_iteration = params.get('steps_per_iteration', 1)
        else:
            self.WFR_ODE_solver = self.Solve_WFR_ODE_Scipy

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

    def WFRStep(self, t, c_t, w_t, c_tplus1, w_tplus1):
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
        self.state_0[:-self.K] = c_t.reshape(-1)
        self.state_0[-self.K:] = w_t

        # Solve the ODE
        state_1 = self.WFR_ODE_solver(tspan)

        # Copy output into the output arrays
        c_tplus1[:] = state_1[:-self.K].reshape((self.K, self.d))
        w_tplus1[:] = state_1[-self.K:].reshape(-1)

    def Solve_WFR_ODE_Scipy(self, tspan):
        """
        Solve the Wasserstein Fisher-Rao mean-field ODE over the given time span
        """
        state_1 = solve_ivp(self.WFR_ODE, tspan, self.state_0, t_eval=(tspan[1],), method=self.ODE_solver_str)
        if len(state_1.y) == 0:
            raise ValueError(f"ODE solver failed to converge for {tspan}, {state_1.y}")
        return state_1.y

    def Solve_WFR_ODE_Euler(self, tspan):
        """
        Solve the Wasserstein Fisher-Rao mean-field ODE over the given time span
        """
        t0, t1 = tspan
        num_steps = self.steps_per_iteration
        dt = (t1-t0)/num_steps
        state_t = self.state_0.copy()
        for t in np.linspace(t0, t1, num_steps, endpoint=False):
            diff = self.WFR_ODE(t, state_t)
            state_t[:] += dt*diff
        return state_t

    def WFR_ODE(self, _, state_t):
        """
        RHS for the Wasserstein Fisher-Rao mean-field ODE
        """
        y_t_vec = state_t[:-self.K]
        y_t = y_t_vec.reshape((self.K, self.d))
        w_t = state_t[-self.K:]
        WFR_ODE_centroid_diff(y_t, w_t, self.kernel_grad2, self.data_array, self.ydot_workspace)
        WFR_ODE_weight_diff(  y_t, w_t, self.kernel, self.data_array, self.wdot_workspace)
        # self.diff_workspace[:-self.K] = y_dot.flatten()
        # self.diff_workspace[-self.K:] = w_dot
        return self.diff_workspace

    def WFR_ODE_weight_diff(self, c_t, w_t):
        F_bar = np.zeros_like(w_t)
        for i in range(self.K):
            F_bar[i]  = w_t.dot(self.kernel(c_t, c_t[i]))
            F_bar[i] -= self.kernel(self.data_array, c_t[i]).mean()
        # int_F_bar = np.sum(F_bar)
        # Fisher-Rao adjustment
        # F_tilde = F_bar - int_F_bar
        w_dot = -w_t * F_bar
        return w_dot

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
    assert np.linalg.norm(wfr.ydot_workspace/wfr.ydot_workspace.size) < 1e-7

    wfr.WFRStep(0, c_0, w_0, wfr.y_workspace, wfr.w_workspace)
    assert np.linalg.norm(wfr.y_workspace - c_0)/np.linalg.norm(c_0) < 1e-7
    assert np.linalg.norm(wfr.w_workspace - w_0)/np.linalg.norm(w_0) < 1e-7