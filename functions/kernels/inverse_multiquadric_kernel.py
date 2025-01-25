#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 09/01/2025 00:00:00
@author: dannys4
"""

import numpy as np
import numba as nb

@nb.jit()
def IMQ_sqrt2_eval(r):
    return 1 / np.sqrt(1 + r)
@nb.jit()
def IMQ_sqrt2_negdiff(r):
    rp1_pow = np.sqrt((1 + r) ** 3)
    return 1 / (2 * rp1_pow)
@nb.jit()
def log_IMQ_sqrt2(r):
    return -0.5*np.log(1 + r)
@nb.jit()
def log_IMQ_sqrt2_negdiff(r):
    return -np.sqrt(2) - 1.5*np.log(1 + r)

def IMQ_eval(neg_power):
    @nb.jit()
    def kernel_aux(r):
        return (1 + r) ** neg_power
    return kernel_aux
def IMQ_negdiff(neg_power):
    @nb.jit()
    def kernel_aux(r):
        return (-neg_power) * ((1 + r) ** (neg_power - 1))
    return kernel_aux
def log_IMQ(neg_power):
    @nb.jit()
    def kernel_aux(r):
        return np.log(1 + r) * neg_power
    return kernel_aux
def log_IMQ_negdiff(neg_power):
    @nb.jit()
    def kernel_aux(r):
        return np.log(-neg_power) + (neg_power - 1) * np.log(1 + r)
    return kernel_aux

class InverseMultiQuadricKernel:
    def __init__(self, bandwidth, power=0.5):
        # Suggested bandwidth = 1/sqrt(2*dim) (Dwivedi, Mackey 2022)
        self.sigma = bandwidth
        self.neg_power = -power
        self.kernel_bar_is_scaled_kernel = False
        kernel1d_eval = IMQ_sqrt2_eval if power == 0.5 else IMQ_eval(self.neg_power)
        kernel1d_negdiff = IMQ_sqrt2_negdiff if power == 0.5 else IMQ_negdiff(self.neg_power)
        kernel1d_log = log_IMQ_sqrt2 if power == 0.5 else log_IMQ(self.neg_power)
        kernel1d_negdiff_log = log_IMQ_sqrt2_negdiff if power == 0.5 else log_IMQ_negdiff(self.neg_power)
        sigma_sq = bandwidth ** 2
        self.kernel = self.kernel_constructor(sigma_sq, kernel1d_eval)
        self.log_kernel = self.kernel_log_constructor(sigma_sq, kernel1d_log)
        self.kernel_bar = self.kernel_bar_constructor(sigma_sq, kernel1d_negdiff)
        self.log_kernel_bar = self.log_kernel_bar_constructor(sigma_sq, kernel1d_negdiff_log)
        self.kernel_grad2 = self.kernel_grad_constructor(sigma_sq, kernel1d_negdiff)

    def kernel_constructor(self, sigma_sq, kernel1d):
        @nb.jit()
        def kernel_aux(x, y):
            dist_sq = np.sum((x - y)**2, axis=-1) / sigma_sq
            return kernel1d(dist_sq)
        return kernel_aux

    def kernel_log_constructor(self, sigma_sq, kernel1d_log):
        @nb.jit()
        def kernel_aux(x, y):
            dist_sq = np.sum((x - y)**2, axis=-1) / sigma_sq
            return kernel1d_log(dist_sq)
        return kernel_aux

    def kernel_bar_constructor(self, sigma_sq, kernel1d_negdiff):
        @nb.jit()
        def kernel_aux(x, y):
            diff = x - y
            dist_sq = np.sum(diff ** 2, axis=-1) / sigma_sq
            diff_eval = kernel1d_negdiff(dist_sq)
            return 2 * diff_eval / sigma_sq
        return kernel_aux

    def log_kernel_bar_constructor(self, sigma_sq, kernel1d_negdiff_log):
        @nb.jit()
        def kernel_aux(x, y):
            dist_sq = np.sum((x - y)**2, axis=-1) / sigma_sq
            return kernel1d_negdiff_log(dist_sq) + np.log(2/sigma_sq)
        return kernel_aux

    def kernel_grad_constructor(self, sigma_sq, kernel1d_negdiff):
        @nb.jit()
        def kernel_aux(x, y):
            diff = x - y
            dist_sq = np.sum(diff ** 2, axis=-1) / sigma_sq
            diff_eval = kernel1d_negdiff(dist_sq)
            for i in range(diff.shape[0]):
                diff[i] *= 2 * diff_eval[i] / sigma_sq
            return diff
        return kernel_aux

    def get_key(self):
        return f"InverseMultiQuadricKernel_{self.neg_power}", self.sigma

    def __reduce__(self):
        return (self.__class__, (self.sigma, self.neg_power))