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
def IMQ_sqrt2_diff(r):
    rp1_pow = np.sqrt((1 + r) ** 3)
    return -1 / (2 * rp1_pow)
@nb.jit()
def log_IMQ_sqrt2(r):
    return -np.log(1 + r) / 2

def IMQ_eval(neg_power):
    @nb.jit()
    def kernel_aux(r):
        return (1 + r) ** neg_power
    return kernel_aux
def IMQ_diff(neg_power):
    @nb.jit()
    def kernel_aux(r):
        return neg_power * ((1 + r) ** (neg_power - 1))
    return kernel_aux
def log_IMQ(neg_power):
    @nb.jit()
    def kernel_aux(r):
        return np.log(1 + r) * neg_power
    return kernel_aux

class InverseMultiQuadricKernel:
    def __init__(self, bandwidth, power=0.5):
        # Suggested bandwidth = 1/sqrt(2*dim) (Dwivedi, Mackey 2022)
        self.sigma = bandwidth
        self.neg_power = -power
        kernel1d_eval = IMQ_sqrt2_eval if power == 0.5 else IMQ_eval(self.neg_power)
        kernel1d_diff = IMQ_sqrt2_diff if power == 0.5 else IMQ_diff(self.neg_power)
        kernel1d_log = log_IMQ_sqrt2 if power == 0.5 else log_IMQ(self.neg_power)
        sigma_sq = bandwidth ** 2
        self.kernel = self.kernel_constructor(sigma_sq, kernel1d_eval)
        self.log_kernel = self.kernel_log_constructor(sigma_sq, kernel1d_log)
        self.kernel_bar = self.kernel_bar_constructor(sigma_sq, kernel1d_diff)

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

    def kernel_bar_constructor(self, sigma_sq, kernel1d_diff):
        @nb.jit()
        def kernel_aux(x, y):
            dist_sq = np.sum((x - y)**2, axis=-1) / sigma_sq
            return -kernel1d_diff(dist_sq) / sigma_sq
        return kernel_aux

    def get_key(self):
        return f"InverseMultiQuadricKernel_{self.neg_power}", self.sigma

    def __reduce__(self):
        return (self.__class__, (self.sigma, self.neg_power))