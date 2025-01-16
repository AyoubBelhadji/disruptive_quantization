#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 09/01/2025 00:00:00
@author: dannys4
"""

import numpy as np

class InverseMultiQuadricKernel:
    def __init__(self, bandwidth, power=0.5):
        # Suggested bandwidth = 1/sqrt(2*dim) (Dwivedi, Mackey 2022)
        self.sigma = bandwidth
        self.neg_power = -power
        self.kernel = self.kernel_constructor(self.sigma, self.neg_power)
        self.pre_kernel = self.log_kernel_constructor(self.sigma, self.neg_power)

    def kernel_constructor(self, sigma, neg_power):
        if neg_power == -0.5:
            def kernel_aux(x, y):
                if not (x.ndim == 1 and y.ndim == 1):
                    x = np.expand_dims(x, axis=1)
                    y = np.expand_dims(y, axis=0)
                dist_sq = np.sum((x - y)**2, axis=-1) / (sigma ** 2)
                return 1 / np.sqrt(1 + dist_sq)
        else:
            def kernel_aux(x, y):
                if not (x.ndim == 1 and y.ndim == 1):
                    x = np.expand_dims(x, axis=1)
                    y = np.expand_dims(y, axis=0)
                dist_sq = np.sum((x - y)**2, axis=-1) / (sigma ** 2)
                return (1 + dist_sq) ** neg_power
        return kernel_aux

    def log_kernel_constructor(self, sigma, neg_power):
        def kernel_aux(x, y):
            dist_sq = np.sum((x - y)**2, axis=-1) / (sigma ** 2)
            return np.log(1 + dist_sq) * neg_power
        return kernel_aux

    def get_key(self):
        return f"InverseMultiQuadricKernel_{self.neg_power}", self.sigma