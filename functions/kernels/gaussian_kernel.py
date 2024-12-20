#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 25/25/2525 25:25:25
Also created on Sun Dec 10 12:26:17 2023
Also also created on Mon Nov 18 6:22:10 2024
@author: ayoubbelhadji
"""

import numpy as np

class GaussianKernel:
    def __init__(self, bandwidth):
        self.sigma = bandwidth
        self.kernel = self.Gaussian_kernel(self.sigma)
        self.pre_kernel = self.quadratic_function(self.sigma)
        self.kernel_grad2 = self.Gaussian_kernel_grad2(self.sigma)

    # def Gaussian_kernel(self, sigma):
    #     def kernel_aux(x, y):
    #         log_output = -((np.linalg.norm(x - y, axis=-1)) ** 2) / (2 * (sigma ** 2))
    #         return np.exp(log_output)
    #     return kernel_aux

    def Gaussian_kernel(self,sigma):
        def kernel_aux(x, y):
            if x.ndim == 1 and y.ndim == 1:
                squared_norm = np.sum((x - y) ** 2)
                return np.exp(-squared_norm / (2 * (sigma ** 2)))
            else:
                x = np.expand_dims(x, axis=1)  
                y = np.expand_dims(y, axis=0)  
                squared_norms = np.sum((x - y) ** 2, axis=-1)  
                return np.exp(-squared_norms / (2 * (sigma ** 2)))
        
        return kernel_aux

    def Gaussian_kernel_grad2(self, sigma):
        def kernel_aux(x, y):
            diff = x - y
            log_kernel = -np.linalg.norm(diff, axis=-1) ** 2 / (2 * (sigma ** 2))
            kernel_out = np.exp(log_kernel).reshape(-1,1)
            return np.squeeze(kernel_out * diff) / (sigma ** 2)
        return kernel_aux

    def quadratic_function(self, sigma):
        def kernel_aux(x, y):
            log_output = -((np.linalg.norm(x - y, axis=-1)) ** 2) / (2 * (sigma ** 2))
            return log_output
        return kernel_aux

