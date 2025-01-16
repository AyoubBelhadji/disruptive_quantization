#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 25/25/2525 25:25:25
Also created on Sun Dec 10 12:26:17 2023
Also also created on Mon Nov 18 6:22:10 2024
@author: ayoubbelhadji
"""

import numpy as np
import numba as nb
from scipy.special import kv, gamma, gammaln

class MaternKernel:
    def __init__(self, bandwidth, nu = 2.5):
        self.sigma = bandwidth
        self.nu = nu
        if self.nu == 0.5:
            self.kernel = self.matern_nu_0_5(self.sigma)
            self.kernel_grad = self.matern_nu_0_5_grad(self.sigma)
            self.pre_kernel = self.pre_kernel_nu_0_5(self.sigma)
        elif self.nu == 1.5:
            self.kernel = self.matern_nu_1_5(self.sigma)
            self.kernel_grad = self.matern_nu_1_5_grad(self.sigma)
            self.pre_kernel = self.pre_kernel_nu_1_5(self.sigma)
        elif self.nu == 2.5:
            self.kernel = self.matern_nu_2_5(self.sigma)
            self.kernel_grad = self.matern_nu_2_5_grad(self.sigma)
            self.pre_kernel = self.pre_kernel_nu_2_5(self.sigma)
        else:
            raise ValueError("Unsupported value of nu. Supported values are 0.5, 1.5, and 2.5.")
        self.kernel = nb.jit(self.kernel)
        self.kernel_grad = nb.jit(self.kernel_grad)
        self.pre_kernel = nb.jit(self.pre_kernel)

    # Matérn Kernel with nu = 0.5
    def matern_nu_0_5(self, sigma):
        def kernel_aux(x, y):
            r = np.linalg.norm(x - y, axis=-1)
            k = np.exp(-r / sigma)
            return k
        return kernel_aux

    # TODO: Fix grad
    def matern_nu_0_5_grad(self, sigma):
        def kernel_aux(x, y):
            diff = x - y
            r = np.linalg.norm(diff, axis=-1, keepdims=True)
            r_safe = np.maximum(r, np.finfo(float).eps)
            exp_term = np.exp(-r_safe / sigma)
            grad = - (exp_term / (sigma * r_safe)) * diff
            return grad
        return kernel_aux

    def pre_kernel_nu_0_5(self, sigma):
        def kernel_aux(x, y):
            r = np.linalg.norm(x - y, axis=-1)
            log_k = -r / sigma
            return log_k
        return kernel_aux


    # Matérn Kernel with nu = 1.5

    def matern_nu_1_5(self, sigma):
        def kernel_aux(x, y):
            r = np.linalg.norm(x - y, axis=-1)
            sqrt3_r_l = np.sqrt(3) * r / sigma
            k = (1 + sqrt3_r_l) * np.exp(-sqrt3_r_l)
            return k
        return kernel_aux

    # TODO: Fix grad
    def matern_nu_1_5_grad(self, sigma):
        def kernel_aux(x, y):
            diff = x - y
            r = np.linalg.norm(diff, axis=-1, keepdims=True)
            sqrt3_r_l = np.sqrt(3) * r / sigma
            exp_term = np.exp(-sqrt3_r_l)
            grad_coeff = - (3 / sigma**2) * exp_term
            grad = grad_coeff * diff
            return grad
        return kernel_aux

    def pre_kernel_nu_1_5(self, sigma):
        def kernel_aux(x, y):
            r = np.linalg.norm(x - y, axis=-1)
            sqrt3_r_l = np.sqrt(3) * r / sigma
            log_k = np.log(1 + sqrt3_r_l) - sqrt3_r_l
            return log_k
        return kernel_aux

    # ----------------------------
    # Matérn Kernel with nu = 2.5
    # ----------------------------

    def matern_nu_2_5(self,sigma):
        def kernel_aux(x, y):
            r = np.sqrt(np.sum((x - y) ** 2, axis=-1))
            sqrt5_r_l = np.sqrt(5) * r / sigma
            k = (1 + sqrt5_r_l + (5 * r**2) / (3 * sigma**2)) * np.exp(-sqrt5_r_l)
            return k
        return kernel_aux

    # TODO: Fix grad
    def matern_nu_2_5_grad(self, sigma):
        def kernel_aux(x, y):
            diff = x - y
            r = np.linalg.norm(diff, axis=-1, keepdims=True)
            sqrt5_r_l = np.sqrt(5) * r / sigma
            r_safe = np.maximum(r, np.finfo(float).eps)
            exp_term = np.exp(-sqrt5_r_l)
            coeff = - (5 / (3 * sigma**4 * r_safe))
            grad = coeff * diff * (sigma**2 + 5 * r_safe**2)
            return grad
        return kernel_aux

    def pre_kernel_nu_2_5(self, sigma):
        def kernel_aux(x, y):
            r_sq = np.sum((x - y) ** 2, axis=-1)
            r = np.sqrt(r_sq)
            sqrt5_r_l = np.sqrt(5) * r / sigma
            term1 = np.log(1 + sqrt5_r_l + (5 * r_sq) / (3 * sigma**2))
            log_k = term1 - sqrt5_r_l
            return log_k
        return kernel_aux

    def get_key(self):
        return "MaternKernel", self.sigma, self.nu