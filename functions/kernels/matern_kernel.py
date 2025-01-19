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

def matern_nu_0_5_eval(r):
    return np.exp(-r)
def matern_nu_0_5_negdiff(r):
    return np.exp(-r)
def matern_nu_0_5_log(r):
    return -r
def matern_nu_0_5_negdiff_log(r):
    return -r

def matern_nu_1_5_eval(r):
    r_sqrt3 = np.sqrt(3) * r
    return (1 + r_sqrt3) * np.exp(-r_sqrt3)
def matern_nu_1_5_negdiff(r):
    r_sqrt3 = np.sqrt(3) * r
    return 3 * r * np.exp(-r_sqrt3)
def matern_nu_1_5_log(r):
    r_sqrt3 = np.sqrt(3) * r
    return np.log(1 + r_sqrt3) - r_sqrt3
def matern_nu_1_5_negdiff_log(r):
    r_sqrt3 = np.sqrt(3) * r
    return np.log(3 * r) - r_sqrt3

def matern_nu_2_5_eval(r):
    r_sqrt5 = np.sqrt(5) * r
    return (1 + r_sqrt5 + r_sqrt5**2 / 3) * np.exp(-r_sqrt5)
def matern_nu_2_5_negdiff(r):
    r_sqrt5 = np.sqrt(5) * r
    return (5*r/3) * np.exp(-r_sqrt5) * (1 + r_sqrt5)
def matern_nu_2_5_log(r):
    r_sqrt5 = np.sqrt(5) * r
    return np.log(1 + r_sqrt5 + r_sqrt5**2 / 3) - r_sqrt5
def matern_nu_2_5_negdiff_log(r):
    r_sqrt5 = np.sqrt(5) * r
    return np.log((5*r/3)*(1 + r_sqrt5)) - r_sqrt5
class MaternKernel:
    def __init__(self, bandwidth, nu = 2.5):
        self.sigma = bandwidth
        self.nu = nu
        if self.nu == 0.5:
            kernel_1d = matern_nu_0_5_eval
            kernel_1d_negdiff = matern_nu_0_5_negdiff
            kernel_1d_log = matern_nu_0_5_log
            kernel_1d_negdiff_log = matern_nu_0_5_negdiff_log
        elif self.nu == 1.5:
            kernel_1d = matern_nu_1_5_eval
            kernel_1d_negdiff = matern_nu_1_5_negdiff
            kernel_1d_log = matern_nu_1_5_log
            kernel_1d_negdiff_log = matern_nu_1_5_negdiff_log
        elif self.nu == 2.5:
            kernel_1d = matern_nu_2_5_eval
            kernel_1d_negdiff = matern_nu_2_5_negdiff
            kernel_1d_log = matern_nu_2_5_log
            kernel_1d_negdiff_log = matern_nu_2_5_negdiff_log
        else:
            raise ValueError("Unsupported value of nu. Supported values are 0.5, 1.5, and 2.5.")
        kernel_1d = nb.jit(kernel_1d)
        kernel_1d_negdiff = nb.jit(kernel_1d_negdiff)
        kernel_1d_log = nb.jit(kernel_1d_log)
        kernel_1d_negdiff_log = nb.jit(kernel_1d_negdiff_log)

        self.kernel = self.matern_kernel(self.sigma, kernel_1d)
        self.kernel_grad = self.matern_kernel_grad(self.sigma, kernel_1d_negdiff)
        self.log_kernel = self.log_kernel_matern(self.sigma, kernel_1d_log)
        self.kernel_bar = self.kernel_bar_matern(self.sigma, kernel_1d_negdiff)
        self.log_kernel_bar = self.log_kernel_bar_matern(self.sigma, kernel_1d_negdiff_log)

    def matern_kernel(self, sigma, kernel1d):
        @nb.jit()
        def kernel_aux(x, y):
            return kernel1d(np.sqrt(np.sum((x - y) ** 2, axis=-1))/sigma)
        return kernel_aux

    def matern_kernel_grad(self, sigma, kernel1d_negdiff):
        @nb.jit()
        def kernel_aux(x, y):
            diff = x - y
            dist = np.sqrt(np.sum(diff ** 2, axis=-1))
            return diff * kernel1d_negdiff(dist/sigma)/(sigma*dist)
        return kernel_aux

    def log_kernel_matern(self, sigma, log_kernel1d):
        @nb.jit()
        def kernel_aux(x, y):
            return log_kernel1d(np.sqrt(np.sum((x - y) ** 2, axis=-1))/sigma)
        return kernel_aux

    def kernel_bar_matern(self, sigma, kernel1d_negdiff):
        @nb.jit()
        def kernel_aux(x, y):
            dist = np.sqrt(np.sum((x - y) ** 2, axis=-1))
            return kernel1d_negdiff(dist/sigma)/(sigma*dist)
        return kernel_aux

    def log_kernel_bar_matern(self, sigma, log_kernel1d_negdiff):
        @nb.jit()
        def kernel_aux(x, y):
            dist = np.sqrt(np.sum((x - y) ** 2, axis=-1))
            return log_kernel1d_negdiff(dist/sigma) - np.log(sigma*dist)
        return kernel_aux

    def get_key(self):
        return "MaternKernel", self.sigma, self.nu

    def __reduce__(self):
        return (self.__class__, (self.sigma, self.nu))