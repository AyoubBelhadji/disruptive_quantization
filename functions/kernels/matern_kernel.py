#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 25/25/2525 25:25:25
Also created on Sun Dec 10 12:26:17 2023
Also also created on Mon Nov 18 6:22:10 2024
@author: ayoubbelhadji
"""

import numpy as np
from scipy.special import kv, gamma, gammaln
import time


import numpy as np

class MaternKernel:
    def __init__(self, bandwidth):
        self.sigma = bandwidth  # Length-scale parameter
        self.nu = 2.5            # Smoothness parameter
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

    # ----------------------------
    # Matérn Kernel with nu = 0.5
    # ----------------------------
    def matern_nu_0_5(self, sigma):
        def kernel_aux(x, y):
            r = np.linalg.norm(x - y, axis=-1)
            k = np.exp(-r / sigma)
            return k
        return kernel_aux

    def matern_nu_0_5_grad(self, sigma):
        def kernel_aux(x, y):
            diff = x - y
            r = np.linalg.norm(diff, axis=-1, keepdims=True)
            # Avoid division by zero
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

    # ----------------------------
    # Matérn Kernel with nu = 1.5
    # ----------------------------
    def matern_nu_1_5(self, sigma):
        def kernel_aux(x, y):
            r = np.linalg.norm(x - y, axis=-1)
            sqrt3_r_l = np.sqrt(3) * r / sigma
            k = (1 + sqrt3_r_l) * np.exp(-sqrt3_r_l)
            #print(k)
            #time.sleep(0.1)
            return k
        return kernel_aux

    def matern_nu_1_5_grad(self, sigma):
        def kernel_aux(x, y):
            diff = x - y
            r = np.linalg.norm(diff, axis=-1, keepdims=True)
            sqrt3_r_l = np.sqrt(3) * r / sigma
            # Avoid division by zero
            r_safe = np.maximum(r, np.finfo(float).eps)
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
            #print(log_k)
            #print('log_k')
            #time.sleep(0.1)
            return log_k
        return kernel_aux

    # ----------------------------
    # Matérn Kernel with nu = 2.5
    # ----------------------------
    # def matern_nu_2_5(self, sigma):
    #     def kernel_aux(x, y):
    #         r = np.linalg.norm(x - y, axis=-1)
    #         sqrt5_r_l = np.sqrt(5) * r / sigma
    #         k = (1 + sqrt5_r_l + (5 * r**2) / (3 * sigma**2)) * np.exp(-sqrt5_r_l)
    #         return k
    #     return kernel_aux
    
    
    def matern_nu_2_5(self,sigma):
        def kernel_aux(x, y):
            if x.ndim == 1 and y.ndim == 1:
                r = np.linalg.norm(x - y)
                sqrt5_r_l = np.sqrt(5) * r / sigma
                k = (1 + sqrt5_r_l + (5 * r**2) / (3 * sigma**2)) * np.exp(-sqrt5_r_l)
                return k
            else:
                x = np.expand_dims(x, axis=1) 
                y = np.expand_dims(y, axis=0)
                
                r = np.sqrt(np.sum((x - y) ** 2, axis=-1))
                sqrt5_r_l = np.sqrt(5) * r / sigma
                k = (1 + sqrt5_r_l + (5 * r**2) / (3 * sigma**2)) * np.exp(-sqrt5_r_l)
                return k
        
        return kernel_aux

    def matern_nu_2_5_grad(self, sigma):
        def kernel_aux(x, y):
            diff = x - y  # Shape: (..., D)
            r = np.linalg.norm(diff, axis=-1, keepdims=True)
            sqrt5_r_l = np.sqrt(5) * r / sigma
            # Avoid division by zero
            r_safe = np.maximum(r, np.finfo(float).eps)
            exp_term = np.exp(-sqrt5_r_l)
            coeff = - (5 / (3 * sigma**4 * r_safe))
            grad = coeff * diff * (sigma**2 + 5 * r_safe**2)
            return grad
        return kernel_aux

    def pre_kernel_nu_2_5(self, sigma):
        def kernel_aux(x, y):
            r = np.linalg.norm(x - y, axis=-1)
            sqrt5_r_l = np.sqrt(5) * r / sigma
            term1 = np.log(1 + sqrt5_r_l + (5 * r**2) / (3 * sigma**2))
            log_k = term1 - sqrt5_r_l
            return log_k
        return kernel_aux


# class MaternKernel:
#     def __init__(self, bandwidth):
#         self.sigma = bandwidth  # Length-scale parameter
#         self.nu = 5.5            # Smoothness parameter
#         self.kernel = self.matern_kernel(self.sigma, self.nu)
#         self.pre_kernel = self.pre_kernel_function(self.sigma, self.nu)
#         self.kernel_grad = self.matern_kernel_grad(self.sigma, self.nu)
        
#     def matern_kernel(self, sigma, nu):
#         def kernel_aux(x, y):
#             diff = x - y
#             r = np.linalg.norm(diff, axis=-1)
#             zero_r = (r == 0)
#             # Replace zeros with a small epsilon to avoid division by zero
#             r_safe = np.where(zero_r, np.finfo(float).eps, r)
#             scaled_r = np.sqrt(2 * nu) * r_safe / sigma
#             # Compute the kernel
#             coeff = (2 ** (1 - nu)) / gamma(nu)
#             k = coeff * (scaled_r ** nu) * kv(nu, scaled_r)
#             # Handle small scaled_r values
#             small_scaled_r = (scaled_r < 1e-6)
#             k = np.where(zero_r, 1.0, k)
#             k = np.where(small_scaled_r & (~zero_r), 1.0, k)
#             return k
#         return kernel_aux

#     def matern_kernel_grad(self, sigma, nu):
#         def kernel_aux(x, y):
#             diff = x - y  # Shape: (..., D)
#             r = np.linalg.norm(diff, axis=-1, keepdims=True)  # Shape: (..., 1)
#             zero_r = (r == 0)
#             # Replace zeros with a small epsilon to avoid division by zero
#             r_safe = np.where(zero_r, np.finfo(float).eps, r)
#             scaled_r = np.sqrt(2 * nu) * r_safe / sigma  # Shape: (..., 1)
#             # Compute the gradient
#             coeff = (2 ** (1 - nu)) / gamma(nu)
#             kv_nu = kv(nu, scaled_r)
#             kv_nu1 = kv(nu - 1, scaled_r)
#             # Derivative of the Bessel function
#             bessel_derivative = -kv_nu1 - (nu / scaled_r) * kv_nu
#             # Gradient computation
#             grad_coeff = coeff * (scaled_r ** (nu - 1)) * bessel_derivative * (np.sqrt(2 * nu) / sigma)
#             # Avoid division by zero in diff / r_safe
#             grad = grad_coeff * (diff / r_safe)
#             # Handle small scaled_r values
#             small_scaled_r = (scaled_r < 1e-6)
#             grad = np.where(zero_r, 0.0, grad)
#             grad = np.where(small_scaled_r & (~zero_r), 0.0, grad)
#             return grad
#         return kernel_aux

#     def pre_kernel_function(self, sigma, nu):
#         def kernel_aux(x, y):
#             diff = x - y
#             r = np.linalg.norm(diff, axis=-1)
#             zero_r = (r == 0)
#             # Replace zeros with a small epsilon to avoid division by zero
#             r_safe = np.where(zero_r, np.finfo(float).eps, r)
#             scaled_r = np.sqrt(2 * nu) * r_safe / sigma
#             # Compute the logarithm of the kernel
#             log_coeff = (1 - nu) * np.log(2) - gammaln(nu)
#             kv_vals = kv(nu, scaled_r)
#             # Handle small scaled_r values
#             small_scaled_r = (scaled_r < 1e-6)
#             # For small scaled_r, use the asymptotic expansion
#             log_kv = np.where(
#                 small_scaled_r,
#                 np.log(0.5 * gamma(nu)) - nu * np.log(0.5 * scaled_r),
#                 np.log(kv_vals)
#             )
#             log_kernel = log_coeff + nu * np.log(scaled_r) + log_kv
#             # Handle cases where scaled_r is zero
#             log_kernel = np.where(zero_r, 0.0, log_kernel)
#             return log_kernel
#         return kernel_aux


# class GaussianKernel:
#     def __init__(self, bandwidth):
#         self.sigma = bandwidth
#         self.kernel = self.Gaussian_kernel(self.sigma)
#         self.pre_kernel = self.quadratic_function(self.sigma)
#         self.kernel_grad2 = self.Gaussian_kernel_grad2(self.sigma)

#     def Gaussian_kernel(self, sigma):
#         def kernel_aux(x, y):
#             log_output = -((np.linalg.norm(x - y, axis=-1)) ** 2) / (2 * (sigma ** 2))
#             return np.exp(log_output)
#         return kernel_aux

#     def Gaussian_kernel_grad2(self, sigma):
#         def kernel_aux(x, y):
#             diff = y - x
#             log_kernel = -np.linalg.norm(diff, axis=-1) ** 2 / (2 * (sigma ** 2))
#             kernel_out = np.exp(log_kernel).reshape(-1,1)
#             return np.squeeze(kernel_out * diff) * sigma
#         return kernel_aux

#     def quadratic_function(self, sigma):
#         def kernel_aux(x, y):
#             log_output = -((np.linalg.norm(x - y, axis=-1)) ** 2) / (2 * (sigma ** 2))
#             return log_output
#         return kernel_aux





# class MaternKernel:
#     def __init__(self, bandwidth):
#         self.sigma = bandwidth  
#         self.nu = 3     
#         self.kernel = self.matern_kernel(self.sigma, self.nu)
#         self.pre_kernel = self.pre_kernel_function(self.sigma, self.nu)
#         self.kernel_grad2 = self.matern_kernel_grad2(self.sigma, self.nu)
    
#     def matern_kernel(self, sigma, nu):
#         def kernel_aux(x, y):
#             r = np.linalg.norm(x - y, axis=-1)
#             if np.any(r == 0):
#                 r = np.where(r == 0, np.finfo(float).eps, r)  
#             scaled_r = np.sqrt(2 * nu) * r / sigma
#             coeff = (2 ** (1 - nu)) / gamma(nu)
#             k = coeff * (scaled_r ** nu) * kv(nu, scaled_r)
#             k[scaled_r == 0] = 1  
#             return k
#         return kernel_aux

#     def matern_kernel_grad2(self, sigma, nu):
#         def kernel_aux(x, y):
#             diff = x - y  
#             r = np.linalg.norm(diff, axis=-1, keepdims=True)  
#             if np.any(r == 0):
#                 r = np.where(r == 0, np.finfo(float).eps, r)  
#             scaled_r = np.sqrt(2 * nu) * r / sigma  
#             coeff = (2 ** (1 - nu)) / gamma(nu)
#             kv_nu = kv(nu, scaled_r)
#             kv_nu1 = kv(nu - 1, scaled_r)
            
#             bessel_derivative = -kv_nu1 - (nu / scaled_r) * kv_nu
            
#             grad = coeff * (scaled_r ** (nu - 1)) * bessel_derivative * (np.sqrt(2 * nu) / sigma) * (diff / r)
#             return grad
#         return kernel_aux

#     def pre_kernel_function(self, sigma, nu):
#         def kernel_aux(x, y):
#             r = np.linalg.norm(x - y, axis=-1)
#             if np.any(r == 0):
#                 r = np.where(r == 0, np.finfo(float).eps, r)  
#             scaled_r = np.sqrt(2 * nu) * r / sigma
#             log_coeff = np.log((2 ** (1 - nu)) / gamma(nu))
#             log_kernel = log_coeff + nu * np.log(scaled_r) + np.log(kv(nu, scaled_r))
#             return log_kernel
#         return kernel_aux
    
    
    
    
# class MaternKernel:
#     def __init__(self, bandwidth):
#         self.sigma = bandwidth  # Length-scale parameter
#         self.nu = 1.5            # Smoothness parameter
#         self.kernel = self.matern_kernel(self.sigma, self.nu)
#         self.pre_kernel = self.pre_kernel_function(self.sigma, self.nu)
#         self.kernel_grad = self.matern_kernel_grad(self.sigma, self.nu)
    
#     def matern_kernel(self, sigma, nu):
#         def kernel_aux(x, y):
#             diff = x - y
#             r = np.linalg.norm(diff, axis=-1)
#             zero_r = (r == 0)
#             # Replace zeros with a small epsilon to avoid division by zero
#             r_safe = np.where(zero_r, np.finfo(float).eps, r)
#             scaled_r = np.sqrt(2 * nu) * r_safe / sigma
#             coeff = (2 ** (1 - nu)) / gamma(nu)
#             # Compute the kernel
#             k = coeff * (scaled_r ** nu) * kv(nu, scaled_r)
#             # Handle the case when r=0
#             k = np.where(zero_r, 1.0, k)
#             return k
#         return kernel_aux

#     def matern_kernel_grad(self, sigma, nu):
#         def kernel_aux(x, y):
#             diff = x - y  # Shape: (..., D)
#             r = np.linalg.norm(diff, axis=-1, keepdims=True)  # Shape: (..., 1)
#             zero_r = (r == 0)
#             # Replace zeros with a small epsilon to avoid division by zero
#             r_safe = np.where(zero_r, np.finfo(float).eps, r)
#             scaled_r = np.sqrt(2 * nu) * r_safe / sigma  # Shape: (..., 1)
#             coeff = (2 ** (1 - nu)) / gamma(nu)
#             kv_nu = kv(nu, scaled_r)
#             kv_nu1 = kv(nu - 1, scaled_r)
#             # Derivative of the Bessel function
#             bessel_derivative = -kv_nu1 - (nu / scaled_r) * kv_nu
#             # Gradient computation
#             grad_coeff = coeff * (scaled_r ** (nu - 1)) * bessel_derivative * (np.sqrt(2 * nu) / sigma)
#             # Avoid division by zero in diff / r_safe
#             grad = grad_coeff * (diff / r_safe)
#             # Handle the case when r=0
#             grad = np.where(zero_r, 0.0, grad)
#             return grad
#         return kernel_aux

#     def pre_kernel_function(self, sigma, nu):
#         def kernel_aux(x, y):
#             diff = x - y
#             r = np.linalg.norm(diff, axis=-1)
#             zero_r = (r == 0)
#             # Replace zeros with a small epsilon to avoid division by zero
#             r_safe = np.where(zero_r, np.finfo(float).eps, r)
#             scaled_r = np.sqrt(2 * nu) * r_safe / sigma
#             log_coeff = (1 - nu) * np.log(2) - np.log(gamma(nu))
#             log_kernel = log_coeff + nu * np.log(scaled_r) + np.log(kv(nu, scaled_r))
#             # Handle the case when r=0
#             log_kernel = np.where(zero_r, 0.0, log_kernel)
#             return log_kernel
#         return kernel_aux
