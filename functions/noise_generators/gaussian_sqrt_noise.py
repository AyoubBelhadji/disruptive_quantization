#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 25/25/2525 25:25:25
Also created on Sun Dec 10 12:26:17 2023
Also also created on Mon Nov 18 6:22:10 2024
@author: ayoubbelhadji
"""

import numpy as np

class GaussianSqrtNoise:
    def __init__(self, params):
        """
        Initialize the GaussianSqrtNoise class with parameters.

        Parameters:
        - params: dict
            A dictionary containing 'mean' and 'covariance' keys.
            - 'd': the dimension
            - 'beta': the initial noise level
        """
        self.d = params.get('d')
        self.mean = np.zeros(self.d)
        self.covariance = np.eye(self.d)
        self.beta = params.get('beta_ns', 0.0)

    def generate_noise(self, c_array, t):
        """
        Generate noise and add it to the input array.

        Parameters:
        - c_array (numpy.ndarray): The input array with shape (M, d).
        - t (float): The time step or parameter affecting the noise scale.

        Returns:
        - numpy.ndarray: The input array with added noise.
        """
        if self.beta == 0.0:
            return c_array
        else:
            M, _ = c_array.shape
            covariance = (self.beta / np.sqrt(t + 1)) * self.covariance
            noise = np.random.multivariate_normal(self.mean, covariance, M)
            return c_array + noise
