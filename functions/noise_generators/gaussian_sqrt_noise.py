#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from functions.noise_generators.add_sqrt_noise_generator import AddSqrtNoiseGenerator

class GaussianSqrtNoise(AddSqrtNoiseGenerator):
    def __init__(self, params, rng: np.random.Generator):
        """
        Initialize the GaussianSqrtNoise class with parameters.

        Parameters:
        - params: dict
            A dictionary containing 'mean' and 'covariance' keys.
            - 'd': the dimension
            - 'beta': the initial noise level
        """
        super().__init__(params, rng)
        self.d = params.get('d')
        self.mean = np.zeros(self.d)
        self.covariance = np.eye(self.d)

    def generate_noise_internal(self, c_array):
        M = len(c_array)
        return self.rng.multivariate_normal(self.mean, self.covariance, M)
