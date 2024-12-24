#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 25/25/2525 25:25:25
Also created on Sun Dec 10 12:26:17 2023
Also also created on Mon Nov 18 6:22:10 2024
@author: ayoubbelhadji
"""

import numpy as np


class GaussianDistribution:
    def __init__(self, params):
        """
        Initialize the GaussianDistribution class with parameters.

        Parameters:
        - params: dict
            A dictionary containing 'mean' and 'covariance' keys.
            - 'mean': ndarray of shape (d,)
                Mean vector of the Gaussian distribution.
            - 'covariance': ndarray of shape (d, d)
                Covariance matrix of the Gaussian distribution.
        """
        self.d = params.get('d')
        self.mean = np.asarray(params.get('mean', np.zeros(self.d)))  # Default mean is a zero vector
        self.covariance = np.asarray(params.get('covariance', np.eye(self.d)))  # Default covariance is an identity matrix
        self.rng = params.get('rng', np.random.default_rng()) # Random number generator

    def generate_samples(self, M, *_):
        """
        Generate M random samples from the Gaussian distribution.

        Parameters:
        - M: int
            Number of samples to generate.

        Returns:
        - samples: ndarray of shape (M, d)
            Generated samples.
        """

        return self.rng.multivariate_normal(self.mean, self.covariance, M)

# Example Usage
if __name__ == '__main__':
    params = {
        'mean': np.array([50, 50]),
        'covariance': np.array([[1, 0], [0, 1]])
    }
    gaussian = GaussianDistribution(params, np.random.default_rng())
    samples = gaussian.generate_samples(100)
    print(samples)



