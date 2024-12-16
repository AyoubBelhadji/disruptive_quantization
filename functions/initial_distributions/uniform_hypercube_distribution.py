#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 25/25/2525 25:25:25
Also created on Sun Dec 10 12:26:17 2023
Also also created on Mon Nov 18 6:22:10 2024
@author: ayoubbelhadji
"""
import numpy as np

class UniformHypercubeDistribution:
    def __init__(self, params):
        """
        Initialize the UniformHypercubeDistribution class with parameters.

        Parameters:
        - params: dict
            A dictionary containing 'low', 'high', and 'd'.
            - 'd': int
                Dimension of the space.
            - 'low': ndarray of shape (d,)
                Lower bounds of the hypercube along each dimension.
            - 'high': ndarray of shape (d,)
                Upper bounds of the hypercube along each dimension.
        """
        self.d = params.get('d')
        self.low = np.asarray(params.get('low', np.zeros(self.d)))
        self.high = np.asarray(params.get('high', np.ones(self.d)))

        if self.low.shape != (self.d,):
            raise ValueError("Shape of 'low' must be (d,)")
        if self.high.shape != (self.d,):
            raise ValueError("Shape of 'high' must be (d,)")

    def generate_samples(self, M, *_):
        """
        Generate M random samples uniformly distributed in the hypercube.

        Parameters:
        - M: int
            Number of samples to generate.

        Returns:
        - samples: ndarray of shape (M, d)
            Generated samples.
        """
        # Expand bounds to match shape for broadcasting
        low_expanded = self.low
        high_expanded = self.high

        return np.random.uniform(low=low_expanded, high=high_expanded, size=(M, self.d))

# Example Usage
if __name__ == '__main__':
    params = {
        'd': 2,
        'low': np.array([0, 0]),
        'high': np.array([1, 2])
    }
    uniform_dist = UniformHypercubeDistribution(params)
    samples = uniform_dist.generate_samples(100)
    print(samples)


