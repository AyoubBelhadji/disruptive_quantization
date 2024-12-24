#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 25/25/2525 25:25:25
Also created on Mon Nov 26 11:22:00 2024
@author: dannys4
"""

import numpy as np

class KmeansPlusPlusDistribution:
    def __init__(self, params):
        """
        Initialize the GreedyMaxDistanceDistribution class with (unused) parameters.

        """
        self.rng = params.get("rng", np.random.default_rng())

    def generate_samples(self, M, data_array):
        """
        Generate M random samples from the data distribution according to the kmeans++ algorithm.

        """
        N_, d = data_array.shape
        # Initialize the set of centers
        centers = np.empty((M, d))
        centers[0] = data_array[self.rng.choice(N_)]
        dists = np.empty((M-1,N_)) # We only have to search M-1 times
        weights = np.empty((N_,))
        for i in range(1, M):
            # Compute the distance of each point to the previously chosen center
            dists[i-1,:] = np.sum((data_array - centers[i-1])**2, axis=1)
            # Find distance to nearest center for each point
            np.min(dists[:i], axis=0, out=weights)
            weights[:] /= np.sum(weights)
            # Choose the next center with probability proportional to the squared-distance to closest node
            centers[i] = data_array[self.rng.choice(N_, p=weights)]

        return centers