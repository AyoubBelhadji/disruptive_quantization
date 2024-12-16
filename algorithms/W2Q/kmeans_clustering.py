#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  25 11:37:45 2024

@author: dannys4
"""

from algorithms.base_algorithm import AbstractAlgorithm
from scipy.spatial.distance import cdist
import numpy as np
from tqdm import tqdm

class KmeansClustering(AbstractAlgorithm):
    def __init__(self, params):
        super().__init__(params)
        self.name = "kmeans"
        self.shared_param = params.get('shared_param', 0.5)

        self.initial_distribution = params.get('initial_distribution')
        self.freeze_init = params.get('freeze_init', False)
        self.R = params.get('R')
        self.T = params.get('T')
        self.K = params.get('K')
        self.d = params.get('d')
        self.N = params.get('N')
        self.data_array = None

        self.c_array_trajectory = np.zeros((self.R, self.T, self.K, self.d))
        self.w_array_trajectory = np.zeros((self.R, self.T, self.K))

        self.labels = np.zeros((self.N), dtype=np.int32)
        self.label_workspace = np.zeros((self.N, self.K))

        self.params = params

    def evaluate(self, metric):
        return 0

    def run(self, data_array):
        self.data_array = data_array

        if self.T < 1:
            raise ValueError(
                "The number of iterations 'T' must be at least 1.")

        N_ = data_array.shape[0]
        if self.N == 0:
            raise ValueError("The data_array is empty.")
        if self.N != N_:
            raise ValueError("The shape of data_array doesn't correspond to N")

        c_0_array = self.initial_distribution.generate_samples(self.K, data_array)

        for r in range(self.R):
            if self.freeze_init:
                self.c_array_trajectory[r, 0, :, :] = c_0_array
            else:
                self.c_array_trajectory[r, 0, :, :] = self.initial_distribution.generate_samples(self.K, data_array)

            for t in tqdm(range(self.T), position=0):
                c_t = self.c_array_trajectory[r, t, :, :]
                w_t = self.w_array_trajectory[r, t, :]
                self.calculate_labels(c_t)
                self.calculate_weights(w_t)
                if t == self.T - 1:
                    break # Skip calculating next nodes for last iteration
                c_tplus1 = self.c_array_trajectory[r, t+1, :, :]
                self.calculate_centroids(c_tplus1)

        return self.c_array_trajectory, self.w_array_trajectory

    def calculate_labels(self, c_array):
        # Calculate L2 distance between each pair of data points and centroids
        cdist(self.data_array, c_array, 'euclidean', out=self.label_workspace)
        self.labels[:] = self.label_workspace.argmin(axis=1, )

    def calculate_weights(self, w_array):
        # Use labels to calculate weights
        w_array[:] = np.bincount(self.labels, minlength=self.K)
        w_array[:] /= self.N

    def calculate_centroids(self, c_array):
        # Use labels to calculate centroids
        for k in range(self.K):
            c_array[k] = np.mean(self.data_array[self.labels == k], axis=0)