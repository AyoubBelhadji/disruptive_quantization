#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 25/25/2525 25:25:25
Also created on Sun Dec 10 12:26:17 2023
Also also created on Mon Nov 18 6:22:10 2024
@author: ayoubbelhadji
"""

from algorithms.base_algorithm import AbstractAlgorithm
import numpy as np
from tqdm import tqdm
import numba as nb
from goodpoints import kt



@nb.jit(parallel=True)
def calculate_labels(labels, data_array, c_array):
    # Calculate L2 distance between each pair of data points and centroids
    for i in nb.prange(data_array.shape[0]):
        diff = data_array[i] - c_array
        dists = np.sqrt(np.sum(diff**2, axis=-1))
        labels[i] = np.argmin(dists)

@nb.jit()
def calculate_weights(w_array, labels, N):
    # Use labels to calculate weights
    w_array[:] = np.bincount(labels, minlength=len(w_array))
    w_array[:] /= N

@nb.jit()
def calculate_centroids(y_array, prev_y, data_array, labels):
    # Use labels to calculate centroids
    for k in range(len(y_array)):
        data_k = data_array[labels == k]
        if len(data_k) == 0:
            y_array[k] = prev_y[k]
        else:
            y_array[k] = np.sum(data_k, axis=0)/len(data_k)

class KernelThinning(AbstractAlgorithm):
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
        
        self.HR = params.get('HR')
        
        self.kernel = self.kernel_scheduler.GetKernel()
        self.split_kernel = self.kernel
        self.swap_kernel = self.kernel
        
        self.data_array = None

        self.y_trajectory = np.zeros((self.R, self.T, self.K, self.d))
        self.w_trajectory = np.zeros((self.R, self.T, self.K))

        self.labels = np.zeros((self.N), dtype=np.int32)

        self.params = params

    def evaluate(self, metric):
        return 0

    def run(self, data_array):
        self.data_array = data_array
        
        coreset = kt.thin(self.data_array, self.HR, self.split_kernel, self.swap_kernel, delta=0.5, seed=None, store_K=False, 
                  meanK=None, unique=False, verbose=False)
        print(coreset.shape)
        return {"centroids": coreset, "weights": 0}
        # if self.T < 1:
        #     raise ValueError(
        #         "The number of iterations 'T' must be at least 1.")

        # N_ = data_array.shape[0]
        # if self.N == 0:
        #     raise ValueError("The data_array is empty.")
        # if self.N != N_:
        #     raise ValueError("The shape of data_array doesn't correspond to N")

        # c_0_array = self.initial_distribution.generate_samples(self.K, data_array)

        # for r in range(self.R):
        #     if self.freeze_init:
        #         self.y_trajectory[r, 0, :, :] = c_0_array
        #     else:
        #         self.y_trajectory[r, 0, :, :] = self.initial_distribution.generate_samples(self.K, data_array)

        #     for t in tqdm(range(self.T), position=0):
        #         y_t = self.y_trajectory[r, t, :, :]
        #         w_t = self.w_trajectory[r, t, :]
        #         calculate_labels(self.labels, self.data_array, y_t)
        #         calculate_weights(w_t, self.labels, self.N)
        #         if t == self.T - 1:
        #             break # Skip calculating next nodes for last iteration
        #         y_tplus1 = self.y_trajectory[r, t+1, :, :]
        #         calculate_centroids(y_tplus1, y_t, self.data_array, self.labels)

        # return {"centroids": self.y_trajectory, "weights": self.w_trajectory}
