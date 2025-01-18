#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 25/25/2525 25:25:25
Also created on Sun Dec 10 12:26:17 2023
Also also created on Mon Nov 18 6:22:10 2024
@author: ayoubbelhadji
"""

# from .base_algorithm import AbstractAlgorithm
from algorithms.base_algorithm import AbstractAlgorithm
from abc import abstractmethod
import numpy as np
from tqdm import tqdm


class IterativeKernelBasedQuantization(AbstractAlgorithm):
    def __init__(self, params):
        super().__init__(params)
        self.name = "IKBQ"
        self.shared_param = params.get('shared_param', 0.5)

        self.R = params.get('R')
        self.T = params.get('T')
        self.K = params.get('K')
        self.M = params.get('K')
        self.d = params.get('d')
        self.N = params.get('N')
        self.data_array = None
        self.kernel_scheduler = params.get('kernel')
        self.initial_distribution = params.get('initial_distribution')
        self.freeze_init = params.get('freeze_init')

        self.domain = params.get('domain')
        self.noise_schedule_function = params.get('noise_schedule_function')
        self.use_projection = params.get('use_projection')

        self.c_array_trajectory = np.zeros((self.R, self.T, self.K, self.d))
        self.w_array_trajectory = np.zeros((self.R, self.T, self.K))

        self.params = params

    def evaluate(self, metric):
        return 0

    def inject_noise_centroids(self, c_array, t):
        if self.noise_schedule_function is None:
            return c_array
        c_array_ni = self.noise_schedule_function.generate_noise(c_array, t)
        return c_array_ni

    def run(self, data_array):
        self.data_array = data_array

        if self.T < 1:
            raise ValueError(
                "The number of iterations 'T' must be at least 1.")

        N_ = data_array.shape[0]
        if self.N == 0:
            raise ValueError("The data_array is empty.")
        if self.N != N_:
            raise ValueError(f"The length of data_array {N_} doesn't correspond to N {self.N}.")

        c_0_array = self.initial_distribution.generate_samples(self.K, self.data_array)

        for r in range(self.R):
            self.kernel_scheduler.IncrementSchedule()
            if self.freeze_init:
                self.c_array_trajectory[r, 0, :, :] = c_0_array
            else:
                self.c_array_trajectory[r, 0, :, :] = self.initial_distribution.generate_samples(
                    self.K, self.data_array)

            self.w_array_trajectory[r, 0, :] = self.calculate_weights(
                self.c_array_trajectory[r, 0, :, :], 0, np.zeros(self.K))

            for t in tqdm(range(self.T - 1), position=0):
                c_t = self.c_array_trajectory[r, t, :, :]
                w_t = self.w_array_trajectory[r, t, :]
                c_t_plus_1 = self.calculate_centroids(c_t, t, w_t)

                # Adjust the centroids according to noise schedule
                c_t_plus_1 = self.inject_noise_centroids(c_t_plus_1, t)
                # Project the centroids back into the domain
                if self.use_projection:
                    c_t_plus_1 = self.domain.project(c_t_plus_1)

                self.c_array_trajectory[r, t+1, :, :] = c_t_plus_1
                self.w_array_trajectory[r, t+1, :] = self.calculate_weights(c_t, t, w_t)

        return {"centroids": self.c_array_trajectory, "weights": self.w_array_trajectory}

    @abstractmethod
    def calculate_weights(self, c_array: np.ndarray, t: float, w_array: np.ndarray):
        pass

    @abstractmethod
    def calculate_centroids(self, c_array: np.ndarray, t: float, w_array: np.ndarray):
        pass

    def log(self, message):
        print(f"Log from {self.__class__.__name__}: {message}")