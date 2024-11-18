#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 25/25/2525 25:25:25
Also created on Sun Dec 10 12:26:17 2023
Also also created on Mon Nov 18 6:22:10 2024
@author: ayoubbelhadji
"""

#from .base_algorithm import AbstractAlgorithm
from algorithms.base_algorithm import AbstractAlgorithm

import numpy as np
from tqdm import tqdm


class IterativeKernelBasedQuantization(AbstractAlgorithm):
    def __init__(self, params):
        super().__init__(params)
        #print('params')
        #print(params)
        self.name = "IKBQ"
        self.shared_param = params.get('shared_param', 0.5)

        self.R = params.get('R')
        self.T = params.get('T')
        self.K = params.get('K')
        self.M = params.get('K')
        self.d = params.get('d') 
        self.N = params.get('N')
        self.data_array = None
        self.kernel = params.get('kernel')
        
        #print(self.kernel)
        self.domain = params.get('domain')
        #self.noise_schedule_functions_list = params.get('noise_schedule_functions_list')

        self.c_array_trajectory = np.zeros((self.R,self.T,self.K,self.d))
        self.w_array_trajectory = np.zeros((self.R,self.T,self.K))
        
        self.params = params
        
        
        
    def evaluate(self,metric):
        return 0
    
    
    def run(self, data_array):
        self.data_array = data_array

        # np.zeros((self.T,self.K))
        
        
        # np.zeros((self.T,self.K,self.d))
        
        
        # centroids_0_array = self.initial_distribution(self.K,self.d)
        # weights_0_array = float(1/self.K)*self.ones(self.K)
        # #np.random.multivariate_normal(mean, 1*np.asarray(cov1), K)
        
        
        if self.T < 1:
            raise ValueError("The number of iterations 'T' must be at least 1.")
        
        N_ = data_array.shape[0]
        if self.N == 0:
            raise ValueError("The data_array is empty.")
        if self.N != N_:
            raise ValueError("The shape of data_array doesn't correspond to N")

        c_0_array = self.initial_distribution.generate_samples(self.K)
        #w_0_array = 
        for r in range(self.R):
            if self.freeze_init == True:
                #print(self.c_array_trajectory[r,0,:,:].shape)
                self.c_array_trajectory[r,0,:,:] = c_0_array
            else:
                self.c_array_trajectory[r,0,:,:] = self.initial_distribution(self.K,self.d)
            self.w_array_trajectory[r,0,:] = float(1/self.K)*np.ones(self.K)
            
            for t in tqdm(range(self.T - 1), position=0):
                #print(f"Iteration {t + 1}/{self.T-1}")
                self.c_array_trajectory[r,t+1,:,:] = self.calculate_centroids(self.c_array_trajectory[r,t,:,:],t) 
                self.w_array_trajectory[r,t+1,:] = self.calculate_weights(self.c_array_trajectory[r,t,:,:])
            
        
        self.log(f"TODO")
        
        return self.c_array_trajectory, self.w_array_trajectory


    
    def log(self, message):
        print(f"Log from {self.__class__.__name__}: {message}")



# o_params = {
#     "name": 'MSIP',
#     "T": 100,
#     "reg": 0.000001,
#     "noise_schedule_params_list": noise_schedule_params_list
# }


# k_params = {
#     "sigma" : 2,
#     "kernel" : Gaussian_kernel,
#     "pre_kernel": quadratic_function
# }


# e_params = {
#     "init_freeze" : 0,
#     "init_distribution": init_distribution,
#     "save_gif": 1,
#     "save_pdf": 1,
#     "save_npy": 1,
#     "R": 10,
#     "plt_xlim": [-15,15],
#     "plt_ylim": [-15,15]
# }