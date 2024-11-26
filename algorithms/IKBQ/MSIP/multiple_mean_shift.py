#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 25/25/2525 25:25:25
Also created on Sun Dec 10 12:26:17 2023
Also also created on Mon Nov 18 6:22:10 2024
@author: ayoubbelhadji
"""


#from .sub_algorithm import SubAlgorithm
from algorithms.IKBQ.iterative_kernel_based_quantization import IterativeKernelBasedQuantization

import numpy as np





###### Adjugate matrix


def cofactor_matrix(matrix):
    """
    Calculate the cofactor matrix of a given square matrix.
    """
    n = matrix.shape[0]
    cofactor = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            # Minor of element (i, j)
            minor = np.delete(np.delete(matrix, i, axis=0), j, axis=1)
            # Cofactor is the determinant of the minor, times (-1)^(i+j)
            cofactor[i, j] = ((-1) ** (i + j)) * np.linalg.det(minor)
    
    return cofactor

def adjugate_matrix(matrix):
    """
    Calculate the adjugate (adjoint) of a given square matrix.
    """
    cofactor = cofactor_matrix(matrix)
    adjugate = cofactor.T  # Adjugate is the transpose of the cofactor matrix
    return adjugate





def stable_ms_map(x, n_array, pre_kernel):
    """
    Compute the stable mean shift map using pre-kernel values and array operations.
    
    Args:
        x: The input scalar.
        n_array: A NumPy array of scalars (e.g., n_list).
        pre_kernel: A function that computes the pre-kernel values.
    
    Returns:
        The updated position after applying the mean shift map.
    """
    
    # Compute pre-kernel values as a NumPy array
    pre_kernel_array = np.array([pre_kernel(n, x) for n in n_array])  # Assume pre_kernel can handle array input
    #print(n_array.shape)
    #print(pre_kernel_array.shape)
    pre_kernel_offset = np.max(pre_kernel_array)
    #print(x)
    #print(pre_kernel_offset)
    # Compute the weights (using broadcasting)
    weights = np.exp(pre_kernel_array - pre_kernel_offset)

    weights = weights.reshape((n_array.shape[0],1))
    # Compute the weighted sum of the differences
    a = np.sum(weights * n_array, axis=0)
    b = np.sum(weights)
    #print(weights.shape)
    #print(n_array.shape)
    #print('b')
    #print(b)
    #print(a)
    #print(a.shape)

    # Return the mean shift map value
    return (1 / b) * a




def stable_log_kde(x, n_array, pre_kernel):
    """
    Compute the stable mean shift map using pre-kernel values and array operations.
    
    Args:
        x: The input scalar.
        n_array: A NumPy array of scalars (e.g., n_list).
        pre_kernel: A function that computes the pre-kernel values.
    
    Returns:
        The updated position after applying the mean shift map.
    """
    
    # Compute pre-kernel values as a NumPy array
    pre_kernel_array = np.array([pre_kernel(n, x) for n in n_array])  # Assume pre_kernel can handle array input
    #print(n_array.shape)
    #print(pre_kernel_array.shape)
    pre_kernel_offset = np.max(pre_kernel_array)
    #print(x)
    #print(pre_kernel_offset)
    # Compute the weights (using broadcasting)
    weights = np.exp(pre_kernel_array - pre_kernel_offset)

    weights = weights.reshape((n_array.shape[0],1))
    # Compute the weighted sum of the differences
    a = np.sum(weights * n_array, axis=0)
    b = np.sum(weights)
    #print(weights.shape)
    #print(n_array.shape)
    #print('b')
    #print(b)
    #print(a)
    #print(a.shape)

    # Return the mean shift map value
    return pre_kernel_offset+np.log(b)


def average_x_v(x,y, v):
    """
    Compute the weighted mean of vectors using the Log-Sum-Exp trick for stability.
    Handles zero or near-zero weights gracefully.
    
    Args:
        x: An array of scalars (weights), shape (N,).
        y: An array of positive scalars (weights), shape (N,).
        v: An array of vectors (N, D), where each row is a vector v_i of dimension D.
        epsilon: Small value to handle near-zero weights.
    
    Returns:
        A vector representing the weighted mean.
    """
    
    
    nnzeros = np.where(np.abs(x) > 0.0)[0]
    #print('zeros')
    #print(len(nnzeros))
    #print(nnzeros)
    #print(x)
    #print(y)
    #print(nnzeros)
    
    if len(nnzeros)==x.shape[0]-1:
        zeros = np.where(np.abs(x) == 0.0)[0]
        #print(zeros)
        return v[zeros[0],:]
    elif len(nnzeros)==1:
        return v[nnzeros[0],:]
    else:
    
        t = np.log(y)
        t_max = np.max(t)
        
        t_adjusted = t - t_max 
        
        exp_y_adjusted = np.exp(t_adjusted)
        
        z = x*exp_y_adjusted
        
        
        N = z.shape[0]
        
        z = z.reshape((N,1))
        a = np.sum(z  * v, axis=0)
        b = np.sum(z)
    
    
        # Return the ratio
        return (1 / b) * a



class MultipleMeanShift(IterativeKernelBasedQuantization):
    
    def __init__(self, params):
        super().__init__(params)
        self.algo_name = 'Vanilla MMS'
        self.reg_K = params.get('reg_K')
        self.initial_distribution = params.get('initial_distribution')
        self.noise_schedule_function = params.get('noise_schedule_function')
        self.freeze_init = params.get('freeze_init')
        self.use_projection = params.get('use_projection')
        self.reg_K = 0.0001

    def calculate_weights(self, c_array):
        x_array = self.data_array
        K_matrix = np.zeros((self.M,self.M)) #### Be careful because K means kernel matrix and number of centroids
        mu_array = np.zeros((self.M))
        for m_1 in list(range(self.M)):
            for m_2 in list(range(self.M)):
                K_matrix[m_1,m_2] = self.kernel.kernel(c_array[m_1,:],c_array[m_2,:])
        
        for m in list(range(self.M)):
            tmp_list = [self.kernel.kernel(c_array[m,:], x_array[n,:]) for n in list(range(self.N))]
            mu_array[m] = sum(tmp_list)/self.N
        weights_array = np.dot(np.linalg.inv(K_matrix+self.reg_K*np.eye(self.M)),mu_array)
        return weights_array
    
#def calculate_weights(centroids, X, my_kernel,reg):

    def inject_noise(self,c_array,t):
        c_array_ni = self.noise_schedule_function.generate_noise(c_array,t)
        #,self.noise_schedule_functions_list[r]
        return c_array_ni
    
    


    def calculate_centroids(self, c_array,t):
        #N = len(centroids)
        #M = len(X)
        x_array = self.data_array

        K_matrix = np.zeros((self.M,self.M))
        K_inv_matrix = np.zeros((self.M,self.M))
        mu_0_array = np.zeros((self.M,1))
        v_0_array = np.zeros(self.M)
        log_v_0_array = np.zeros(self.M)
        ms_array = np.zeros((self.M,self.d))
        c_tplus1_array = np.zeros((self.M,self.d))
        
        
        #w_list = []
        #print(c_array.shape)
        for m_1 in list(range(self.M)):
            for m_2 in list(range(self.M)):
                K_matrix[m_1,m_2] = self.kernel.kernel(c_array[m_1,:],c_array[m_2,:])
        
        
        #print(K_matrix)
        K_inv_matrix = adjugate_matrix(K_matrix)
        
        for m in list(range(self.M)):
            tmp_0_list = [self.kernel.kernel(c_array[m,:], x_array[n,:]) for n in list(range(self.N))]
            v_0_array[m] = (1/self.N)*sum(tmp_0_list)
            ms_array[m,:] = stable_ms_map(c_array[m,:], self.data_array, self.kernel.pre_kernel) 
            log_v_0_array[m] = stable_log_kde(c_array[m,:], self.data_array, self.kernel.pre_kernel)
        #output_centroids_list = []
        for m in list(range(self.M)):
            arr_tmp = K_inv_matrix[m,:]*v_0_array
            arr_tmp = arr_tmp.reshape((self.M,1))
            c_tplus1_array[m,:] = average_x_v(K_inv_matrix[m,:],v_0_array, ms_array)
            #print(v_0_array)
            #print(ms_array)
            #print(K_inv_matrix[m,:])
        
        #print('hhhhrrrr')
        #print(c_tplus1_array)
        #print(self.noise_schedule_functions_list)
        c_tplus1_array_ni = self.inject_noise(c_tplus1_array,t)
        #print(c_tplus1_array)
        
        if self.use_projection==True:
            return self.domain.project(c_tplus1_array_ni)
        else:
            return c_tplus1_array_ni
        

    def run(self, data_array):
        super().run(data_array)