# from .sub_algorithm import SubAlgorithm
from algorithms.IKBQ import IterativeKernelBasedQuantization
import numpy as np
import numba as nb
from tools.utils import proj_simplex, broadcast_kernel, kernel_avg, kernel_bar_moment

@nb.jit()
def get_weights(kernel, y_array, x_array):
    K_matrix = broadcast_kernel(kernel, y_array, y_array)
    mu_array = kernel_avg(kernel, y_array, x_array)

    w_tplus1_tilde = mu_array / np.diag(K_matrix)
    # w_tplus1 = proj_simplex(w_tplus1_tilde)

    return w_tplus1_tilde

@nb.jit()
def get_centroids(y_array, x_array, kernel_bar, step_size):
    v1_bar = kernel_bar_moment(kernel_bar, y_array, x_array)
    v0_bar = kernel_avg(kernel_bar, y_array, x_array)
    for i in range(len(v0_bar)):
        v1_bar[i] = v1_bar[i] / v0_bar[i]

    y_tplus1_array = (1 - step_size) * y_array + step_size * v1_bar

    return y_tplus1_array

class MeanShift(IterativeKernelBasedQuantization):
    # IID Mean shift algorithm
    def __init__(self, params):
        super().__init__(params)
        self.algo_name = "IID Mean Shift"
        self.step_size = params.get("step_size", 1.0)

    def calculate_weights(self, y_array, t, w_array):
        x_array = self.data_array

        # Be careful because K means kernel matrix and number of centroids
        kernel = self.kernel_scheduler.GetKernel()
        return get_weights(kernel, y_array, x_array)

    def calculate_centroids(self, y_array, t, w_array):
        x_array = self.data_array

        # Get the kernel and prekernel functions
        kernel = self.kernel_scheduler.GetKernelInstance()

        return get_centroids(y_array, x_array, kernel.kernel_bar, self.step_size)