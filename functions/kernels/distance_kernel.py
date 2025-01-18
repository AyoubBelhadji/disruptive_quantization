# Abstract class for distance kernels
import numpy as np
import numba as nb

@nb.jit()
def L2_distance(x, y):
    return np.sqrt(np.sum((x - y) ** 2, axis=-1))

class L2DistanceKernel():
    """Abstract class for distance kernels."""

    def __init__(self, bandwidth):
        self.sigma = bandwidth
        self.use_kernel_bar = True
        univariate_kernel = self.univariate_kernel
        univariate_kernel_diff = self.univariate_kernel_diff
        self.kernel = self.KernelConstructor(bandwidth, univariate_kernel)
        self.kernel_bar = self.KernelBarConstructor(bandwidth, univariate_kernel_diff)
        self.kernel_grad = self.KernelGrad2Constructor(bandwidth, univariate_kernel_diff)
        self.log_kernel = self.PrekernelConstructor(bandwidth, univariate_kernel_diff)

    def get_key(self):
        """Return a unique key for the kernel."""
        pass

    def KernelConstructor(self, sigma, univariate_kernel):
        """ Construct kernel evaluation """
        @nb.jit()
        def kernel_aux(x, y):
            return univariate_kernel(L2_distance(x, y)/sigma)
        return kernel_aux

    def KernelGrad2Constructor(self, sigma, univariate_kernel_diff):
        """ Construct kernel gradient evaluation """
        @nb.jit()
        def kernel_aux(x, y):
            dist = L2_distance(x, y)
            return (x - y) * univariate_kernel_diff(dist/sigma)/(sigma*dist)
        return kernel_aux

    def KernelBarConstructor(self, sigma, univariate_kernel):
        """ Construct kernel bar evaluation """
        @nb.jit()
        def kernel_aux(x, y):
            dist = L2_distance(x, y)
            return univariate_kernel(dist/sigma)/(sigma*dist)
        return kernel_aux

    def PrekernelConstructor(self, sigma, log_univariate_kernel):
        """ Construct log-kernel evaluation """
        print("Constructing log kernel")
        @nb.jit()
        def kernel_aux(x, y):
            return log_univariate_kernel(L2_distance(x, y)/sigma)
        return kernel_aux