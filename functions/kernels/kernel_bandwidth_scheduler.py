#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 11/22/2024 17:40:25
@author: dannys4
"""

from abc import ABC, abstractmethod
import numpy as np

def KernelBandwidthScheduleFactory(kernel_info, function_map):
    """
    Factory method for creating kernel bandwidth scheduler objects.
    """
    kernel_name = kernel_info.get('kernel_name')
    kernel_class = function_map[kernel_name]
    kernel_params = kernel_info.get('params')
    schedule_function = kernel_params.get("bandwidth_schedule_function")

    if schedule_function is None:
        return ConstantKernelBandwidth(kernel_params, kernel_class)

    kernel_schedule_constructor = function_map[schedule_function.get("bandwidth_schedule_function_name")]
    schedule_params = schedule_function.get("params")
    return kernel_schedule_constructor(schedule_params, kernel_class)

class KernelBandwidthScheduler(ABC):
    """
    Abstract class for kernel bandwidth scheduler. Subclasses must implement the get_bandwidth method.
    """
    def __init__(self, _, kernel_constructor):
        # Get the kernel function from params
        self.iter = 0
        self.kernel_constructor = kernel_constructor

    def IncrementSchedule(self):
        """ Change kernel according to the current iteration """
        bandwidth = self.get_bandwidth()
        self.KernelConstructor(bandwidth)
        self.iter += 1

    def KernelConstructor(self, bandwidth):
        self.kernel_inst = self.kernel_constructor(bandwidth)

    def GetKernel(self):
        return self.kernel_inst.kernel

    def GetPreKernel(self):
        return self.kernel_inst.pre_kernel

    def GetKernelGrad2(self):
        return self.kernel_inst.kernel_grad2

    # Abstract methods
    @abstractmethod
    def get_bandwidth(self):
        """ Returns the bandwidth for the current iteration """
        assert False
        pass


class ConstantKernelBandwidth(KernelBandwidthScheduler):
    """ Constant kernel bandwidth """
    def __init__(self, params, *args):
        super().__init__(params, *args)
        self.bandwidth = params.get("bandwidth")
        self.KernelConstructor(self.bandwidth)

    def IncrementSchedule(self):
        pass

    def get_bandwidth(self):
        return self.bandwidth


class ExponentialDecayKernelBandwidth(KernelBandwidthScheduler):
    """
    Exponential decay kernel bandwidth scheduler:
    $$s = s_inf + (s_0 - s_{inf}) exp(a t)$$
    """
    def __init__(self, params, *args):
        super().__init__(params, *args)
        self.decay_rate = params.get("decay_rate")
        assert self.decay_rate < 0, "Decay rate must be strictly negative"
        self.start_value = params.get("start_value")
        self.end_value = params.get("end_value")

    def get_bandwidth(self):
        return self.end_value + (self.start_value - self.end_value) * np.exp(self.decay_rate * self.iter)
