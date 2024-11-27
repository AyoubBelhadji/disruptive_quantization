#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 10:03:00 2024

@author: dannys4
"""

import numpy as np
from abc import ABC, abstractmethod

class TimeParameterization(ABC):
    """
    Abstract class for time parameterizations in Wasserstein Fisher-Rao infinite-time mean-field ODE
    """
    def __init__(self):
        pass

    def SetLength(self, T):
        """ Set the total number of iterations """
        self.T = T
        self.setup()

    @abstractmethod
    def __call__(self, t):
        """ Return the time span for the ODE solver at iteration t """
        pass

    @abstractmethod
    def setup(self):
        """ Setup the time parameterization once the total number of iterations is known """
        pass


class LinearTimeParameterization(TimeParameterization):
    """
    Linear time parameterization. The time span for iteration t is [t*dt, (t+1)*dt]
    """

    def __init__(self, params):
        self.end_time = params.get('end_time')

    def setup(self):
        self.dt = self.end_time / (self.T-1)

    def __call__(self, t):
        """ Linear time parameterization """
        return (t*self.dt, (t+1)*self.dt)


class LogarithmicTimeParameterization(TimeParameterization):
    """
    Logarithmic time parameterization. Each time t is spaced logarithmically between 0 and end_time
    """

    def __init__(self, params):
        self.base = params.get('base', 10)
        assert self.base > 1, "Base must be greater than 1"
        self.start_time_log = np.emath.logn(
            self.base, params.get('start_time', 1e-4))
        assert not np.iscomplex(self.start_time_log), "Start time must be positive"
        self.end_time_log = np.emath.logn(self.base, params.get('end_time'))
        assert not np.iscomplex(self.end_time_log), "End time must be positive"

    def setup(self):
        self.time_points = np.concat(([0.], np.logspace(
            self.start_time_log, self.end_time_log, num=self.T, base=self.base)))

    def __call__(self, t):
        """ Logarithmic time parameterization """
        return (self.time_points[t], self.time_points[t+1])
