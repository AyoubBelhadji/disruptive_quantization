# Abstract class for additive sqrt noise generators

import numpy as np
from abc import ABC, abstractmethod

class AddSqrtNoiseGenerator(ABC):
    def __init__(self, params, rng: np.random.Generator):
        """
        Initialize the NoiseGenerator class with parameters.

        Parameters:
        - params: dict
            A dictionary containing 'd' and 'beta_ns' keys.
            - 'd': the dimension
            - 'beta_ns': the initial noise level
        """
        self.beta = params.get('beta_ns', 0.0)
        self.rng = rng

    def generate_noise(self, c_array: np.ndarray, t: float):
        """
        Generate noise and add it to the input array.

        Parameters:
        - c_array (numpy.ndarray): The input array with shape (M, d).
        - t (float): The time step or parameter affecting the noise scale.

        Returns:
        - numpy.ndarray: The input array with added noise.
        """
        if self.beta == 0.0:
            return c_array
        else:
            std = self.beta / np.sqrt(t + 1)
            noise = self.generate_noise_internal(c_array) * std
            return c_array + noise

    @abstractmethod
    def generate_noise_internal(self, c_array: np.ndarray):
        """
        Generate a noised version of c_array

        Parameters:
        - c_array (numpy.ndarray): The input array with shape (M, d).

        Returns:
        - numpy.ndarray: noise to add.
        """
        pass