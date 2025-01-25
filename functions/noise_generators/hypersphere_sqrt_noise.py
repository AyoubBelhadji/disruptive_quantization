import numpy as np
from functions.noise_generators.add_sqrt_noise_generator import AddSqrtNoiseGenerator

class HypersphereSqrtNoise(AddSqrtNoiseGenerator):
    def __init__(self, params, rng: np.random.Generator):
        """
        Initialize the HypersphereSqrtNoise class with parameters.

        Parameters:
        - params: dict
            A dictionary containing 'mean' and 'covariance' keys.
            - 'd': the dimension
            - 'beta_ns': the initial noise level
        - rng: numpy.random.Generator
        """
        super().__init__(params, rng)
        self.d = params.get('d')

    def generate_noise_internal(self, c_array):
        unnorm_noise = self.rng.normal(size=(len(c_array), self.d))
        norms = np.linalg.norm(unnorm_noise, axis=1)
        return unnorm_noise / norms[:, np.newaxis]