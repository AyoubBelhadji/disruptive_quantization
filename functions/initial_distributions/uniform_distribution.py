import numpy as np

class UniformDistribution:
    def __init__(self, params, rng: np.random.Generator):
        """
        Initialize the UniformDistribution class with parameters.

        Parameters:
        - params: dict
            A dictionary containing 'LB' and 'UB' keys.
            - 'LB': ndarray of shape (d,)
                Vector of lower bounds for each dimension.
            - 'UB': ndarray of shape (d, d)
                Vector of upper bounds for each dimension.
        """
        self.d = params.get('d')
        self.LB = np.asarray(params.get('LB', np.zeros(self.d)))  # Default LB is a zero vector
        self.UB = np.asarray(params.get('UB', np.ones(self.d)))  # Default UB is a vector of ones
        assert np.all(self.LB <= self.UB), "Lower bounds must be less than or equal to upper bounds"
        self.rng = rng # Random number generator

    def generate_samples(self, M, *_):
        """
        Generate M random samples from the uniform distribution.

        Parameters:
        - M: int
            Number of samples to generate.

        Returns:
        - samples: ndarray of shape (M, d)
            Generated samples.
        """

        return self.rng.uniform(self.LB, self.UB, (M, self.d))

# Example Usage
if __name__ == '__main__':
    params = {
        'd': 2,
        'LB': np.array([50, 52]),
        'UB': np.array([60, 62])
    }
    dist = UniformDistribution(params, np.random.default_rng(1))
    samples = dist.generate_samples(100)
    print(samples)



