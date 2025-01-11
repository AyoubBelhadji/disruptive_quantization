import numpy as np

class DataDistribution():
    def __init__(self, params, rng: np.random.Generator):
        self.params = params
        self.rng = rng

    def generate_samples(self, M, data):
        perm = self.rng.permutation(data.shape[0])
        return data[perm[:M]]