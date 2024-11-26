import numpy as np

class DataDistribution():
    def __init__(self, params):
        self.params = params

    def generate_samples(self, M, data):
        perm = np.random.permutation(self.N)
        return data[perm[:M]]