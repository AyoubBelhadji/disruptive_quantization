
import numpy as np
import matplotlib.pyplot as plt

class AbstractAlgorithm:
    def __init__(self, params):
        self.params = params

    def run(self, data):
        raise NotImplementedError("This method should be implemented by subclasses")

class MultipleMeanShift(AbstractAlgorithm):
    def run(self, data):
        step_size = self.params.get('step_size', 1)
        return np.round(data / step_size) * step_size

data = np.linspace(0, 10, 100)
algorithm = MultipleMeanShift(params={'K': 3, 'freeze_init': 1, 'R': 2, 'T': 100, 'd': 2, 'mean': [50, 50], 'covariance': [[1, 0], [0, 1]], 'initial_distribution': <functions.initial_distributions.gaussian_distribution.GaussianDistribution object at 0x160774490>, 'bandwidth': 6, 'kernel': <functions.kernels.gaussian_kernel.GaussianKernel object at 0x160f2b580>, 'noise_schedule_function': <functions.noise_generators.gaussian_sqrt_noise.GaussianSqrtNoise object at 0x160f2b940>, 'a': -25, 'b': 25, 'dataset_name': 'gmm_data', 'N': 1000})

result = algorithm.run(data)

plt.plot(result)
plt.title("pre_config Results")
plt.show()
        