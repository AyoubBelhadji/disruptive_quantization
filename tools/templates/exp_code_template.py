import numpy as np
import matplotlib.pyplot as plt

class AbstractAlgorithm:
    def __init__(self, params):
        self.params = params

    def run(self, data):
        raise NotImplementedError("This method should be implemented by subclasses")

class {algorithm_name}(AbstractAlgorithm):
    def run(self, data):
        step_size = self.params.get('step_size', 1)
        return np.round(data / step_size) * step_size

data = np.linspace(0, 10, 100)
algorithm = {algorithm_name}(params={params})

result = algorithm.run(data)

plt.plot(result)
plt.title("{experiment_name} Results")
plt.show()