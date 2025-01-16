import os
import importlib

from algorithms.base_algorithm import AbstractAlgorithm

class AlgorithmManager:
    def __init__(self, directory='algorithms', debug=False):
        """Dynamically load algorithm classes from the specified directory and its subdirectories."""
        algorithms = {}

        # Recursively walk through directories
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith('.py') and file not in ['__init__.py', 'base_algorithm.py', 'sub_algorithm.py']:
                    # Construct the module path relative to the base directory
                    relative_path = os.path.relpath(
                        root, directory).replace(os.sep, '.')
                    module_name = file[:-3]  # Remove the '.py' extension
                    if relative_path == '.':
                        full_module_name = f"{directory}.{module_name}"
                    else:
                        full_module_name = f"{directory}.{relative_path}.{module_name}"
                    if debug:
                        print(f'{file}, {module_name}, {relative_path}, {full_module_name}')
                    # Import the module
                    try:
                        module = importlib.import_module(full_module_name)

                        # Generate the expected class name in PascalCase
                        class_name = ''.join([part.capitalize()
                                            for part in module_name.split('_')])

                        # Retrieve the class from the module
                        algorithm_class = getattr(module, class_name, None)
                        if algorithm_class and isinstance(algorithm_class, type):
                            algorithms[class_name] = algorithm_class
                    except ModuleNotFoundError as e:
                        print(f"ModuleNotFoundError for {full_module_name}: {e}")
                    except Exception as e:
                        print(f"Error importing {full_module_name}: {e}")

        self.algorithms = algorithms

    # Load and initialize algorithm
    def get_algorithm(self, algorithm_name, params) -> AbstractAlgorithm:
        if algorithm_name in self.algorithms:
            return self.algorithms[algorithm_name](params=params)
        else:
            raise ValueError(
                f"Algorithm '{algorithm_name}' not found in the algorithms dictionary.")
