import os
import pickle

MMD_VALS_FILENAME = "self_mmd_vals.pkl"
class Self_MMD_Dict():
    def __init__(self, dataset, N, dataset_prefix = "datasets"):
        # Create dataset path
        dataset_path = os.path.join(dataset_prefix, dataset, MMD_VALS_FILENAME)
        self.path = dataset_path
        # Create an instance correspondence to the file
        self.mmd_vals = None
        # Load mmd_vals.pkl if it exists
        if os.path.exists(dataset_path):
            with open(dataset_path, 'rb') as f:
                self.mmd_vals = pickle.load(f)
        else:
            self.mmd_vals = {}
        self.N = N

    def __getitem__(self, kernel):
        key = (self.N, kernel.get_key())
        return self.mmd_vals.get((self.N, key), None)

    def __setitem__(self, kernel, self_MMD):
        key = (self.N, kernel.get_key())

        if key not in self.mmd_vals:
            self.mmd_vals[(self.N,key)] = self_MMD
            with open(self.path, 'wb') as f:
                pickle.dump(self.mmd_vals, f)
