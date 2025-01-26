import os
import pickle
import fcntl

MMD_VALS_FILENAME = "self_mmd_vals.pkl"
LOCKFILE_PATH = os.environ.get("LOCKFILE_PATH", None)

if LOCKFILE_PATH is not None and not os.path.exists(LOCKFILE_PATH):
    parent_path = "/".join(LOCKFILE_PATH.split("/")[:-1])
    if not os.path.exists(parent_path):
        os.makedirs(parent_path, exist_ok=True)
    open(LOCKFILE_PATH, 'a').close()

def lock(kind, path):
    lock = fcntl.LOCK_SH if kind == 'sh' else fcntl.LOCK_EX
    with open(path, 'w') as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_SH)

def unlock(path):
    with open(path, 'w') as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)


class Self_MMD_Dict():
    def __init__(self, dataset, N, dataset_prefix = "datasets"):
        # Create dataset path
        dataset_path = os.path.join(dataset_prefix, dataset, MMD_VALS_FILENAME)
        self.path = dataset_path
        self.lockfile_path = LOCKFILE_PATH if LOCKFILE_PATH is not None else os.path.join("./.keep")
        # Create an instance correspondence to the file
        self.mmd_vals = None
        # Load mmd_vals.pkl if it exists
        if os.path.exists(dataset_path):
            lock('sh', self.lockfile_path)
            with open(dataset_path, 'rb') as f:
                self.mmd_vals = pickle.load(f)
            unlock(self.lockfile_path)
        else:
            self.mmd_vals = {}
            open(dataset_path, 'a').close()
        self.last_modified = os.path.getmtime(self.path)
        self.N = N

    def __getitem__(self, kernel):
        if os.path.getmtime(self.path) > self.last_modified:
            
            lock('sh', self.lockfile_path)
            with open(self.path, 'rb') as f:
                self.mmd_vals = pickle.load(f)
            unlock(self.lockfile_path)
            self.last_modified = os.path.getmtime(self.path)
        key = (self.N, kernel.get_key())
        return self.mmd_vals.get((self.N, key), None)

    def __setitem__(self, kernel, self_MMD):
        key = (self.N, kernel.get_key())

        if key not in self.mmd_vals:
            self.mmd_vals[(self.N,key)] = self_MMD
            lock('ex', self.lockfile_path)
            with open(self.path, 'wb') as f:
                pickle.dump(self.mmd_vals, f)
            unlock(self.lockfile_path)
