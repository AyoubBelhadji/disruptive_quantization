#!/usr/bin/env python
from tools.utils import broadcast_kernel
from functions.kernels import GaussianKernel, MaternKernel
from tqdm import tqdm
import numpy as np
import argparse
import os
import pickle

# Read the dataset
def read_dataset(dataset, datapath = "datasets"):
    file_path = os.path.join(datapath, dataset, "data.pkl")
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
        return data['data']

def get_eigenvalues(kernel_class, bandwidth, data):
    kernel = kernel_class(bandwidth)
    K = broadcast_kernel(kernel.kernel, data, data)
    return np.linalg.eigvalsh(K)

def produce_results(data, bandwidths):
    all_eigenvalues = {}
    kernels = [("sqexp",GaussianKernel), ("matern",MaternKernel)]

    prog = tqdm(total=len(bandwidths)*len(kernels))
    for (kernel_name, kernel_class) in kernels:
        for bandwidth in bandwidths:
            eigenvalues = get_eigenvalues(kernel_class, bandwidth, data)
            all_eigenvalues[(kernel_name, bandwidth)] = eigenvalues
            prog.update(1)
    return all_eigenvalues

def save_results(results, dataset, dataset_path = "datasets", file_name="eigenvalues.pkl"):
    file_path = os.path.join(dataset_path, dataset, file_name)
    if os.path.exists(file_path):
        with open(file_name, 'rb') as f:
            prev_results = pickle.load(f)
            results.update(prev_results)
    with open(file_name, 'wb') as f:
        pickle.dump(results, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Assume we use all available kernels
    # only args are a list of bandwidths separated by commas
    parser.add_argument('dataset', type=str)
    parser.add_argument('bandwidths', type=str)
    args = parser.parse_args()
    data = read_dataset(args.dataset)
    bandwidths = [float(b) for b in args.bandwidths.split(",")]
    results = produce_results(data, bandwidths)
    save_results(results, args.dataset)