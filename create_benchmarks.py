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

def get_eigenvalues(kernel_class, bandwidth, data, kernel_kwargs):
    kernel = kernel_class(bandwidth, **kernel_kwargs)
    K = broadcast_kernel(kernel.kernel, data, data)/len(data)
    return np.linalg.eigvalsh(K)

def produce_results(data, bandwidths, kernel_kwargs):
    all_eigenvalues = {}
    kernels = [("sqexp",GaussianKernel), ("matern",MaternKernel)]

    prog = tqdm(total=len(bandwidths)*len(kernels))
    for (kernel_name, kernel_class) in kernels:
        for bandwidth in bandwidths:
            eigenvalues = get_eigenvalues(kernel_class, bandwidth, data, kernel_kwargs)
            all_eigenvalues[(kernel_name, bandwidth)] = eigenvalues
            prog.update(1)
    return all_eigenvalues

def save_results(results, dataset, dataset_path = "datasets", file_name="eigenvalues.pkl"):
    file_path = os.path.join(dataset_path, dataset, file_name)
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            prev_results = pickle.load(f)
            new_results = results
            results = prev_results.copy()
            results.update(new_results)
    with open(file_path, 'wb') as f:
        pickle.dump(results, f)

def get_args():
    parser = argparse.ArgumentParser()
    # Assume we use all available kernels
    # only args are a list of bandwidths separated by commas
    parser.add_argument('dataset', type=str)
    parser.add_argument('bandwidths', type=str)
    # Add parser keywords for kernel kwargs at end of input
    parser.add_argument("kwargs", nargs="*")
    return parser.parse_args()

def parse_kwargs(kwargs):
    def parse_value(value):
        try:
            return float(value)
        except ValueError:
            return value
    return {key: parse_value(value) for key, value in [kwarg.split("=") for kwarg in kwargs]}

if __name__ == '__main__':
    args = get_args()
    data = read_dataset(args.dataset)
    kernel_kwargs = parse_kwargs(args.kwargs)
    bandwidths = [float(b) for b in args.bandwidths.split(",")]
    results = produce_results(data, bandwidths, kernel_kwargs)
    save_results(results, args.dataset)