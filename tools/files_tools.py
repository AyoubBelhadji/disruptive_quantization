#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import pickle
from functions.kernels.kernel_bandwidth_scheduler import KernelBandwidthScheduleFactory
import numpy as np

def create_folder_if_needed(folder_path):
    """
    Check if a folder exists, and create it if it doesn't.

    Parameters:
    folder_path (str): Path to the folder to check/create.
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder created: {folder_path}")
    else:
        print(f"Folder already exists: {folder_path}")


class DataLoader:
    def __init__(self, datasets_folder='datasets'):
        """
        Initializes the data loader.

        Args:
            datasets_folder (str): Path to the folder containing datasets.
        """
        self.datasets_folder = datasets_folder

    def load_dataset(self, dataset_name, datafile_name = "data.pkl", data_component='data', label_component="labels"):
        """
        Loads a dataset stored in Pickle format and extracts the specified component.

        Args:
            dataset_name (str): Name of the dataset file (e.g., "dataset.pkl").
            data_component (str): Dictionary key to extract the data from the dataset.
            label_component (str): Dictionary key to extract the labels of data from the dataset.

        Returns:
            np.ndarray or dict: The requested component of the dataset.

        Raises:
            FileNotFoundError: If the dataset file is not found.
            ValueError: If the dataset cannot be loaded or the component is invalid.
            KeyError: If the specified component is not found in the dataset dictionary.
        """
        dataset_path = os.path.join(self.datasets_folder, dataset_name, datafile_name)
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(
                f"Dataset '{dataset_name}' not found in '{self.datasets_folder}'")
        print(f"Loading dataset '{dataset_name}' from {dataset_path}: component {data_component}...")
        try:
            with open(dataset_path, 'rb') as f:
                data_dict = pickle.load(f)

                # Extract the requested component
                if data_component not in data_dict:
                    raise KeyError(
                        f"Component '{data_component}' not found in the dataset.")
                data = data_dict[data_component]
                labels = data_dict.get(label_component, None)
                return data, labels

        except Exception as e:
            raise ValueError(f"Error loading dataset '{dataset_name}': {e}")

    def get_data(self, dataset_name, N, debug):
        try:
            data, labels = self.load_dataset(dataset_name)
            print(f"Loaded dataset shape for {dataset_name}: {data.shape}")
            if N > 0:
                print(f"Using subset of size {N}")
                data = data[:N]
        except Exception as e:
            print(f"Failed to load dataset for {dataset_name}: {e}")
            if debug:
                raise e
            else:
                return None
        return data, labels

def load_config(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Configuration file not found: {file_path}")
    with open(file_path, 'r') as f:
        config = json.load(f)
    return config


def categorize_params(config, function_map):
    """Flatten parameters from config into a single-level dictionary without text descriptions."""
    params = {}

    # Use a random number generator with seed if provided
    rng_seed = config.get('rng_seed',None)
    params['rng'] = np.random.default_rng(rng_seed)
    params['rng_seed'] = rng_seed
    init_rng_seed = config.get('initialization_seed', None)
    init_rng = params['rng'] if init_rng_seed is None else np.random.default_rng(init_rng_seed)
    params['initialization_rng'] = init_rng

    # Load any constant hyperparameters
    hyperparams = config['params'].get('hyperparams', {})
    params.update(hyperparams)

    # Load solution parameters and add to params
    solution_params = config['params'].get('solution', {})
    params.update(solution_params)

    # Load experience parameters and add to params
    experience_params = config['params'].get('experience', {})
    params.update(experience_params)

    # Process optimization parameters, including noise_schedule_functions_list
    optimization_params = config['params'].get('optimization', {})
    params["T"] = optimization_params.get("T", None)
    params["step_size"] = optimization_params.get("step_size", None)

    # Process and flatten intial distribution parameters directly
    initial_distribution_info = config['params'].get(
        'initial_distribution', {})
    initial_distribution_name = initial_distribution_info.get(
        'distribution_name')
    initial_distribution_class = function_map[initial_distribution_name]
    intial_distribution_params = initial_distribution_info.get('params', {})

    # Add 'bandwidth' directly to params
    params.update(intial_distribution_params)

    params['initial_distribution'] = initial_distribution_class(
        intial_distribution_params, params.get('initialization_rng'))

    kernel_info = config['params'].get('kernel', {})
    kernel_params = kernel_info.get('params', {})
    if len(kernel_info) > 0:
        add_kernel_params, params['kernel'] = KernelBandwidthScheduleFactory(
            kernel_info.get('kernel_name'), kernel_params, function_map)
        params.update(add_kernel_params)

    # Parse noise_schedule_functions_list as a list of function names without parameters
    noise_schedule_function_info = optimization_params.get(
        'noise_schedule_function', {})
    if len(noise_schedule_function_info) > 0:
        noise_schedule_function_name = noise_schedule_function_info.get(
            'noise_schedule_function_name')
        noise_schedule_function_class = function_map[noise_schedule_function_name]
        noise_schedule_function_params = noise_schedule_function_info.get(
            'params', {})
        params.update(noise_schedule_function_params)

        params['noise_schedule_function'] = noise_schedule_function_class(
            noise_schedule_function_params, params['rng'])

    # Process and flatten domain parameters directly
    domain_info = config['params'].get('domain', {})
    if len(domain_info) > 0:
        domain_params = domain_info.get("params", {})
        params.update(domain_params)  # Add 'd', 'a', 'b' directly to params

    # Process and flatten any time parameterization parameters directly
    time_parameterization_info = config['params'].get('time_parameterization', {})
    if len(time_parameterization_info) > 0:
        time_parameterization_params = time_parameterization_info.get('params', {})
        time_parameterization_name = time_parameterization_info.get('time_discretization_name', 'linear_time_parameterization')
        time_parameterization_class = function_map[time_parameterization_name]
        params.update(time_parameterization_params)

        params['time_parameterization'] = time_parameterization_class(
            time_parameterization_params)

    # Dataset parameters
    dataset_params = config['params'].get('dataset', {})
    params.update(dataset_params)

    return params
