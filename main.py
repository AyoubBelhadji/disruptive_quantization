#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 25/25/2525 25:25:25
Also created on Sun Dec 10 12:26:17 2023
Also also created on Mon Nov 18 6:22:10 2024
@author: ayoubbelhadji
"""


import os
import importlib
import numpy as np
import json
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# from algorithms.IKBQ.

from functions.kernels.gaussian_kernel import *
from functions.kernels.kernel_bandwidth_scheduler import *
from functions.initial_distributions.gaussian_distribution import *
from functions.initial_distributions.data_distribution import *
from functions.initial_distributions.kmeanspp_distribution import *
from functions.noise_generators.gaussian_sqrt_noise import *
from functions.time_parameterizations.time_parameterizations import *
from tools.files_tools import *
from tools.visualization_tools import *
from tools.simulation_manager import *


# Map function names to function objects
function_map = {
    "gaussian_distribution": GaussianDistribution,
    "gaussian_sqrt_noise": GaussianSqrtNoise,
    "gaussian_kernel": GaussianKernel,
    "data_distribution": DataDistribution,
    "kmeans++": KmeansPlusPlusDistribution,
    "constant_kernel_bandwidth": ConstantKernelBandwidth,
    "exponential_decay_kernel_bandwidth": ExponentialDecayKernelBandwidth,
    "linear_time_parameterization": LinearTimeParameterization,
    "logarithmic_time_parameterization": LogarithmicTimeParameterization
}

# Load available algorithms
algorithms = get_available_algorithms()
show_gif_visualization = True


# Main execution
if __name__ == "__main__":
    # Define the folder containing experiment configuration files
    config_folder = os.path.join(
        os.path.dirname(__file__), 'experiment_configs')

    # Initialize the ExperimentManager
    sim_manager = SimulationManager()

    # Iterate over all JSON files in the config folder
    for config_filename in os.listdir(config_folder):
        if config_filename.endswith('.json'):  # Ensure it's a JSON file
            config_path = os.path.join(config_folder, config_filename)

            # Load the configuration
            config = load_config(config_path)

            # Extract experiment details
            algorithm_name = config['algorithm_name']
            params = categorize_params(config, function_map)

            # Check if debugging
            debug = config.get('debug', False)

            # Initialize the data loader
            data_loader = DataLoader(datasets_folder='datasets')

            # Load the dataset
            dataset_name = params['dataset_name'] + '.pkl'
            try:
                data = data_loader.load_dataset(dataset_name)
                print(
                    f"Loaded dataset shape for {config_filename}: {data.shape}")
            except Exception as e:
                print(f"Failed to load dataset for {config_filename}: {e}")
                continue

            # Example metadata
            experiment_metadata = {
                "description": f"Experiment from {config_filename}"}

            # Initialize and run the algorithm
            try:
                rand_algo = initialize_algorithm(
                    algorithms, algorithm_name, params)
                rand_algo.run(data)

                print(
                    f"Successfully ran {algorithm_name} for {config_filename}")

                # Save the experiment results
                experiment_full_id = sim_manager.save_experiments(
                    # Use the filename (without extension) as the experiment name
                    experiment_name=config_filename.split('.')[0],
                    results_folder_base="experiments",
                    category="sandbox",
                    algorithm=rand_algo,
                    comment=f"Experiment based on {config_filename}"
                )

                # Visualize the dynamics using a gif
                if show_gif_visualization:
                    visualize_and_save_dynamics(
                        experiment_full_id, rand_algo.c_array_trajectory, rand_algo.data_array)

            except ValueError as e:
                print(
                    f"Error running {algorithm_name} for {config_filename}: {e}")
                if debug:
                    raise e
                continue
