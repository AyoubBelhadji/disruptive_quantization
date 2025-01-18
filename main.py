#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 25/25/2525 25:25:25
Also created on Sun Dec 10 12:26:17 2023
Also also created on Mon Nov 18 6:22:10 2024
@author: ayoubbelhadji
"""

import os
import argparse

# Import relevant functions
from functions import kernels, initial_distributions, noise_generators
from functions.time_parameterizations import (
    LinearTimeParameterization,
    LogarithmicTimeParameterization,
)
from tools import files_tools, visualization_tools, AlgorithmManager
from tools.simulation_manager import SimulationManager


# Map function names to function objects
function_map = {
    "gaussian_distribution": initial_distributions.GaussianDistribution,
    "uniform_distribution": initial_distributions.UniformDistribution,
    "gaussian_sqrt_noise": noise_generators.GaussianSqrtNoise,
    "hypersphere_sqrt_noise": noise_generators.HypersphereSqrtNoise,
    "gaussian_kernel": kernels.GaussianKernel,
    "matern_kernel": kernels.MaternKernel,
    "inverse_multiquadric_kernel": kernels.InverseMultiQuadricKernel,
    "data_distribution": initial_distributions.DataDistribution,
    "kmeans++": initial_distributions.KmeansPlusPlusDistribution,
    "constant_kernel_bandwidth": kernels.ConstantKernelBandwidth,
    "exponential_decay_kernel_bandwidth": kernels.ExponentialDecayKernelBandwidth,
    "linear_time_parameterization": LinearTimeParameterization,
    "logarithmic_time_parameterization": LogarithmicTimeParameterization,
}


def resolve_config_path(config_subdir):
    # Define the folder containing experiment configuration files
    config_folder = os.path.join(os.path.dirname(__file__), config_subdir)
    output_subdir = config_subdir if config_subdir != "examples" else ""

    # If the folder does not exist, try the experiment_configs folder
    if not os.path.isdir(config_folder):
        config_folder = os.path.join(
            os.path.dirname(__file__), "experiment_configs", config_subdir
        )

    if not os.path.isdir(config_folder):
        raise FileNotFoundError(
            f"Could not find the experiment configuration folder {config_folder}"
        )

    return config_folder, output_subdir


def load_algorithm_configuration(debug, config_folder, config_filename):
    config_path = os.path.join(config_folder, config_filename)

    # Load the configuration
    config = files_tools.load_config(config_path)

    # Extract experiment details
    algorithm_name = config["algorithm_name"]
    params = files_tools.categorize_params(config, function_map)

    # Initialize the data loader
    data_loader = files_tools.DataLoader(datasets_folder="datasets")

    # Load the dataset
    data, labels = data_loader.get_data(
        params["dataset_name"], params.get("N", 0), debug
    )

    params["N"] = data.shape[0]

    return algorithm_name, params, data, labels


def create_visualizations(
    algorithm_name,
    c_array,
    w_array,
    data,
    labels,
    params,
    subpath,
    show_gif,
    show_mmd,
    show_nns,
):
    # Visualize the dynamics using a gif
    if show_gif:
        visualization_tools.centroid_dynamics(
            algorithm_name,
            c_array,
            data,
            subpath,
        )

    if show_mmd:
        if "kernel" in params:
            test_kernel = params["kernel"].GetKernelInstance()
        else:
            test_kernel_str = params.get("test_kernel", "gaussian_kernel")
            test_kernel_bandwidth = params.get("test_kernel_bandwidth", 1.0)
            test_kernel_fcn = function_map[test_kernel_str]
            test_kernel = test_kernel_fcn(test_kernel_bandwidth)

        visualization_tools.evolution_weights_mmd(
            algorithm_name,
            c_array,
            w_array,
            data,
            test_kernel,
            params["dataset_name"],
            subpath,
        )

    if show_nns:
        plot_path = os.path.join("figures", subpath, "plots")
        files_tools.create_folder_if_needed(plot_path)
        if labels is not None:
            nearest_neighbors_params = params.get("nearest_neighbors", {})

            visualization_tools.nearest_neighbors(
                c_array,
                w_array,
                data,
                labels,
                algorithm_name,
                plot_path,
                **nearest_neighbors_params,
            )
        else:
            print("No labels available for nearest neighbors visualization")


def main(show_gif, show_mmd, show_nns, config_subdir, debug):
    # Load available algorithms
    algorithm_manager = AlgorithmManager(debug=debug)

    config_folder, output_subdir = resolve_config_path(config_subdir)

    sim_manager = SimulationManager()

    # Iterate over all JSON files in the config folder
    for config_filename in os.listdir(config_folder):
        if not config_filename.endswith(".json"):  # Ensure it's a JSON file
            continue

        algorithm_name, params, data, labels = load_algorithm_configuration(
            debug, config_folder, config_filename
        )
        try:
            # Initialize and run the algorithm
            algorithm = algorithm_manager.get_algorithm(algorithm_name, params)
            # algorithm.run(data)
            sim_manager.run_simulation(data, algorithm, algorithm_name)
            # Use the filename (without extension) as the experiment name
            experiment_name = ".".join(config_filename.split(".")[:-1])
            comment = f"Experiment based on {config_filename}"

            print(f"Successfully ran {algorithm_name} for {config_filename}")

            # Save the experiment results
            experiment_full_id = sim_manager.save_last_experiment(
                algorithm,
                experiment_name,
                comment,
                experiment_subdir=output_subdir,
            )

            subpath = os.path.join(output_subdir, experiment_full_id)
            c_array = algorithm.c_array_trajectory
            w_array = algorithm.w_array_trajectory

            create_visualizations(
                algorithm_name,
                c_array,
                w_array,
                data,
                labels,
                params,
                subpath,
                show_gif,
                show_mmd,
                show_nns,
            )
        except ValueError as e:
            print(f"Error running {algorithm_name} for {config_filename}: {e}")
            if debug:
                raise e
            continue


# Set up the argument parser
parser = argparse.ArgumentParser(description="Run quantization experiments")
parser.add_argument("-g", "--gif", help="Visualize centroids", action="store_true")
parser.add_argument("-m", "--mmd", help="Visualize MMD evolution", action="store_true")
parser.add_argument(
    "-n",
    "--neighbors",
    help="Visualize nearest neighbors",
    action="store_true",
)
parser.add_argument(
    "-d",
    "--dir",
    help="Configuration subdirectory in ./ or ./experiment_configs",
    type=str,
    default="examples",
)
parser.add_argument("--debug", help="Turn on debug mode", action="store_true")

# Main execution
if __name__ == "__main__":
    # Parse the arguments
    args = parser.parse_args()
    show_gif_visualization = args.gif
    show_mmd_visualization = args.mmd
    show_nns_visualization = args.neighbors
    config_subdir = args.dir
    debug = args.debug

    main(
        show_gif_visualization,
        show_mmd_visualization,
        show_nns_visualization,
        config_subdir,
        debug,
    )
