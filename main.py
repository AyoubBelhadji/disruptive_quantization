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
from functions.kernels import (
    gaussian_kernel,
    matern_kernel,
    inverse_multiquadric_kernel,
)
from functions.kernels.kernel_bandwidth_scheduler import (
    ConstantKernelBandwidth,
    ExponentialDecayKernelBandwidth,
)
from functions.initial_distributions import (
    gaussian_distribution,
    data_distribution,
    kmeanspp_distribution,
)
from functions.noise_generators.gaussian_sqrt_noise import GaussianSqrtNoise
from functions.time_parameterizations.time_parameterizations import (
    LinearTimeParameterization,
    LogarithmicTimeParameterization,
)
from tools import files_tools, visualization_tools
from tools.simulation_manager import (
    SimulationManager,
    get_available_algorithms,
    initialize_algorithm,
)


# Map function names to function objects
function_map = {
    "gaussian_distribution": gaussian_distribution.GaussianDistribution,
    "gaussian_sqrt_noise": GaussianSqrtNoise,
    "gaussian_kernel": gaussian_kernel.GaussianKernel,
    "matern_kernel": matern_kernel.MaternKernel,
    "inverse_multiquadric_kernel": inverse_multiquadric_kernel.InverseMultiQuadricKernel,
    "data_distribution": data_distribution.DataDistribution,
    "kmeans++": kmeanspp_distribution.KmeansPlusPlusDistribution,
    "constant_kernel_bandwidth": ConstantKernelBandwidth,
    "exponential_decay_kernel_bandwidth": ExponentialDecayKernelBandwidth,
    "linear_time_parameterization": LinearTimeParameterization,
    "logarithmic_time_parameterization": LogarithmicTimeParameterization,
}

# Set up the argument parser
parser = argparse.ArgumentParser(description="Run quantization experiments")
parser.add_argument(
    "--no-viz",
    help="No visualization (default generates gif + MMD)",
    action="store_true",
)
parser.add_argument("-g", "--gif", help="Just visualize gif", action="store_true")
parser.add_argument("-m", "--mmd-viz", help="Just visualize mmd", action="store_true")
parser.add_argument(
    "-n",
    "--neighbor-viz",
    help="Just visualize nearest neighbors",
    action="store_true",
)
parser.add_argument(
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
    no_viz = args.no_viz
    just_gif = args.gif
    show_gif_visualization = not args.no_viz and not (args.mmd_viz or args.neighbor_viz)
    show_mmd_visualization = not args.no_viz and not (args.gif or args.neighbor_viz)
    show_nns_visualization = not args.no_viz and not (args.gif or args.mmd_viz)
    config_subdir = args.dir
    debug = args.debug

    # Load available algorithms
    algorithms = get_available_algorithms(debug=debug)

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

    # Initialize the ExperimentManager
    sim_manager = SimulationManager()

    # Iterate over all JSON files in the config folder
    for config_filename in os.listdir(config_folder):
        if config_filename.endswith(".json"):  # Ensure it's a JSON file
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

            # Example metadata
            experiment_metadata = {"description": f"Experiment from {config_filename}"}

            # Initialize and run the algorithm
            try:
                rand_algo = initialize_algorithm(algorithms, algorithm_name, params)
                rand_algo.run(data)

                print(f"Successfully ran {algorithm_name} for {config_filename}")

                # Save the experiment results
                experiment_full_id = sim_manager.save_experiments(
                    # Use the filename (without extension) as the experiment name
                    experiment_name=config_filename.split(".")[0],
                    results_folder_base="experiments",
                    experiment_subdir=output_subdir,
                    category="sandbox",
                    algorithm=rand_algo,
                    comment=f"Experiment based on {config_filename}",
                )

                subpath = os.path.join(output_subdir, experiment_full_id)
                c_array = rand_algo.c_array_trajectory
                w_array = rand_algo.w_array_trajectory

                # Visualize the dynamics using a gif
                if show_gif_visualization:
                    visualization_tools.centroid_dynamics(
                        algorithm_name,
                        c_array,
                        data,
                        subpath,
                    )

                if show_mmd_visualization:
                    if "kernel" in params:
                        test_kernel = params["kernel"].GetKernel()
                    else:
                        test_kernel_str = params.get("test_kernel", "gaussian_kernel")
                        test_kernel_bandwidth = params.get("test_kernel_bandwidth", 1.0)
                        test_kernel = function_map[test_kernel_str](
                            test_kernel_bandwidth
                        ).kernel
                    visualization_tools.evolution_weights_mmd(
                        algorithm_name,
                        c_array,
                        w_array,
                        data,
                        test_kernel,
                        subpath,
                    )

                if show_nns_visualization:
                    plot_path = os.path.join("figures", subpath, "plots")
                    files_tools.create_folder_if_needed(plot_path)
                    if labels is None:
                        print("No labels available for nearest neighbors visualization")
                        continue
                    nearest_neighbors_params = params.get("nearest_neighbors", {})

                    visualization_tools.nearest_neighbors(algorithm_name, c_array, w_array, data, labels, plot_path, **nearest_neighbors_params)

            except ValueError as e:
                print(f"Error running {algorithm_name} for {config_filename}: {e}")
                if debug:
                    raise e
                continue
