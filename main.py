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
from tools import files_tools, visualization_tools, AlgorithmManager, metrics
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
    plot_gif,
    plot_mmd,
    plot_nns,
    show_plots,
):
    if plot_gif or plot_mmd or plot_nns:
        print(f"Creating visualizations in figures/{subpath}")
    # Visualize the dynamics using a gif
    if plot_gif:
        visualization_tools.centroid_dynamics(
            algorithm_name,
            c_array,
            data,
            subpath,
        )

    if plot_mmd:
        test_kernel = params["test_kernel"]

        visualization_tools.evolution_weights_mmd(
            algorithm_name,
            c_array,
            w_array,
            data,
            test_kernel,
            params["dataset_name"],
            subpath,
            show_plots,
        )

    if plot_nns:
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
                show_plots,
                **nearest_neighbors_params,
            )
        else:
            print("No labels available for nearest neighbors visualization")


def create_results(
    algorithm_name,
    y_array,
    w_array,
    data,
    params,
    subpath,
):
    # Calculate the MMD values, Voronoi MSE, Hausdorff, log determinant distance
    mmd_folder_serial = os.path.join("experiments", "sandbox", subpath)
    print("Saving results in ", mmd_folder_serial)
    os.makedirs(mmd_folder_serial, exist_ok=True)
    mmd_self = metrics.Self_MMD_Dict(params["dataset_name"], data.shape[0])
    kernel = params["test_kernel"]
    metrics.calculate_all_metrics(
        algorithm_name, y_array, w_array, data, kernel, mmd_self, subpath
    )


def main(plot_gif, plot_mmd, plot_nns, show_plots, config_subdir, calc_results, debug):
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
            sim_manager.run_simulation(data, algorithm, algorithm_name, config_filename)
            # Use the filename (without extension) as the experiment name
            experiment_name = ".".join(config_filename.split(".")[:-1])
            comment = f"Experiment based on {config_filename}"

            # Save the experiment results
            experiment_full_id = sim_manager.save_last_experiment(
                algorithm,
                experiment_name,
                comment,
                experiment_subdir=output_subdir,
            )

            subpath = os.path.join(output_subdir, experiment_full_id)
            c_array = algorithm.y_trajectory
            w_array = algorithm.w_trajectory

            if "kernel" in params:
                test_kernel = params["kernel"].GetKernelInstance()
            else:
                test_kernel_str = params.get("test_kernel", "gaussian_kernel")
                test_kernel_bandwidth = params.get("test_kernel_bandwidth", 1.0)
                test_kernel_kwargs = [(key,params[key]) for key in params.keys() if key.startswith("test_kernel")]
                def val_fcn(val):
                    try:
                        return float(val)
                    except ValueError:
                        return val
                start_idx = len("test_kernel_")
                test_kernel_kwargs = {k[0][start_idx:]:val_fcn(k[1]) for k in test_kernel_kwargs}
                test_kernel_kwargs.pop("bandwidth", None)
                test_kernel_fcn = function_map[test_kernel_str]
                test_kernel = test_kernel_fcn(test_kernel_bandwidth, **test_kernel_kwargs)
            params["test_kernel"] = test_kernel

            if calc_results:
                create_results(
                    algorithm_name,
                    c_array,
                    w_array,
                    data,
                    params,
                    subpath,
                )

            create_visualizations(
                algorithm_name,
                c_array,
                w_array,
                data,
                labels,
                params,
                subpath,
                plot_gif,
                plot_mmd,
                plot_nns,
                show_plots,
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
    "-n", "--neighbors", help="Visualize nearest neighbors", action="store_true"
)
parser.add_argument("-p", "--plots", help="Show plots", action="store_true")
parser.add_argument(
    "-d",
    "--dir",
    help="Configuration subdirectory in ./ or ./experiment_configs",
    type=str,
    default="joker",
)
parser.add_argument("-r", "--results", help="Calculate results", action="store_true")
parser.add_argument("--debug", help="Turn on debug mode", action="store_true")

# Main execution
if __name__ == "__main__":
    # Parse the arguments
    args = parser.parse_args()
    plot_gif = True
    #args.gif
    plot_mmd = args.mmd
    plot_nns = args.neighbors
    show_plots = args.plots
    config_subdir = args.dir
    calc_results = args.results
    debug = args.debug
    if debug:
        import faulthandler
        faulthandler.enable()

    main(
        plot_gif,
        plot_mmd,
        plot_nns,
        show_plots,
        config_subdir,
        calc_results,
        debug,
    )
