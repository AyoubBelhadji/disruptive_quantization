#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 25/25/2525 25:25:25
Also created on Sun Dec 10 12:26:17 2023
Also also created on Mon Nov 18 6:22:10 2024
@author: ayoubbelhadji
"""

import os
import pickle
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from algorithms.base_algorithm import AbstractAlgorithm


def get_next_experiment_id():
    counter_file = "experiments/experiment_counter.txt"

    if os.path.exists(counter_file):
        with open(counter_file, "r") as f:
            current_id = int(f.read())
    else:
        current_id = 0

    next_id = current_id + 1
    with open(counter_file, "w") as f:
        f.write(str(next_id))

    return next_id

def templates_dir():
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), "templates")

def update_experiment_id_mapping(experiment_id, experiment_folder):
    # Update or create a mapping file for integer IDs to folder paths
    mapping_file = "experiments/experiment_ids.txt"
    with open(mapping_file, "a") as f:
        f.write(f"{experiment_id},{experiment_folder}\n")
    print(f"Experiment ID mapping updated: {experiment_id} -> {experiment_folder}")


class SimulationManager:
    def __init__(self):
        self.experiments = {}

    def run_simulation(self, data, algorithm, name, metadata=None):
        result = algorithm.run(data)
        if metadata is None:
            metadata = {
                "experiment_name": name,
                "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "data_shape": data.shape,
            }
        self.experiments[name] = {
            "data": result,
            "params": algorithm.params,
            "metadata": metadata,
        }
        return result

    def save_experiments(
        self,
        experiment_name="experiment",
        results_folder_base="results",
        category="sandbox",
        experiment_subdir=None,
        algorithm: AbstractAlgorithm = None,
        python_file_name=None,
        comment=None,
    ):
        # Allow the user to specify if the experiment is "sandbox" or "validated"
        results_folder_base = self.create_results_folder(
            results_folder_base, category, experiment_subdir
        )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_id, experiment_folder = self.store_experiment_data(
            experiment_name, results_folder_base, timestamp
        )

        if algorithm:
            self.save_experiment_code(
                experiment_folder,
                experiment_name,
                algorithm.__class__.__name__,
                algorithm.params,
                python_file_name,
            )

        # Update the experiment ID mapping file
        update_experiment_id_mapping(experiment_id, experiment_folder)

        self.update_research_log(
            experiment_id,
            experiment_name,
            category,
            algorithm.__class__.__name__,
            algorithm.params,
            python_file_name or "experiment_run.py",
            experiment_folder,
            comment,
        )

        return experiment_name + "_" + str(experiment_id) + "_" + str(timestamp)

    def store_experiment_data(self, experiment_name, results_folder_base, timestamp):
        experiment_id = get_next_experiment_id()  # Get the next integer ID
        experiment_folder = os.path.join(
            results_folder_base, f"{experiment_name}_{experiment_id}_{timestamp}"
        )
        os.makedirs(experiment_folder, exist_ok=True)

        pkl_file_path = os.path.join(
            experiment_folder, "experiment_data_with_metadata.pkl"
        )
        npy_file_path = os.path.join(
            experiment_folder, "experiment_data_with_metadata.npy"
        )
        with open(pkl_file_path, "wb") as f:
            pickle.dump(self.experiments, f)
        print(f"Experiments saved in {pkl_file_path}")

        np.save(npy_file_path, self.experiments)
        print(f"Experiments saved in {npy_file_path}")
        return experiment_id, experiment_folder

    def create_results_folder(self, results_folder_base, category, experiment_subdir):
        results_folder_base = os.path.join(results_folder_base, category)
        if experiment_subdir:
            results_folder_base = os.path.join(results_folder_base, experiment_subdir)
        os.makedirs(results_folder_base, exist_ok=True)
        return results_folder_base

    def save_experiment_code(
        self,
        experiment_folder,
        experiment_name,
        algorithm_name,
        params,
        python_file_name=None,
    ):
        if python_file_name is None:
            python_file_name = "experiment_run.py"

        # Read templates/exp_code_template.py as string and format it
        python_code = (
            open(os.path.join(templates_dir(),"exp_code_template.py"))
            .read()
            .format(
                algorithm_name=algorithm_name,
                params=params,
                experiment_name=experiment_name,
            )
        )

        python_file_path = os.path.join(experiment_folder, python_file_name)
        with open(python_file_path, "w") as f:
            f.write(python_code)

        print(f"Experiment rerun code saved in {python_file_path}")

    def update_research_log(
        self,
        experiment_id,
        experiment_name,
        category,
        algorithm_name,
        params,
        python_file_name,
        experiment_folder,
        comment=None,
    ):
        individual_log_file = os.path.join(experiment_folder, "research_log.txt")
        global_log_file = os.path.join("experiments", "global_research_log.txt")

        # Ensure 'experiments' directory exists
        os.makedirs("experiments", exist_ok=True)

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        log_entry = (
            open(os.path.join(templates_dir(), "research_log_entry_template.txt"))
            .read()
            .format(
                experiment_id=experiment_id,
                experiment_name=experiment_name,
                category=category,
                algorithm_name=algorithm_name,
                params=params,
                timestamp=timestamp,
                python_file_name=python_file_name,
                comment=comment or "No comment",
            )
        )

        with open(individual_log_file, "a") as f:
            f.write(log_entry)

        with open(global_log_file, "a") as f:
            f.write(log_entry)

        print(f"Research log updated: {individual_log_file} and {global_log_file}")

    def compare_results(self, save_pdf=False, pdf_file_path=None):
        fig, axs = plt.subplots(
            len(self.experiments), 1, figsize=(10, 5 * len(self.experiments))
        )
        if len(self.experiments) == 1:
            axs = [axs]

        for i, (name, experiment) in enumerate(self.experiments.items()):
            result = experiment["data"]
            params = experiment["params"]
            metadata = experiment["metadata"]
            axs[i].plot(result)
            axs[i].set_title(
                f"{metadata['experiment_name']} (params: {params}, date: {metadata['date']})"
            )

        plt.tight_layout()
        if save_pdf and pdf_file_path:
            plt.savefig(pdf_file_path)
        else:
            plt.show()
