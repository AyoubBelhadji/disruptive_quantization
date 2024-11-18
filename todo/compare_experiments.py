#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 25/25/2525 25:25:25
Also created on Sun Dec 10 12:26:17 2023
Also also created on Mon Nov 18 6:22:10 2024
@author: ayoubbelhadji
"""

import os
from experiment_manager import ExperimentManager

def load_experiment_id_mapping():
    # Load the experiment ID mapping from experiment_ids.txt
    mapping_file = 'experiment_ids.txt'
    experiment_choices = {}
    if os.path.exists(mapping_file):
        with open(mapping_file, 'r') as f:
            for line in f:
                try:
                    experiment_id, folder_path = line.strip().split(',', 1)
                    experiment_choices[int(experiment_id)] = folder_path
                except ValueError:
                    print(f"Skipping malformed line in {mapping_file}: {line}")
    else:
        print(f"Mapping file {mapping_file} not found.")
    return experiment_choices

def prompt_user_for_experiment_choice(experiment_choices):
    # Display available folders with integer IDs and prompt for selection
    print("\nAvailable experiments (Integer ID: folder path):")
    for experiment_id, folder in experiment_choices.items():
        print(f"{experiment_id}: {folder}")

    selected_ids = input("Enter integer experiment IDs to compare, separated by commas: ").split(',')
    selected_folders = [experiment_choices.get(int(experiment_id.strip())) for experiment_id in selected_ids if experiment_id.strip().isdigit() and int(experiment_id.strip()) in experiment_choices]

    if all(selected_folders):
        return selected_folders
    else:
        print("Invalid selection or some IDs not found. Please enter valid integer experiment IDs as listed above.")
        return None

# Load all experiments from the mapping file
all_experiments = load_experiment_id_mapping()

# Prompt user to select multiple experiments by integer ID
selected_experiments = prompt_user_for_experiment_choice(all_experiments)

if selected_experiments:
    # Initialize the ExperimentManager for loading and comparison
    experiment_manager = ExperimentManager()

    # Load the selected experiment folders for comparison
    experiment_manager.load_experiments(selected_experiments)

    # Compare results of the loaded experiments (show or save as PDF)
    experiment_manager.compare_results(save_pdf=True, pdf_file_path="experiments/comparison_results.pdf")
else:
    print("Comparison could not proceed due to invalid experiment selection.")
