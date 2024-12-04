#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 25/25/2525 25:25:25
Also created on Sun Dec 10 12:26:17 2023
Also also created on Mon Nov 18 6:22:10 2024
@author: ayoubbelhadji
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from .files_tools import *
#from tqdm import tqdm

def visualize_and_save_dynamics(experiment_name, c_array_trajectory, data_array):
    R,T,M,d = c_array_trajectory.shape
    animations = []  # To store animations for each repetition
    
    for r in range(R):
        fig, ax = plt.subplots()
    
        # def update(frame, current_R):
        #     ax.clear()
            
        #     ax.set_xlim([-15, 28])
        #     ax.set_ylim([-15, 28])
            
        #     # Plot static positions
        #     ax.scatter(data_array[:, 0], data_array[:, 1], color='black', label='target distribution')
        #     # Plot trajectories for each particle at this frame
        #     for k in range(M):
        #         ax.scatter(c_array_trajectory[current_R, frame, k, 0], c_array_trajectory[current_R, frame, k, 1], color='red', 
        #                    label=f'Particle {k}' if frame == 0 else "")
        #     ax.set_title(f'Repetition {current_R + 1}, Time step {frame + 1}')
        #     ax.legend()
    
    
    
        def animate(t, current_R):
            ax.clear()
            
            centroids_0 = c_array_trajectory[r,0, :, :]
            ## Change the following: the windows size should be in the config file
            
            ax.set_xlim([-15, 28])
            ax.set_ylim([-15, 28])
            # Plot all data points
            ax.scatter(data_array[:, 0], data_array[:, 1], color='black', alpha=0.5)
            
            # Plot initial centroids
            ax.scatter(centroids_0[:, 0], centroids_0[:, 1], color='red', alpha=0.5, label='Initial Centroids')
            
            # Plot moving centroids for frame t
            ax.scatter(list(c_array_trajectory[current_R,t, :, 0]), list(c_array_trajectory[current_R,t, :, 1]), color='green', alpha=1, label='Centroids')
            
            ax.set_title('Iteration t='+str(t))
            
            #ax.set_xlabel('X1')
            #ax.set_ylabel('X2')
            ax.legend()
            
            
        # Animate for the first repetition (adjust if you want all repetitions)
        #current_R = 0  # Focus on the first repetition
        ani = FuncAnimation(fig, animate, frames=T, interval=200, fargs=(r,))
        #experiment_name = "experiment_name"
        
        folder_name = "figures/"+experiment_name+"/gif/"
        
        create_folder_if_needed(folder_name)
        
        # Save animation as a GIF
        gif_path = folder_name + "particle_evolution_"+str(r)+".gif"
        ani.save(gif_path, writer=PillowWriter(fps=10))
        plt.close(fig)  # Close the figure to avoid displaying static plots

    return gif_path




def rbf_kernel(x, y, bandwidth=1.0):
    """
    Computes the RBF kernel between two sets of points.
    
    Parameters:
    - x (numpy array): First set of points (N x d).
    - y (numpy array): Second set of points (M x d).
    - bandwidth (float): Bandwidth parameter for the RBF kernel.
    
    Returns:
    - kernel_matrix (numpy array): Kernel matrix (N x M).
    """
    x = np.expand_dims(x, axis=1)  # Shape: (N, 1, d)
    y = np.expand_dims(y, axis=0)  # Shape: (1, M, d)
    return np.exp(-np.sum((x - y) ** 2, axis=2) / (2 * bandwidth ** 2))


def compute_mmd(X, Y, kernel):
    """
    Computes the MMD between two datasets using the RBF kernel.

    Parameters:
    - X (numpy array): First dataset (N x d).
    - Y (numpy array): Second dataset (M x d).
    - bandwidth (float): Bandwidth parameter for the RBF kernel.

    Returns:
    - mmd (float): Maximum Mean Discrepancy value.
    """
    K_XX = kernel.kernel(X, X)
    K_YY = kernel.kernel(Y, Y)
    K_XY = kernel.kernel(X, Y)

    mmd = np.mean(K_XX) + np.mean(K_YY) - 2 * np.mean(K_XY)
    return mmd


def visualize_and_save_dynamics_with_mmd(experiment_name, c_array_trajectory, data_array, kernel):
    """
    Visualizes particle dynamics, saves GIF animations, and plots MMD over iterations.

    Parameters:
    - experiment_name (str): Name of the experiment for saving files.
    - c_array_trajectory (numpy array): Trajectory of centroids (R x T x M x d).
    - data_array (numpy array): Target distribution points (N x d).
    - bandwidth (float): Bandwidth parameter for the RBF kernel.
    
    Returns:
    - gif_paths (list): List of paths to saved GIFs.
    - mmd_plot_path (str): Path to the saved MMD plot.
    """
    R, T, M, d = c_array_trajectory.shape
    gif_paths = []  # To store paths of saved GIFs
    mmd_values = np.zeros((R, T))  # To store MMD values

    # Ensure output folders exist
    #gif_folder = f"figures/{experiment_name}/gif/"
    mmd_folder = f"figures/{experiment_name}/plots/"
    #os.makedirs(gif_folder, exist_ok=True)
    os.makedirs(mmd_folder, exist_ok=True)

    # Compute MMD for each repetition and time step
    for r in range(R):
        for t in range(T):
            mmd_values[r, t] = compute_mmd(data_array, c_array_trajectory[r, t], kernel)


    # Plot MMD over iterations for all repetitions
    plt.figure(figsize=(10, 6))
    for r in range(R):
        plt.plot(range(T), mmd_values[r], color='black')
    
    plt.title("MMD evolution over iterations")
    plt.xlabel("Iteration")
    plt.ylabel("MMD")
    plt.legend()
    plt.grid()

    mmd_plot_path = os.path.join(mmd_folder, "mmd_evolution.png")
    plt.savefig(mmd_plot_path)
    plt.close()

    return mmd_plot_path

