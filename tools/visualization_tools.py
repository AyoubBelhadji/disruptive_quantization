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
import tools.mmd_tools as mmd_tools
import numba as nb

from .files_tools import *
#from tqdm import tqdm
def expand_limits(min, max, factor):
    """
    Expand the limits of a plot by a factor.
    """
    delta = max - min
    return min - factor * delta, max + factor * delta

def visualize_and_save_dynamics(alg_name, experiment_name, c_array_trajectory, data_array, config_folder = "", limit_margin=0.1):
    R,T,M,d = c_array_trajectory.shape
    animations = []  # To store animations for each repetition
    xlims = expand_limits(np.min(data_array[:, 0]), np.max(data_array[:, 0]), limit_margin)
    ylims = expand_limits(np.min(data_array[:, 1]), np.max(data_array[:, 1]), limit_margin)
    for r in range(R):
        fig, ax = plt.subplots()

        def animate(t, current_R):
            ax.clear()

            centroids_0 = c_array_trajectory[r,0, :, :]
            ## Change the following: the windows size should be in the config file

            ax.set_xlim(xlims)
            ax.set_ylim(ylims)
            # Plot all data points
            ax.scatter(data_array[:, 0], data_array[:, 1], color='black', alpha=0.5)

            # Plot initial centroids
            ax.scatter(centroids_0[:, 0], centroids_0[:, 1], color='red', label='Initial Centroids', alpha=1, marker='D')

            # Plot moving centroids for frame t
            ax.scatter(list(c_array_trajectory[current_R,t, :, 0]), list(c_array_trajectory[current_R,t, :, 1]), color='green', alpha=1, label='Centroids', marker='P', s=100)

            ax.set_title(f'Iteration t={t}, r={r}, {alg_name}')

            ax.legend()

        ani = FuncAnimation(fig, animate, frames=T, interval=200, fargs=(r,))

        folder_name = os.path.join("figures", config_folder, experiment_name, "gif")

        create_folder_if_needed(folder_name)

        # Save animation as a GIF
        gif_path = os.path.join(folder_name, f"particle_evolution_{r}.gif")
        ani.save(gif_path, writer=PillowWriter(fps=10))
        plt.close(fig)  # Close the figure to avoid displaying static plots

    return gif_path

@nb.jit(parallel=True)
def compute_all_mmds_uncached(all_nodes_arr, X, kernel, all_weights_arr):
    M, D = all_nodes_arr.shape[-2:]
    all_nodes = all_nodes_arr.reshape(-1, M, D)
    all_weights = all_weights_arr.reshape(-1, M)
    mmds = np.empty(len(all_nodes))
    for i in nb.prange(len(all_nodes)):
        Y = all_nodes[i]
        weights_Y = all_weights[i]
        mmds[i] = mmd_tools.compute_mmd_weighted(X, Y, kernel, weights_Y = weights_Y)
    return mmds.reshape(all_nodes_arr.shape[:-2])

def compute_all_mmds_cached(all_nodes_arr, X, kernel, all_weights_arr):
    M, D = all_nodes_arr.shape[-2:]
    all_nodes = all_nodes_arr.reshape(-1, M, D)
    all_weights = all_weights_arr.reshape(-1, M)
    mmds = mmd_tools.cached_large_mmd(X, all_nodes, all_weights, kernel)
    return mmds.reshape(all_nodes_arr.shape[:-2])

def compute_all_mmds(all_nodes_arr, X, kernel, all_weights_arr):
    MIN_CACHED_SIZE = 1000
    use_cache = all_nodes_arr.size > MIN_CACHED_SIZE
    if use_cache:
        return compute_all_mmds_cached(all_nodes_arr, X, kernel, all_weights_arr)
    else:
        return compute_all_mmds_uncached(all_nodes_arr, X, kernel, all_weights_arr)

def visualize_and_save_dynamics_with_mmd(alg_name, experiment_name, c_array_trajectory, w_array, data_array, kernel, config_folder = ""):
    R, T, M, _ = c_array_trajectory.shape
    mmd_values = np.zeros((R, T))

    mmd_folder = os.path.join("figures", config_folder , experiment_name, "plots")
    os.makedirs(mmd_folder, exist_ok=True)

    mmd_values = compute_all_mmds(c_array_trajectory, data_array, kernel, w_array)
    w_sums = w_array.sum(axis=2)
    plt.figure()
    for r in range(R):
        plt.plot(range(T), w_sums[r], label=f"r={r}", lw=3)
    plt.title(f"Sum of weights, {alg_name}")
    plt.xlabel("t")
    plt.ylabel("Sum of weights")
    plt.legend()
    plt.savefig(os.path.join(mmd_folder, "sum_of_weights.png"))
    plt.show()

    for r in range(R):
        for m in range(M):
            plt.figure()
            plt.plot(range(T), w_array[r, :, m], label=f"m={m}")
            plt.title(f"Plot of weights, r={r}, {alg_name}")
            plt.xlabel("t")
            plt.ylabel("w_array")
            plt.xscale('log')
            plt.legend()
            plt.grid(True)

            w_array_plot_path = os.path.join(mmd_folder, "w_array_evolution_r_"+str(r)+"_m_"+str(m)+".png")
            plt.savefig(w_array_plot_path)

            plt.show()

    for r in range(R):
        fig, axes = plt.subplots( M+1,1, figsize=(8, (M+1) * 5))  # 1 row, M+1 columns
        ax = axes[0]
        ax.plot(range(T), mmd_values[r], color = "black")
        ax.set_xscale('log')
        ax.set_title(f"Evolution of MMD, r={r}, {alg_name}")
        for m in range(M):
            ax = axes[m+1]  # Select the current subplot
            ax.plot(range(T), w_array[r, :, m], label=f"m={m}", color = "black")
            ax.set_xlabel("t")
            ax.set_xscale('log')
            ax.legend()
            ax.grid(True)

        # Adjust layout and save the figure
        #fig.suptitle(f"Evolution of w_array for m={m}")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust space for the title
        w_array_plot_path = os.path.join(mmd_folder, f"w_array_evolution_r_{r}_horizontal.png")
        plt.savefig(w_array_plot_path)
        plt.show()

    for r in range(R):
        fig, axes = plt.subplots( M+1,1, figsize=(8, (M+1) * 5))  # 1 row, M+1 columns
        ax = axes[0]
        ax.plot(range(T), mmd_values[r], label="MMD", color = "black")
        ax.set_xscale('log')
        ax.set_title(f"Evolution of MMD with weight signs, r={r}, {alg_name}")
        #plt.plot(range(T), mmd_values[r], color='black')
        for m in range(M):
            ax = axes[m+1]  # Select the current subplot
            ax.plot(range(T), np.sign(w_array[r, :, m]), label=f"m={m}", color = "black")
            #ax.set_title(f"r={r}, m={m}")
            ax.set_xlabel("t")
            #ax.set_ylabel("w_array")
            ax.set_xscale('log')
            ax.legend()
            ax.grid(True)

        # Adjust layout and save the figure
        #fig.suptitle(f"Evolution of w_array for m={m}")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust space for the title
        w_array_plot_path = os.path.join(mmd_folder, f"w_array_evolution_r_{r}_horizontal_sign.png")
        plt.savefig(w_array_plot_path)
        plt.show()

    # Plot MMD over iterations for all repetitions
    plt.figure(figsize=(10, 6))
    for r in range(R):
        plt.plot(range(T), mmd_values[r], label=f"r={r}", lw=3)

    plt.title(f"MMD evolution over iterations, {alg_name}")
    plt.xlabel("Iteration")
    plt.ylabel("MMD")
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.grid()
    mmd_plot_path = os.path.join(mmd_folder, "mmd_evolution.png")
    plt.savefig(mmd_plot_path)
    plt.show()
    plt.close()

    return mmd_plot_path