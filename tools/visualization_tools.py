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
import tools.mmd_tools as mmd
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
def compute_all_mmds(all_nodes_arr, Y, kernel, all_weights_arr):
    M, D = all_nodes_arr.shape[-2:]
    all_nodes = all_nodes_arr.reshape(-1, M, D)
    all_weights = all_weights_arr.reshape(-1, M)
    mmds = np.empty(len(all_nodes))
    for i in nb.prange(len(all_nodes)):
        X = all_nodes[i]
        weights_X = all_weights[i]
        mmds[i] = mmd.compute_mmd_weighted(X, Y, kernel, weights_X)
    return mmds.reshape(all_nodes_arr.shape[:-2])

@nb.jit(parallel=True)
def compute_all_kernel_logdets(all_nodes_arr, kernel):
    M, D = all_nodes_arr.shape[-2:]
    all_nodes = all_nodes_arr.reshape(-1, M, D)
    logdets = np.empty(len(all_nodes))
    for i in nb.prange(len(all_nodes)):
        X = all_nodes[i]
        k_XX = mmd.broadcast_kernel(kernel, X, X)
        logdets[i] = np.linalg.slogdet(k_XX)[1]
    return logdets.reshape(all_nodes_arr.shape[:-2])

def weight_sum_plot(alg_name, mmd_folder, w_sums):
    """ Plot the sum of weights over iterations for all repetitions """
    R, T = w_sums.shape
    plt.figure()
    for r in range(R):
        plt.plot(range(T), w_sums[r], label=f"r={r}", lw=3)
    plt.title(f"Sum of weights, {alg_name}")
    plt.xlabel("t")
    plt.ylabel("Sum of weights")
    plt.legend()
    plt.savefig(os.path.join(mmd_folder, "sum_of_weights.png"))
    plt.show()

def weight_evolution_plot(alg_name, w_array, mmd_folder, r, m):
    """ Plot the evolution of each weight for a given repetition and centroid """
    T = w_array.shape[1]
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

def mmd_weight_evolution_plot(alg_name, w_array, mmd_values, mmd_folder, r):
    """ Plot MMD evolution and the value of each weight for a given repetition """
    T, M = w_array.shape[1:]
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

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust space for the title
    w_array_plot_path = os.path.join(mmd_folder, f"w_array_evolution_r_{r}_horizontal.png")
    plt.savefig(w_array_plot_path)
    plt.show()

def mmd_weight_signs_plot(alg_name, w_array, mmd_values, mmd_folder, r):
    """ Plot MMD evolution and the sign of each weight for a given repetition """
    T, M = w_array.shape[1:]
    fig, axes = plt.subplots( M+1,1, figsize=(8, (M+1) * 5))  # 1 row, M+1 columns
    ax = axes[0]
    ax.plot(range(T), mmd_values[r], label="MMD", color = "black")
    ax.set_xscale('log')
    ax.set_title(f"Evolution of MMD with weight signs, r={r}, {alg_name}")
    for m in range(M):
        ax = axes[m+1]  # Select the current subplot
        ax.plot(range(T), np.sign(w_array[r, :, m]), label=f"m={m}", color = "black")
        ax.set_xlabel("t")
        ax.set_xscale('log')
        ax.legend()
        ax.grid(True)

        # Adjust layout and save the figure
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust space for the title
    w_array_plot_path = os.path.join(mmd_folder, f"w_array_evolution_r_{r}_horizontal_sign.png")
    plt.savefig(w_array_plot_path)
    plt.show()

def mmd_all_plot(alg_name, mmd_values, mmd_folder):
    """ Plot MMD evolution over iterations for all repetitions """
    R, T = mmd_values.shape
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

def logdet_all_plot(alg_name, logdets, mmd_folder):
    """ Plot logdet evolution over iterations for all repetitions """
    R, T = logdets.shape
    plt.figure(figsize=(10, 6))
    for r in range(R):
        plt.plot(range(T), logdets[r], label=f"r={r}", lw=3)

    plt.title(f"Logdet evolution over iterations, {alg_name}")
    plt.xlabel("Iteration")
    plt.ylabel("Logdet")
    plt.xscale('log')
    plt.legend()
    plt.grid()
    logdet_plot_path = os.path.join(mmd_folder, "logdet_evolution.png")
    plt.savefig(logdet_plot_path)
    plt.show()

def visualize_and_save_dynamics_with_mmd(alg_name, experiment_name, c_array_trajectory, w_array, data_array, kernel, config_folder = ""):
    R, _, M, _ = c_array_trajectory.shape

    mmd_folder = os.path.join("figures", config_folder , experiment_name, "plots")
    os.makedirs(mmd_folder, exist_ok=True)

    mmd_values = compute_all_mmds(c_array_trajectory, data_array, kernel, w_array)
    logdets = compute_all_kernel_logdets(c_array_trajectory, kernel)

    w_sums = w_array.sum(axis=2)
    # Plot the sum of weights over iterations
    weight_sum_plot(alg_name, mmd_folder, w_sums)

    for r in range(R):
        for m in range(M):
            # Plot the evolution of each weight
            weight_evolution_plot(alg_name, w_array, mmd_folder, r, m)

    for r in range(R):
        # Plot MMD evolution and the value of each weight
        mmd_weight_evolution_plot(alg_name, w_array, mmd_values, mmd_folder, r)

    for r in range(R):
        # Plot MMD evolution and the sign of each weight
        mmd_weight_signs_plot(alg_name, w_array, mmd_values, mmd_folder, r)

    # Plot MMD over iterations for all repetitions
    mmd_all_plot(alg_name, mmd_values, mmd_folder)

    # Plot logdet over iterations for all repetitions
    logdet_all_plot(alg_name, logdets, mmd_folder)
    plt.close()