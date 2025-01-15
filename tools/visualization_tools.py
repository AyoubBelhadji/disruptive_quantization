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
from matplotlib.animation import FuncAnimation, PillowWriter, ImageMagickWriter, FFMpegWriter
import tools.mmd_tools as mmd

import os
from tools.files_tools import create_folder_if_needed


def expand_limits(min, max, factor):
    """
    Expand the limits of a plot by a factor.
    """
    delta = max - min
    return min - factor * delta, max + factor * delta


def create_dynamics_gif(data_array, centroids, config_folder, experiment_name, r, alg_name, file_format, **ax_kwargs):
    T = centroids.shape[0]
    fig, ax = plt.subplots()
    ax.set(**ax_kwargs)
    ax.scatter(data_array[:, 0], data_array[:, 1], color='black', alpha=0.5)
    ax.scatter(centroids[0, :, 0], centroids[0, :, 1], color='red',
               label='Initial Centroids', alpha=1, marker='D')
    centroids_t_scatter = ax.scatter(
        centroids[0, :, 0], centroids[0, :, 1], color='green', alpha=1, label='Centroids', marker='P', s=100)

    def update(t):
        ax.set_title(f"Centroid dynamics, r={r}, t={t}, {alg_name}")
        centroids_t_scatter.set_offsets(centroids[t])
        return centroids_t_scatter

    ani = FuncAnimation(fig, update, frames=T, interval=200)
    folder_name = os.path.join(
        "figures", config_folder, experiment_name, "gif")
    create_folder_if_needed(folder_name)

    # Save animation as a GIF
    gif_path = os.path.join(
        folder_name, f"particle_evolution_{r}." + file_format)
    writer_classes = [FFMpegWriter, ImageMagickWriter, PillowWriter]
    for writer_class in writer_classes:
        try:
            writer = writer_class(fps=10)
            break
        except:
            continue
    ani.save(gif_path, writer=writer)
    plt.close(fig)  # Close the figure to avoid displaying static plots


def centroid_dynamics(alg_name, experiment_name, c_array_trajectory, data_array, config_folder="", file_format="gif", limit_margin=0.1):
    R = c_array_trajectory.shape[0]
    xlims = expand_limits(np.min(data_array[:, 0]), np.max(
        data_array[:, 0]), limit_margin)
    ylims = expand_limits(np.min(data_array[:, 1]), np.max(
        data_array[:, 1]), limit_margin)
    for r in range(R):
        centroids_r = c_array_trajectory[r]
        create_dynamics_gif(data_array, centroids_r, config_folder,
                            experiment_name, r, alg_name, file_format, xlim=xlims, ylim=ylims)


def weight_sum_plot(alg_name, mmd_folder, w_sums):
    """ Plot the sum of weights over iterations for all repetitions """
    R, T = w_sums.shape
    fig, ax = plt.subplots()
    for r in range(R):
        ax.plot(range(T), w_sums[r], label=f"r={r}", lw=3)
    ax.set(title=f"Sum of weights, {alg_name}",
           xlabel="t", ylabel="Sum of weights")
    ax.legend()
    fig.savefig(os.path.join(mmd_folder, "sum_of_weights.png"))
    plt.show()


def weight_evolution_plot(alg_name, w_array, mmd_folder, r, m):
    """ Plot the evolution of each weight for a given repetition and centroid """
    T = w_array.shape[1]
    fig, ax = plt.subplots()
    ax.plot(range(T), w_array[r, :, m], label=f"m={m}")
    ax.set(title=f"Plot of weights, r={r}, m={m}, {
           alg_name}", xlabel="t", ylabel="w_array", xscale='log')
    ax.legend()
    ax.grid(True)

    w_array_plot_path = os.path.join(
        mmd_folder, "w_array_evolution_r_"+str(r)+"_m_"+str(m)+".png")
    fig.savefig(w_array_plot_path)
    plt.show()


def mmd_weight_evolution_plot(alg_name, w_array, mmd_values, mmd_folder, r):
    """ Plot MMD evolution and the value of each weight for a given repetition """
    T, M = w_array.shape[1:]
    fig, axes = plt.subplots(
        M+1, 1, figsize=(8, (M+1) * 5))  # 1 row, M+1 columns
    ax = axes[0]
    ax.plot(range(T), mmd_values[r], color="black")
    ax.set(xscale='log', title=f"Evolution of MMD, r={r}, {alg_name}")
    for m in range(M):
        ax = axes[m+1]  # Select the current subplot
        ax.plot(range(T), w_array[r, :, m], label=f"m={m}", color="black")
        ax.set(title=f"Evolution of weights, r={r}, m={
               m}, {alg_name}", xlabel="t", xscale='log')
        ax.legend()
        ax.grid(True)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust space for the title
    w_array_plot_path = os.path.join(
        mmd_folder, f"w_array_evolution_r_{r}_horizontal.png")
    fig.savefig(w_array_plot_path)
    plt.show()


def mmd_weight_signs_plot(alg_name, w_array, mmd_values, mmd_folder, r):
    """ Plot MMD evolution and the sign of each weight for a given repetition """
    T, M = w_array.shape[1:]
    fig, axes = plt.subplots(
        M+1, 1, figsize=(8, (M+1) * 5))  # 1 row, M+1 columns
    ax = axes[0]
    ax.plot(range(T), mmd_values[r], label="MMD", color="black")
    ax.set(title=f'Evolution of MMD with weight signs, r={
           r}, {alg_name}', xscale='log', xlabel="t")
    for m in range(M):
        ax = axes[m+1]  # Select the current subplot
        ax.plot(range(T), np.sign(
            w_array[r, :, m]), label=f"m={m}", color="black")
        ax.set(xlabel="t", xscale="log")
        ax.legend()
        ax.grid(True)

    # Adjust layout and save the figure
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust space for the title
    w_array_plot_path = os.path.join(
        mmd_folder, f"w_array_evolution_r_{r}_horizontal_sign.png")
    fig.savefig(w_array_plot_path)
    plt.show()


def mmd_all_plot(alg_name, mmd_values, mmd_folder):
    """ Plot MMD evolution over iterations for all repetitions """
    R, T = mmd_values.shape
    fig, ax = plt.subplots()
    for r in range(R):
        ax.plot(range(T), mmd_values[r], label=f"r={r}", lw=3)
    ax.set(title=f"MMD evolution over iterations, {
           alg_name}", xlabel="Iteration", ylabel="MMD", xscale='log', yscale='log')
    ax.legend()
    ax.grid()
    mmd_plot_path = os.path.join(mmd_folder, "mmd_evolution.png")
    fig.savefig(mmd_plot_path)
    plt.show()


def logdet_all_plot(alg_name, logdets, mmd_folder):
    """ Plot logdet evolution over iterations for all repetitions """
    R, T = logdets.shape
    fig, ax = plt.subplots()
    for r in range(R):
        plt.plot(range(T), logdets[r], label=f"r={r}", lw=3)
    ax.set(title=f"Logdet evolution over iterations, {
           alg_name}", xlabel="Iteration", ylabel="Logdet", xscale='log')
    ax.legend()
    ax.grid()
    logdet_plot_path = os.path.join(mmd_folder, "logdet_evolution.png")
    fig.savefig(logdet_plot_path)
    plt.show()


def calculate_mmd_and_logdets(experiment_name, c_array_trajectory, w_array, data_array, kernel, config_folder, cached_MMD=True):
    mmd_folder_serial = os.path.join(
        "experiments", "sandbox", config_folder, experiment_name)
    os.makedirs(mmd_folder_serial, exist_ok=True)

    mmd_values = mmd.mmd_array(
        data_array, c_array_trajectory, w_array, kernel, cached_MMD)
    logdets = mmd.logdet_array(c_array_trajectory, kernel)

    # Save the mmd_values and logdets
    np.save(os.path.join(mmd_folder_serial, "mmd_values.npy"), mmd_values)
    np.save(os.path.join(mmd_folder_serial, "logdets.npy"), logdets)
    print("MMD values and logdets saved to ", mmd_folder_serial)
    return mmd_values, logdets


def evolution_weights_mmd(alg_name, experiment_name, c_array_trajectory, w_array, data_array, kernel, config_folder=""):
    R, _, M, _ = c_array_trajectory.shape

    mmd_folder_plots = os.path.join(
        "figures", config_folder, experiment_name, "plots")
    os.makedirs(mmd_folder_plots, exist_ok=True)
    mmd_values, logdets = calculate_mmd_and_logdets(
        experiment_name, c_array_trajectory, w_array, data_array, kernel, config_folder)

    w_sums = w_array.sum(axis=2)
    # Plot the sum of weights over iterations
    weight_sum_plot(alg_name, mmd_folder_plots, w_sums)

    for r in range(R):
        for m in range(M):
            # Plot the evolution of each weight
            weight_evolution_plot(alg_name, w_array, mmd_folder_plots, r, m)

    for r in range(R):
        # Plot MMD evolution and the value of each weight
        mmd_weight_evolution_plot(
            alg_name, w_array, mmd_values, mmd_folder_plots, r)

    for r in range(R):
        # Plot MMD evolution and the sign of each weight
        mmd_weight_signs_plot(
            alg_name, w_array, mmd_values, mmd_folder_plots, r)

    # Plot MMD over iterations for all repetitions
    mmd_all_plot(alg_name, mmd_values, mmd_folder_plots)

    # Plot logdet over iterations for all repetitions
    logdet_all_plot(alg_name, logdets, mmd_folder_plots)
    plt.close()