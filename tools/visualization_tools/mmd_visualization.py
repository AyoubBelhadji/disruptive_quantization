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
import tools.mmd_tools as mmd

import os

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


def weight_evolution_plot(alg_name, w_array, mmd_folder, r):
    """ Plot the evolution of each weight for a given repetition and centroid """
    T = w_array.shape[1]
    fig, ax = plt.subplots()
    for m in range(w_array.shape[-1]):
        ax.plot(range(T), w_array[r, :, m], label=f"m={m}", lw=3)
    ax.set(title=f"Plot of weights, r={r}, m={m}, {
           alg_name}", xlabel="t", ylabel="w_array", xscale='log')
    ax.legend()
    ax.grid(True)

    w_array_plot_path = os.path.join(
        mmd_folder, "w_array_evolution_r_"+str(r)+"_m_"+str(m)+".png")
    fig.savefig(w_array_plot_path)


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


def calculate_mmd_and_logdets(c_array, w_array, data_array, kernel, mmd_self, subpath):
    mmd_folder_serial = os.path.join(
        "experiments", "sandbox", subpath)
    os.makedirs(mmd_folder_serial, exist_ok=True)

    mmd_values = mmd.mmd_array(c_array, w_array, data_array, kernel, mmd_self)
    logdets = mmd.logdet_array(c_array, kernel)

    # Save the mmd_values and logdets
    np.save(os.path.join(mmd_folder_serial, "mmd_values.npy"), mmd_values)
    np.save(os.path.join(mmd_folder_serial, "logdets.npy"), logdets)
    print("MMD values and logdets saved to ", mmd_folder_serial)
    return mmd_values, logdets


def evolution_weights_mmd(alg_name, c_array, w_array, data_array, kernel, dataset_name, subpath, show_plots):
    R, _, M, _ = c_array.shape
    show_plot_fcn = plt.show if show_plots else lambda: None

    mmd_folder_plots = os.path.join("figures", subpath, "plots")
    os.makedirs(mmd_folder_plots, exist_ok=True)

    mmd_self = mmd.Self_MMD_Dict(dataset_name, data_array.shape[0])
    mmd_values, logdets = calculate_mmd_and_logdets(c_array, w_array, data_array, kernel, mmd_self, subpath)

    w_sums = w_array.sum(axis=-1)
    # Plot the sum of weights over iterations
    weight_sum_plot(alg_name, mmd_folder_plots, w_sums)
    show_plot_fcn()

    for r in range(R):
        # Plot the evolution of each weight
        weight_evolution_plot(alg_name, w_array, mmd_folder_plots, r)
        show_plot_fcn()

    for r in range(R):
        # Plot MMD evolution and the value of each weight
        mmd_weight_evolution_plot(
            alg_name, w_array, mmd_values, mmd_folder_plots, r)
        show_plot_fcn()

    for r in range(R):
        # Plot MMD evolution and the sign of each weight
        mmd_weight_signs_plot(
            alg_name, w_array, mmd_values, mmd_folder_plots, r)
        show_plot_fcn()

    # Plot MMD over iterations for all repetitions
    mmd_all_plot(alg_name, mmd_values, mmd_folder_plots)
    show_plot_fcn()

    # Plot logdet over iterations for all repetitions
    logdet_all_plot(alg_name, logdets, mmd_folder_plots)
    show_plot_fcn()
    plt.close()
