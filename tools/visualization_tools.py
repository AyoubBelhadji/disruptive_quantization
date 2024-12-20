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

def visualize_and_save_dynamics(experiment_name, c_array_trajectory, data_array, config_folder = ""):
    R,T,M,d = c_array_trajectory.shape
    animations = []  # To store animations for each repetition
    xlims = [np.min(data_array[:, 0]), np.max(data_array[:, 0])]
    ylims = [np.min(data_array[:, 1]), np.max(data_array[:, 1])]
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

        folder_name = os.path.join("figures", config_folder, experiment_name, "gif")

        create_folder_if_needed(folder_name)

        # Save animation as a GIF
        gif_path = os.path.join(folder_name, f"particle_evolution_{r}.gif")
        ani.save(gif_path, writer=PillowWriter(fps=10))
        plt.close(fig)  # Close the figure to avoid displaying static plots

    return gif_path





def compute_mmd(X, Y, kernel):

    K_XX = kernel.kernel(X, X)
    K_YY = kernel.kernel(Y, Y)
    K_XY = kernel.kernel(X, Y)

    mmd = np.mean(K_XX) + np.mean(K_YY) - 2 * np.mean(K_XY)
    return mmd


def compute_mmd_weighted(X, Y, kernel, weights_X=None, weights_Y=None):
    """
    Computes the weighted MMD between two sets of samples X and Y.

    Parameters:
    - X: numpy array of shape (n, d), first set of samples.
    - Y: numpy array of shape (m, d), second set of samples.
    - kernel: a kernel function object with a `kernel` method.
    - weights_X: numpy array of shape (n,), weights for samples in X (optional, defaults to uniform).
    - weights_Y: numpy array of shape (m,), weights for samples in Y (optional, defaults to uniform).

    Returns:
    - mmd: the weighted MMD value.
    """
    if weights_X is None:
        weights_X = np.ones(len(X)) / len(X)
    if weights_Y is None:
        weights_Y = np.ones(len(Y)) / len(Y)

    weights_X = weights_X[:, np.newaxis]
    weights_Y = weights_Y[:, np.newaxis]

    K_XX = kernel(X, X)
    K_YY = kernel(Y, Y)
    K_XY = kernel(X, Y)

    # Weighted means
    mmd = (
        np.sum(weights_X * weights_X.T * K_XX) +
        np.sum(weights_Y * weights_Y.T * K_YY) -
        2 * np.sum(weights_X * weights_Y.T * K_XY)
    )
    return mmd


def visualize_and_save_dynamics_with_mmd(experiment_name, c_array_trajectory, data_array, kernel, config_folder = ""):

    R, T, M, d = c_array_trajectory.shape
    N,_ = data_array.shape
    mmd_values = np.zeros((R, T))

    w_array = np.zeros((R,T,M))


    mmd_folder = os.path.join("figures", config_folder , experiment_name, "plots")
    os.makedirs(mmd_folder, exist_ok=True)

    for r in range(R):
        for t in range(T):
            K_matrix = np.zeros((M, M))
            v_0_array = np.zeros((M))


            for m in list(range(M)):
                tmp_0_list = [kernel(
                    c_array_trajectory[r,t,m, :], data_array[n, :]) for n in range(N)]
                v_0_array[m] = (1/N)*sum(tmp_0_list)
            for m_1 in range(M):
                for m_2 in range(M):
                    K_matrix[m_1, m_2] = kernel(
                        c_array_trajectory[r,t,m_1, :], c_array_trajectory[r,t,m_2, :])

            w_array[r,t,:] = np.dot(np.linalg.inv(K_matrix),v_0_array)
            mmd_values[r, t] = compute_mmd_weighted(data_array, c_array_trajectory[r, t],kernel, None, w_array[r,t,:])

    for r in range(R):

        for m in range(M):
            plt.figure()
            plt.plot(range(T), w_array[r, :, m], label=f"m={m}")
            plt.title(f"Plot of w_array")
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
        ax.plot(range(T), mmd_values[r], label="MMD", color = "black")
        ax.set_xscale('log')
        #plt.plot(range(T), mmd_values[r], color='black')
        for m in range(M):
            ax = axes[m+1]  # Select the current subplot
            ax.plot(range(T), w_array[r, :, m], label=f"m={m}", color = "black")
            #ax.set_title(f"r={r}, m={m}")
            ax.set_xlabel("t")
            #ax.set_ylabel("w_array")
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
        plt.plot(range(T), mmd_values[r], color='black')

    plt.title("MMD evolution over iterations")
    plt.xlabel("Iteration")
    plt.ylabel("MMD")
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.grid()

    mmd_plot_path = os.path.join(mmd_folder, "mmd_evolution.png")
    plt.savefig(mmd_plot_path)
    plt.close()

    return mmd_plot_path

