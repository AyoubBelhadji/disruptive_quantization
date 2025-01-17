
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, ImageMagickWriter, FFMpegWriter

import os
from tools.files_tools import create_folder_if_needed

def expand_limits(min, max, factor):
    """
    Expand the limits of a plot by a factor.
    """
    delta = max - min
    return min - factor * delta, max + factor * delta


def create_dynamics_gif(data_array, centroids, r, alg_name, file_format, subpath, **ax_kwargs):
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
        "figures", subpath, "mp4")
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


def centroid_dynamics(alg_name, c_array_trajectory, data_array, subpath, file_format="gif", limit_margin=0.1):
    if data_array.shape[-1] != 2:
        print("Data array must have shape (N, 2) for visualizing centroid dynamics")
        return
    R = c_array_trajectory.shape[0]
    xlims = expand_limits(np.min(data_array[:, 0]), np.max(
        data_array[:, 0]), limit_margin)
    ylims = expand_limits(np.min(data_array[:, 1]), np.max(
        data_array[:, 1]), limit_margin)
    for r in range(R):
        centroids_r = c_array_trajectory[r]
        create_dynamics_gif(data_array, centroids_r, r, alg_name, file_format, subpath, xlim=xlims, ylim=ylims)