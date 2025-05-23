
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
    N_data = data_array.shape[0]
    alpha_data = min(0.5, max(15/N_data,1))
    fig, ax = plt.subplots()
    ax.set(**ax_kwargs)
    ax.set_aspect('equal')
    ax.scatter(data_array[:, 0], data_array[:, 1], color='black', alpha=alpha_data)
    ax.scatter(centroids[0, :, 0], centroids[0, :, 1], color='red',
               label='Initial Centroids', alpha=1, marker='D')
    centroids_t_scatter = ax.scatter(
        centroids[0, :, 0], centroids[0, :, 1], color='green', alpha=1, label='Centroids', marker='P', s=100)
    def update(t):
        ax.set_title(f"Centroid dynamics, r={r}, t={t}, {alg_name}")
        centroids_t_scatter.set_offsets(centroids[t])
        return centroids_t_scatter

    max_every_frame_T = 500
    if T <= max_every_frame_T:
        frames = range(T)
    else:
        frames = np.unique(np.linspace(0, T - 1, max_every_frame_T).astype(int))
    ani = FuncAnimation(fig, update, frames=frames, interval=100)
    folder_name = os.path.join(
        "figures", subpath, "gif")
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

def centroid_dynamics(alg_name, y_trajectory, data_array, subpath, file_format="mp4", limit_margin=0.2):
    if data_array.shape[-1] != 2:
        print("Data array must have shape (N, 2) for visualizing centroid dynamics")
        return
    R = y_trajectory.shape[0]
    xlims_data = expand_limits(np.min(data_array[:, 0]), np.max(
        data_array[:, 0]), limit_margin)
    xlims_centroids = expand_limits(np.min(y_trajectory[:, :, :, 0]), np.max(
        y_trajectory[:, :, :, 0]), limit_margin)
    ylims_data = expand_limits(np.min(data_array[:, 1]), np.max(
        data_array[:, 1]), limit_margin)
    ylims_centroids = expand_limits(np.min(y_trajectory[:, :, :, 1]), np.max(
        y_trajectory[:, :, :, 1]), limit_margin)
    # xlims = (min(xlims_data[0], xlims_centroids[0]), max(xlims_data[1], xlims_centroids[1]))
    # ylims = (min(ylims_data[0], ylims_centroids[0]), max(ylims_data[1], ylims_centroids[1]))
    xlims = (-4, 4)
    ylims = (-4, 4)
    for r in range(R):
        centroids_r = y_trajectory[r]
        create_dynamics_gif(data_array, centroids_r, r, alg_name, file_format, subpath, xlim=xlims, ylim=ylims)