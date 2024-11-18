# kde_comparison.py
# kde_simple.py
# kde_matern_standard.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.special import kv, gamma
from gmm_generator import generate_gmm_data

def matern_kernel_pdf(r, nu, length_scale):
    """
    Compute the normalized Matern kernel to be used as a PDF.

    Parameters:
    - r: ndarray
        Distance between points.
    - nu: float
        Smoothness parameter.
    - length_scale: float
        Length scale parameter.

    Returns:
    - k: ndarray
        Kernel values.
    """
    r = r / length_scale
    factor = (2 ** (1 - nu)) / gamma(nu)
    k = factor * (r ** nu) * kv(nu, r)
    k[r == 0] = factor * (r[r == 0] ** nu) * (gamma(nu) * 2 ** (1 - nu))
    k = np.nan_to_num(k)  # Replace NaNs with zeros
    return k

def kde_with_matern_kernel(data, grid_points, nu=1.5, length_scale=1.0):
    """
    Perform KDE using the Matern kernel.

    Parameters:
    - data: ndarray of shape (n_samples, d)
        Input data.
    - grid_points: ndarray of shape (N, d)
        Points where the density is evaluated.
    - nu: float
        Smoothness parameter for the Matern kernel.
    - length_scale: float
        Length scale parameter for the Matern kernel.

    Returns:
    - density: ndarray of shape (N,)
        Estimated density at the grid points.
    """
    # Compute pairwise distances between grid points and data points
    distances = cdist(grid_points, data)

    # Compute the kernel values
    kernel_values = matern_kernel_pdf(distances, nu, length_scale)

    # Sum over data points and normalize
    density = np.sum(kernel_values, axis=1) / len(data)

    # Normalize density to integrate to one
    # For KDE in 2D, we need to normalize over the area
    area = (grid_points[:, 0].max() - grid_points[:, 0].min()) * \
           (grid_points[:, 1].max() - grid_points[:, 1].min())
    density /= np.sum(density) * (area / len(density))

    return density

def plot_kde_comparison(data, kde_gaussian, density_matern, grid, xx, yy):
    """
    Plot the comparison between KDE with Gaussian kernel and Matern kernel.

    Parameters:
    - data: ndarray of shape (n_samples, 2)
        Input data.
    - kde_gaussian: KernelDensity object
        Fitted KDE with Gaussian kernel.
    - density_matern: ndarray
        Density estimated using Matern kernel.
    - grid: ndarray of shape (N, 2)
        Grid points where the density is evaluated.
    - xx, yy: ndarray
        Meshgrid arrays for plotting.
    """
    # Evaluate Gaussian KDE density
    log_dens_gaussian = kde_gaussian.score_samples(grid)
    dens_gaussian = np.exp(log_dens_gaussian)

    # Plotting
    plt.figure(figsize=(16, 6))

    # KDE with Gaussian kernel
    plt.subplot(1, 2, 1)
    plt.title('KDE with Gaussian Kernel')
    plt.scatter(data[:, 0], data[:, 1], s=5, alpha=0.5)
    plt.contourf(xx, yy, dens_gaussian.reshape(xx.shape), levels=50, cmap='viridis', alpha=0.7)
    plt.colorbar(label='Density')

    # KDE with Matern kernel
    plt.subplot(1, 2, 2)
    plt.title('KDE with Matern Kernel')
    plt.scatter(data[:, 0], data[:, 1], s=5, alpha=0.5)
    plt.contourf(xx, yy, density_matern.reshape(xx.shape), levels=50, cmap='viridis', alpha=0.7)
    plt.colorbar(label='Density')

    plt.show()

if __name__ == "__main__":
    # Generate 2D GMM data
    d = 2
    n_samples = 5
    n_components = 3
    data, labels, params = generate_gmm_data(d, n_samples, n_components)

    # Perform KDE with Gaussian kernel
    from sklearn.neighbors import KernelDensity
    bandwidth = 0.5  # Adjust as needed
    kde_gaussian = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
    kde_gaussian.fit(data)

    # Create a grid over the data range
    x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
    y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    grid = np.vstack([xx.ravel(), yy.ravel()]).T

    # Perform KDE with Matern kernel
    nu = 0.1  # Smoothness parameter
    length_scale = 0.5
    density_matern = kde_with_matern_kernel(data, grid, nu=nu, length_scale=length_scale)

    # Plot comparison
    plot_kde_comparison(data, kde_gaussian, density_matern, grid, xx, yy)

