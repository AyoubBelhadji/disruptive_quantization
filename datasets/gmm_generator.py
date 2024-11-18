# gmm_generator.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import pickle

def generate_gmm_data(d, n_samples, n_components, params=None, save_path=None):
    """
    Generate data from a d-dimensional Gaussian Mixture Model and optionally save it to a pickle file.

    Parameters:
    - d: int
        Dimension of the data.
    - n_samples: int
        Total number of samples to generate.
    - n_components: int
        Number of Gaussian components in the mixture.
    - params: dict or None
        Parameters of the GMM. If None, random parameters are generated.
        Should contain 'weights', 'means', and 'covariances'.
    - save_path: str or None
        If provided, the generated data and parameters are saved to this path as a pickle file.

    Returns:
    - data: ndarray of shape (n_samples, d)
        Generated data points.
    - labels: ndarray of shape (n_samples,)
        Component labels for each data point.
    - params: dict
        Parameters used to generate the data.
    """
    if params is None:
        # Randomly generate weights, means, and covariances
        weights = np.random.dirichlet(np.ones(n_components))
        means = np.random.randn(n_components, d) * 5
        covariances = np.zeros((n_components, d, d))
        for i in range(n_components):
            A = np.random.randn(d, d)
            covariances[i] = np.dot(A, A.T) + np.eye(d)  # Ensure positive-definite
        params = {'weights': weights, 'means': means, 'covariances': covariances}
    else:
        weights = params['weights']
        means = params['means']
        covariances = params['covariances']

    # Generate samples for each component
    component_samples = np.random.choice(n_components, size=n_samples, p=weights)
    data = np.zeros((n_samples, d))
    labels = component_samples.copy()
    for i in range(n_components):
        idx = component_samples == i
        num_samples = np.sum(idx)
        if num_samples > 0:
            data[idx] = np.random.multivariate_normal(means[i], covariances[i], num_samples)

    # Save to pickle file if save_path is provided
    if save_path is not None:
        with open(save_path, 'wb') as f:
            pickle.dump({'data': data, 'labels': labels, 'params': params}, f)
        print(f"Data saved to {save_path}")

    return data, labels, params

def plot_gmm_data(data, labels=None, means=None, covariances=None):
    """
    Plot 2D GMM data with optional labels and true means and covariances.

    Parameters:
    - data: ndarray of shape (n_samples, 2)
        Data points to plot.
    - labels: ndarray of shape (n_samples,), optional
        Labels for coloring the data points.
    - means: ndarray of shape (n_components, 2), optional
        Means of the Gaussian components.
    - covariances: ndarray of shape (n_components, 2, 2), optional
        Covariance matrices of the components.
    """
    if data.shape[1] != 2:
        raise ValueError("Data must be 2-dimensional for plotting.")

    plt.figure(figsize=(8, 6))
    if labels is not None:
        plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', s=10, alpha=0.6)
    else:
        plt.scatter(data[:, 0], data[:, 1], s=10, alpha=0.6)

    if means is not None:
        plt.scatter(means[:, 0], means[:, 1], c='red', marker='x', s=100, label='Means')
        if covariances is not None:
            for i in range(len(means)):
                plot_cov_ellipse(covariances[i], means[i])
    plt.title('2D Gaussian Mixture Model Data')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()
    plt.show()

def plot_cov_ellipse(cov, mean, nstd=2, **kwargs):
    """
    Plot an ellipse representing the covariance matrix.

    Parameters:
    - cov: ndarray of shape (2, 2)
        Covariance matrix.
    - mean: ndarray of shape (2,)
        Mean vector.
    - nstd: int
        Number of standard deviations for the ellipse radius.
    """
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
    width, height = 2 * nstd * np.sqrt(eigvals)
    ellip = Ellipse(xy=mean, width=width, height=height, angle=angle, edgecolor='red', fc='None', lw=2, **kwargs)
    plt.gca().add_patch(ellip)

# Example usage when running the module directly
if __name__ == "__main__":
    d = 2
    n_samples = 1000
    n_components = 3
    save_path = 'gmm_data.pkl'  # Specify the filename to save the data
    data, labels, params = generate_gmm_data(d, n_samples, n_components, save_path=save_path)
    plot_gmm_data(data, labels, params['means'], params['covariances'])

