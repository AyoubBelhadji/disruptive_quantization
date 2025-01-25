import numpy as np
import numba as nb

# Hausdorff distance
@nb.jit(parallel=True)
def hausdorff_distance(x, y):
    """Compute the Hausdorff distance between two sets of points."""
    n, m = len(x), len(y)
    max_dist = 0
    for i in nb.prange(n):
        min_dist = np.inf
        for j in range(m):
            dist = np.linalg.norm(x[i] - y[j])
            min_dist = min(min_dist, dist)
        max_dist = max(max_dist, min_dist)
    return max_dist

# Voronoi MSE
@nb.jit(parallel=True)
def voronoi_mse(data, centroids):
    """Compute the mean squared error of the Voronoi diagram."""
    n, m = len(data), len(centroids)
    mse = 0
    for i in nb.prange(n):
        min_dist = np.inf
        for j in range(m):
            dist = np.sum((data[i] - centroids[j])**2)
            min_dist = min(min_dist, dist)
        mse += min_dist**2
    return mse / n