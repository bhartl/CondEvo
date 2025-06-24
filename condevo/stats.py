import torch
import numpy as np


def knn_entropy_estimate(x, y, k=10, p=2, keep_dim=False):
    """ Estimates the entropy of the data x with respect to the distribution y by evaluating
        sum of the log shortest distances of x_i to all points in y. """

    # dimension and datapoints
    N, d = x.shape[:2]

    # volume of the d-dimensional unit-ball
    log_c_d = 0.5 * d * torch.log(torch.tensor(torch.pi, device=x.device)) - torch.lgamma(torch.tensor(0.5 * d, device=x.device) + 1)

    # evaluate piecewise distance of x_i to all points in y
    dist_i_j = torch.cdist(x, y, p=p)

    # find k-nearest neighbors for every sample in x
    dist_i_knn, _ = dist_i_j.topk(k, largest=False, sorted=True, dim=1)

    # extract distance of the i-th sample to the k-th nearest neighbor
    dist_i_k = dist_i_knn[:, -1].unsqueeze(1)

    # eval i-th contribution to entropy as log(c_d) / N + d / N * log(dist_i_k)
    entropy_i = (log_c_d + 0.5 * d * torch.log(dist_i_k)) / N

    if keep_dim:
        return entropy_i

    return entropy_i.sum()


def distance_matrix(x, y):
    """Compute the pairwise distance matrix between x and y.

    Args:
        x: (N, d) tensor.
        y: (M, d) tensor.
    Returns:
        (N, M) tensor, the pairwise distance matrix.
    """
    return torch.cdist(x, y)


def diversity(x, axis=None):
    """Compute the diversity of a set of points in a given space.

    Args:
        x: (N, d) tensor, where N is the number of points and d is the dimensionality.
    Returns:
        float, the diversity of the points.
    """
    # Compute pairwise distances
    dist_matrix = distance_matrix(x, x)

    # Compute mean distance
    mean_distance = dist_matrix.mean(axis=axis)

    return mean_distance


def mst_length(X):
    from sklearn.metrics import pairwise_distances
    from scipy.sparse.csgraph import minimum_spanning_tree
    D = pairwise_distances(X)
    mst = minimum_spanning_tree(D)
    return mst.sum()


def per_point_mst_contributions(X):
    full_mst = mst_length(X)
    contributions = []
    for i in range(len(X)):
        X_subset = np.delete(X, i, axis=0)
        mst_without_i = mst_length(X_subset)
        contributions.append(full_mst - mst_without_i)
        print(i, full_mst - mst_without_i)
    return np.array(contributions)


def dpp_marginal_diversity(X, gamma=0.5):
    from sklearn.metrics.pairwise import rbf_kernel
    if not isinstance(X, np.ndarray):
        X = X.cpu().numpy()

    L = rbf_kernel(X, gamma=gamma)
    K = L @ np.linalg.inv(L + np.eye(len(X)))
    return np.diag(K)


def kde_estimate(x, sample_points=None, bandwidth=0.1):
    """ Kernel Density Estimation (KDE) method.

    """

    n_samples, d = x.shape
    # m_samples = sample_points.shape[0]

    if sample_points is None:
        sample_points = x.clone()

    # Calculate the pairwise distance between x and x_points
    distances = torch.cdist(sample_points, x, p=2)  # Shape (m_samples, n_samples)

    # Apply Gaussian kernel
    kernel_vals = torch.exp(-0.5 * (distances / bandwidth) ** 2)  # Shape (m_samples, n_samples)

    # Sum over all kernels and normalize
    kde_vals = kernel_vals.sum(dim=1)

    return kde_vals / kde_vals.sum()


def kl_divergence(p, q, eps=1e-8):
    """ Kullback-Leibler divergence between two distributions p and q.
    """
    if torch.isnan(torch.log(p)).any():
        print("here p")

    if torch.isnan(torch.log(q)).any():
        print("here q")

    p = p + eps
    p /= p.sum()

    q = q + eps
    q /= q.sum()

    kl = p * (torch.log(p) - torch.log(q))
    return kl.sum()


def grid_entropy_2d(data, grid_size=101, range_min=-10., range_max=10.):
    """
    Calculate the entropy of a set of 2D coordinates.

    :param data: np.ndarray, shape (n_samples, 2), the dataset of 2D coordinates
    :param grid_size: int, the number of grid cells along each dimension
    :param range_min: float, the minimum value for the grid range
    :param range_max: float, the maximum value for the grid range
    :return: float, the entropy of the dataset
    """
    # Normalize the data to fit within the specified range
    try:
        data = torch.clip(data, range_min, range_max)
        grid = torch.zeros((grid_size, grid_size), device=data.device, dtype=data.dtype)
        clip = torch.clip
        log = torch.log
    except TypeError:
        data = np.clip(data, range_min, range_max)
        grid = np.zeros((grid_size, grid_size))
        clip = np.clip
        log = np.log
        
    normalized_data = (data - range_min) / (range_max - range_min)

    # Count points in each grid cell
    indices = (normalized_data * grid_size).astype(int)
    indices = clip(indices, 0, grid_size - 1)  # Ensure indices are within bounds
    for idx in indices:
        grid[tuple(idx)] += 1

    # Convert counts to probabilities
    probabilities = grid / grid.sum()

    # Compute entropy
    non_zero_probs = probabilities[probabilities > 0]
    entropy = -((non_zero_probs * log(non_zero_probs)).sum())

    return entropy


def calculate_kl_divergence(data, grid_size=101, range_min=-10., range_max=10.):
    """
    Calculate the KL divergence of a set of 2D coordinates from a uniform distribution.

    :param data: np.ndarray, shape (n_samples, 2), the dataset of 2D coordinates
    :param grid_size: int, the number of grid cells along each dimension
    :param range_min: float, the minimum value for the grid range
    :param range_max: float, the maximum value for the grid range
    :return: float, the KL divergence of the dataset from a uniform distribution
    """
    # Normalize the data to fit within the specified range
    data = np.clip(data, range_min, range_max)
    normalized_data = (data - range_min) / (range_max - range_min)

    # Create a grid
    grid = np.zeros((grid_size, grid_size))

    # Count points in each grid cell
    indices = (normalized_data * grid_size).astype(int)
    indices = np.clip(indices, 0, grid_size - 1)  # Ensure indices are within bounds
    for idx in indices:
        grid[tuple(idx)] += 1

    # Convert counts to probabilities
    probabilities = grid / grid.sum()

    # Compute KL divergence
    uniform_prob = 1 / (grid_size * grid_size)
    non_zero_probs = probabilities[probabilities > 0]
    kl_divergence = np.sum(non_zero_probs * np.log(non_zero_probs / uniform_prob))

    return kl_divergence
