import torch
import numpy as np


# LEGACY CODE
# @torch.no_grad()
# def roulette_wheel(f, s=3., eps=1e-12, assume_sorted=False, normalize=False, consider_nans=False, threshold=0):

#     if not isinstance(f, (torch.Tensor, np.ndarray)):
#         f = torch.tensor(f)
#
#     f_nan = torch.isnan(f) if isinstance(f, torch.Tensor) else np.isnan(f)
#     f_inf = torch.isinf(f) if isinstance(f, torch.Tensor) else np.isinf(f)
#     f_valid = ~(f_nan | f_inf)
#
#     f_finite = torch.zeros_like(f) if isinstance(f, torch.Tensor) else np.zeros_like(f)
#     if not consider_nans:
#         f = f[f_valid]
#
#     if threshold > 0:
#         f_threshold = f.min() + (f.max() - f.min()) * threshold
#         f[f < f_threshold] = f_threshold  # apply threshold to fitness values
#
#     exp = torch.exp if isinstance(f, torch.Tensor) else np.exp
#     device = f.device if isinstance(f, torch.Tensor) else 'cpu'
#     indices = torch.arange(len(f), device=device) if isinstance(f, torch.Tensor) else np.arange(len(f))
#
#     if not assume_sorted:
#         # sort fitness in ascending order
#         if isinstance(f, torch.Tensor):
#             asc = torch.argsort(f.flatten(), descending=False, dim=0)
#             where = torch.where
#         else:  # numpy
#             asc = f.flatten().argsort()
#             where = np.where
#
#         indices = where(asc[None, :] == indices[:, None])[1]  # original order
#         f = f[asc]
#
#     unique_f = (torch.unique(f) if isinstance(f, torch.Tensor) else np.unique(f))
#     if isinstance(f, torch.Tensor):
#         total_weight = torch.abs(f).sum()
#         fs = torch.zeros_like(f)
#         unique_mapping = [(torch.where(uf == f)[0]).flatten() for uf in unique_f]
#     else:
#         total_weight = np.abs(f).sum()
#         fs = np.zeros_like(f)
#         unique_mapping = [(np.where(uf == f)[0]).flatten() for uf in unique_f]
#
#     f_scaled = (unique_f - f.min()) / (f.max() - f.min() + eps)  # normalize fitness values to [0, 1], and sort
#     f_scaled = exp(s*f_scaled)  # apply selection pressure, s can be positive or negative
#     f_scaled = f_scaled.cumsum(dim=0) if isinstance(f_scaled, torch.Tensor) else np.cumsum(f_scaled) # compute cumulative sum
#     f_scaled += eps
#
#     if len(unique_mapping) != len(fs):
#         # remap to original lengths (allow duplicates)
#         for i, um in enumerate(unique_mapping):
#             fs[um] = f_scaled[i]
#     else:
#         fs = f_scaled
#
#     fs /= fs.sum()
#     if not normalize:
#         fs *= total_weight
#
#     fs = fs[indices]
#     if consider_nans:
#         f_finite = fs
#     else:
#         f_finite[f_valid] = fs
#     return f_finite


@torch.no_grad()
def roulette_wheel(f, s=3., eps=1e-12, normalize=False, consider_nans=False, threshold=0):
    """ Roulette wheel fitness transformation.

    We transform the fitness values f to probabilities p by applying the roulette wheel fitness transformation.
    The roulette wheel fitness transformation is a monotonic transformation that maps the fitness values to
    probabilities. The selection pressure s controls the degree of selection. The higher the selection pressure,
    the more the probabilities are concentrated on the best solutions (s can be positive or negative).

    :param f: torch.Tensor of shape (popsize,), fitness values of the sampled solutions
    :param s: float, selection pressure
    :param eps: float, epsilon to avoid division by zero
    :param threshold: float [0, 1], fitness cutoff ratio (highest - lowest) which are considered as "finite" weights.
    :param normalize: bool, whether to normalize the probabilities to sum to 1 (default False, i.e., the sum over
                      the returned scaled probabilities is equal to the sum over the fitness absolute values)
    :param consider_nans: bool, whether to consider NaN fitness values in the selection process
    :return: torch.Tensor of shape (popsize,), indices of the selected solutions
    """
    original_is_numpy = isinstance(f, np.ndarray)
    if original_is_numpy:
        f = torch.from_numpy(f).to(torch.float32)

    if isinstance(s, np.ndarray):
        s = torch.from_numpy(s).to(torch.float32)

    assert 0 <= threshold <= 1, "Threshold must be in the range [0, 1]"

    f_nan = torch.isnan(f)
    f_inf = torch.isinf(f)
    f_valid = ~(f_nan | f_inf)

    # Initialize result tensor with original shape and all NaNs/Infs handled
    result_probabilities = torch.zeros_like(f) if not consider_nans else torch.full_like(f, float('nan'))

    f_filtered = f[f_valid].clone()  # Work on a copy of valid elements
    original_indices_valid = torch.where(f_valid)[0]  # Store original indices of valid elements
    total_weight = torch.abs(f_filtered).sum()  # Sum of the absolute values of the filtered fitness values

    if f_filtered.numel() == 0:
        return result_probabilities.numpy() if original_is_numpy else result_probabilities

    if threshold > 0:
        f_threshold = f_filtered.min() + (f_filtered.max() - f_filtered.min()) * threshold
        f_filtered.clamp_min_(f_threshold)

    unique_f_sorted, unique_inverse = torch.unique(f_filtered, return_inverse=True)

    f_scaled = (unique_f_sorted - unique_f_sorted.min()) / (unique_f_sorted.max() - unique_f_sorted.min() + eps)
    f_scaled = torch.exp(s * f_scaled)
    f_scaled = torch.cumsum(f_scaled, dim=0) + eps

    # Map back to original f_filtered size and normalize
    f_probabilities = f_scaled[unique_inverse]
    f_probabilities /= f_probabilities.sum()

    if not normalize:
        f_probabilities *= total_weight

    # Assign probabilities back to the original full-sized tensor
    if len(result_probabilities.shape) == 2 and len(f_probabilities.shape) == 1:
        f_probabilities = f_probabilities[:, None]

    result_probabilities[original_indices_valid] = f_probabilities

    if original_is_numpy:
        return result_probabilities.numpy()

    return result_probabilities


@torch.no_grad()
def boltzmann_selection(f, s=3., normalize=True, consider_nans=False, threshold=0, eps=0.):
    """ Boltzmann fitness selection transformation.

    We transform the fitness values f to probabilities p by applying a Boltzmann fitness transformation,
    which is a monotonic transformation that maps the fitness values to probabilities. The selection pressure s
    controls the degree of selection and takes the form of the inverse tempearture.
    The fitness is scaled by the maximum fitness value, and the probabilities are computed

    :param f: torch.Tensor of shape (popsize,), fitness values of the sampled solutions
    :param s: float, selection pressure, corresponds to inverse temperature.
    :param threshold: float [0, 1], fitness cutoff ratio (highest - lowest) which are considered as "finite" weights.
    :param eps: float, epsilon weight for samples below threshold.
    :param normalize: bool, whether to normalize the probabilities to sum to 1 (default False, i.e., the sum over
                      the returned scaled probabilities is equal to the sum over the fitness absolute values)
    :param consider_nans: bool, whether to consider NaN fitness values in the selection process
    :return: torch.Tensor of shape (popsize,), indices of the selected solutions
    """
    original_is_numpy = isinstance(f, np.ndarray)
    if original_is_numpy:
        f = torch.from_numpy(f).to(torch.float32)

    device = f.device
    assert 0 <= threshold <= 1, "Threshold must be in the range [0, 1]"

    f_nan = torch.isnan(f)
    f_inf = torch.isinf(f)
    f_valid = ~(f_nan | f_inf)

    # Initialize result tensor with original shape and all NaNs/Infs handled
    result_probabilities = torch.zeros_like(f) if not consider_nans else torch.full_like(f, float('nan'))

    f_filtered = f[f_valid].clone()  # Work on a copy of valid elements
    original_indices_valid = torch.where(f_valid)[0]  # Store original indices of valid elements
    total_weight = torch.abs(f_filtered).sum()  # Sum of the absolute values of the filtered fitness values

    if f_filtered.numel() == 0:
        return result_probabilities.numpy() if original_is_numpy else result_probabilities

    f_max = f_filtered.max() if isinstance(f_filtered, torch.Tensor) else np.max(f_filtered)
    f_scaled = torch.exp(-s * abs(f_filtered - f_max)) + eps  # apply Boltzmann selection

    if threshold > 0:
        f_threshold = f_filtered.min() + (f_filtered.max() - f_filtered.min()) * threshold
        f_threshold_transform = torch.exp(-s * abs(f_threshold - f_max)) + eps
        f_scaled.clamp_min_(f_threshold_transform)

    f_probabilities = f_scaled/f_scaled.sum()

    if not normalize:
        f_probabilities *= total_weight

    # Assign probabilities back to the original full-sized tensor
    if len(result_probabilities.shape) == 2 and len(f_probabilities.shape) == 1:
        f_probabilities = f_probabilities[:, None]

    result_probabilities[original_indices_valid] = f_probabilities

    if original_is_numpy:
        return result_probabilities.numpy()

    return result_probabilities
