import torch
import numpy as np
from functools import partial
from .selection import roulette_wheel, boltzmann_selection  # legacy import for compatibility


def tensor_to_numpy(t: torch.Tensor):
    t = t.detach()
    try:
        return t.numpy()
    except RuntimeError:  # grad
        return t.detach().numpy()
    except TypeError:  # gpu
        return t.cpu().numpy()


class Optimizer(object):
    def __init__(self, mu, num_params, epsilon=1e-08):
        self.mu = mu
        self.dim = num_params
        self.epsilon = epsilon
        self.t = 0

    def update(self, globalg):
        self.t += 1
        step = self._compute_step(globalg)
        theta = self.mu
        ratio = np.linalg.norm(step) / (np.linalg.norm(theta) + self.epsilon)
        self.mu = theta + step
        return ratio

    def _compute_step(self, globalg):
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, mu, num_params, stepsize, momentum=0.9, epsilon=1e-08):
        Optimizer.__init__(self, mu, num_params, epsilon=epsilon)
        self.v = np.zeros(self.dim, dtype=np.float32)
        self.stepsize, self.momentum = stepsize, momentum

    def _compute_step(self, globalg):
        self.v = self.momentum * self.v + (1. - self.momentum) * globalg
        step = -self.stepsize * self.v
        return step


class Adam(Optimizer):
    def __init__(self, mu, num_params, stepsize, beta1=0.99, beta2=0.999, epsilon=1e-08):
        Optimizer.__init__(self, mu, num_params, epsilon=epsilon)
        self.stepsize = stepsize
        self.beta1 = beta1
        self.beta2 = beta2
        self.m = np.zeros(self.dim, dtype=np.float32)
        self.v = np.zeros(self.dim, dtype=np.float32)

    def _compute_step(self, globalg):
        a = self.stepsize * np.sqrt(1 - self.beta2 ** self.t) / (1 - self.beta1 ** self.t)
        self.m = self.beta1 * self.m + (1 - self.beta1) * globalg
        self.v = self.beta2 * self.v + (1 - self.beta2) * (globalg * globalg)
        step = -a * self.m / (np.sqrt(self.v) + self.epsilon)
        return step


def compute_ranks(x):
    """
    Returns ranks in [0, len(x))
    Note: This is different from scipy.stats.rankdata, which returns ranks in [1, len(x)].
    (https://github.com/openai/evolution-strategies-starter/blob/master/es_distributed/es.py)
    """
    assert x.ndim == 1
    ranks = np.empty(len(x), dtype=int)
    ranks[x.argsort()] = np.arange(len(x))
    return ranks


def compute_centered_ranks(x):
    """
    https://github.com/openai/evolution-strategies-starter/blob/master/es_distributed/es.py
    """
    y = compute_ranks(x.ravel()).reshape(x.shape).astype(np.float32)
    y /= (x.size - 1)
    y -= .5
    return y


def compute_weight_decay(weight_decay, model_param_list, reg='l2'):
    if isinstance(model_param_list, torch.Tensor):
        mean = partial(torch.mean, dim=1)
    else:
        mean = partial(np.mean, axis=1)

    if reg == 'l1':
        return - weight_decay * mean(torch.abs(model_param_list))

    return - weight_decay * mean(model_param_list * model_param_list)


class ScheduledSelectionPressure:
    """ Scheduled Selection Pressure. """
    def __init__(self, selection_pressure, num_steps, rate, mu, offset=1.):
        """ Initialize the ScheduledSelectionPressure.

        :param selection_pressure: float, final selection pressure value
        :param num_steps: int, number of steps for the scheduling
        :param rate: float, rate of the sigmoid function
        :param mu: float, center of the sigmoid function
        """
        self.selection_pressure = selection_pressure
        self.offset = offset
        self.mu = mu
        self.num_steps = num_steps
        self.rate = rate

        self.current_step = 0

    def reset(self):
        self.current_step = 0

    @property
    def scaling_factor(self):
        """ return sigmoid scaling factor based on current step and total steps """
        # alpha = self.current_step / self.num_steps
        x_adjusted = (self.current_step - self.mu) / self.num_steps
        return 1 / (1 + np.exp(-x_adjusted * self.rate))

    def get_value(self):
        value = (self.selection_pressure - self.offset) * self.scaling_factor + self.offset
        self.current_step += 1
        return value

    # override multiplication with numpy array
    def __mul__(self, other):
        return self.get_value() * other

    # override right-side multiplication with numpy array
    def __rmul__(self, other):
        return self.get_value() * other

    # override left-side multiplication with numpy array
    def __lmul__(self, other):
        return self.get_value() * other


def parameter_crowding(parameters, weight=1., sharpness=1., similarity_metric="euclidean"):
    from sklearn.metrics.pairwise import pairwise_distances
    parameter_similarity_matrix = pairwise_distances(parameters.reshape(len(parameters), -1), metric=similarity_metric)
    loss = np.exp(-parameter_similarity_matrix * sharpness)
    return loss.mean(axis=-1) * weight
