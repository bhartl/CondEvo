""" Lunar Lander Experiment

see: https://gymnasium.farama.org/environments/box2d/lunar_lander/
"""

from . import ABCExperiment
from condevo.es.guidance import Condition
import torch
import numpy as np


class GroundThrustCondition(Condition):
    """ Main condition to train a gym environment with a CHARLES-Diffusion module, to bring the lander to turn
        off the thruster after it has landed.
    """
    def __init__(self, agg_episode=torch.mean, label="Ground Thrust", is_continuous=False):
        Condition.__init__(self)
        self.agg_episode = agg_episode
        self.is_continuous = is_continuous

        self.legs_obs = [-2, -1]
        self.thrust_obs = [0, 1] if is_continuous else [1, 2, 3]

        # helpers
        self.evaluation = None
        self.sampling = None
        self.label = label

    @torch.no_grad()
    def evaluate(self, charles_instance, x, f):
        # from array-shape (size x n_episodes x *obs_shape) -> (size x n_episodes x horizon steps x features)

        # retrieve the ground contact and thrust from the observation log
        obs_data = []
        for pd in charles_instance.world_log["observation"]:
            eps_data = []
            for eps in pd:
                ground_contact = eps[:, self.legs_obs].any(axis=1)  # ground contact if any of the last two observations are true
                eps_data.append(torch.tensor(ground_contact, device=x.device, dtype=x.dtype))
            obs_data.append(eps_data)

        # retrieve thrust from actions
        thrust_contact_data = []
        for pd, obs_d in zip(charles_instance.world_log["action"], obs_data):
            eps_data = []
            for eps, contact in zip(pd, obs_d):
                thrust_level = np.linalg.norm(eps[:, self.thrust_obs], axis=-1)  # extract total thrust level from actions
                thrust_level = torch.tensor(thrust_level, device=x.device, dtype=x.dtype)
                contact_thrust = contact * thrust_level  # cound thrust only when ground contact is true
                contact_thrust = contact_thrust.sum()  # sum over the time steps
                eps_data.append(contact_thrust)
            thrust_contact_data.append(torch.stack(eps_data, dim=0))

        thrust_contact_data = torch.stack(thrust_contact_data, dim=0)  # (size x n_episodes x horizon steps)

        obs_data = self.agg_episode(thrust_contact_data, dim=1)  # (size x features)
        if isinstance(obs_data, tuple):
            agg_cart_pos, _ = obs_data

        self.evaluation = obs_data[charles_instance._fitness_argsort]
        return self.evaluation

    def sample(self, charles_instance, num_samples):
        # sample lander agents which don't use the ground thrust
        self.sampling = torch.zeros(num_samples, device=self.evaluation.device, dtype=self.evaluation.dtype)

        print(self.label, "evaluation:", self.evaluation.min(), self.evaluation.mean(), self.evaluation.max())
        print(self.label, "sampling  :", self.sampling.min(), self.sampling.mean(), self.sampling.max())
        return self.sampling

    def __repr__(self):
        return f"{self.__class__.__name__}(agg_episode={self.agg_episode}, is_continuous={self.is_continuous})"


class LunarLander(ABCExperiment):
    GYM_ID = "LunarLander-v2"

    def __init__(self, **kwargs):
        super().__init__(gym_id=self.GYM_ID, **kwargs)

    def get_ground_thrust(self, agg_episode="mean", label="Lander Ground-Thrust"):
        if isinstance(agg_episode, str):
            agg_episode = getattr(torch, agg_episode)

        return GroundThrustCondition(agg_episode=agg_episode, label=label, is_continuous=False)


class LunarLanderContinuous(ABCExperiment):
    GYM_ID = "LunarLanderContinuous-v2"
    def __init__(self, **kwargs):
        super().__init__(gym_id=self.GYM_ID, **kwargs)

    def get_ground_thrust(self, agg_episode="mean", label="Lander Ground-Thrust"):
        if isinstance(agg_episode, str):
            agg_episode = getattr(torch, agg_episode)

        return GroundThrustCondition(agg_episode=agg_episode, label=label, is_continuous=True)
