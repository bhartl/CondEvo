""" Mountain Car Experiment

see: https://gymnasium.farama.org/environments/classic_control/mountain_car/
"""

from .abc_experiment import ABCExperiment
from condevo.es.guidance import Condition
import torch
import numpy as np


class XRangeCondition(Condition):
    def __init__(self, horizon=200, agg=torch.mean, label="Agent Position", pos_observable=0):
        Condition.__init__(self)
        self.agg = agg

        # helpers
        self.evaluation = None
        self.sampling = None
        self.label = label

        self.horizon = horizon
        self.pos_observable = pos_observable

    @torch.no_grad()
    def evaluate(self, charles_instance, x, f):
        # evaluate cart position from log history
        # from array-shape (size x n_episodes x *obs_shape) -> (size x n_episodes x horizon steps x features)

        # horizon_cart_pos = charles_instance.world_log["observation"][:, :, -self.horizon:, self.pos_observable]
        # horizon_cart_pos = torch.tensor(horizon_cart_pos, device=x.device, dtype=x.dtype)
        # mean_cart_pos = self.agg(horizon_cart_pos.mean(dim=2), dim=1)   # mean features, agg over episodes

        # -> time dimension can be different, thus we need to aggregate over episodes
        population_data = []
        for pd in charles_instance.world_log["observation"]:
            eps_data = []
            for eps in pd:
                horizon_data = [d[self.pos_observable] for d in eps[-self.horizon:]]
                min_horizon = np.min(horizon_data)
                max_horizon = np.max(horizon_data)
                x_range = max_horizon - min_horizon
                eps_data.append(x_range)

            population_data.append(torch.tensor(eps_data, device=x.device, dtype=x.dtype))

        agg_cart_pos = torch.stack(population_data, dim=0)  # (size x n_episodes x features)
        agg_cart_pos = self.agg(agg_cart_pos, dim=1)  # (size x features)
        if isinstance(agg_cart_pos, tuple):
            agg_cart_pos, _ = agg_cart_pos

        self.evaluation = agg_cart_pos[charles_instance._fitness_argsort]
        return self.evaluation

    def sample(self, charles_instance, num_samples):
        # sample around the target position, with STD related to the minimum distance to the target
        max_xrange = torch.max(self.evaluation)
        self.sampling = max_xrange + torch.randn(num_samples) * 0.1  # small noise for sampling

        print(self.label, "evaluation:", self.evaluation.min(), self.evaluation.mean(), self.evaluation.max())
        print(self.label, "sampling  :", self.sampling.min(), self.sampling.mean(), self.sampling.max())
        return self.sampling

    def __repr__(self):
        return f"{self.__class__.__name__}(horizon={self.horizon}, agg={self.agg}, " \
               f"label='{self.label}', pos_observable={self.pos_observable})"


class MountainCar(ABCExperiment):
    GYM_ID = "MountainCar-v0"
    DEFAULT = {
        "agent": "RGRN",
        "agent_kwargs": {
            "num_hidden": 4,
            "num_layers": 1,
        },
        "world_kwargs": {
            "n_episodes": 16,
            "log_fields": ("reward", "done",),
        }
    }

    def __init__(self, **kwargs):
        kwargs["gym_id"] = kwargs.get("gym_id", self.GYM_ID)
        super().__init__(**kwargs)

    def get_x_range_condition(self, horizon=200, agg="mean", label="Cart Position", pos_observable=0):
        if isinstance(agg, str):
            agg = getattr(torch, agg)

        return XRangeCondition(horizon=horizon, agg=agg, label=label, pos_observable=pos_observable)


class MountainCarContinuous(MountainCar):
    GYM_ID = "MountainCarContinuous-v0"
