""" Cartpole Experiment

see: https://gymnasium.farama.org/environments/classic_control/cart_pole/
"""

from . import ABCExperiment
from condevo.es.guidance import Condition
import torch


class PositionCondition(Condition):
    def __init__(self, target=0.5, horizon=1, agg_episode=torch.mean, agg_horizon=torch.mean, label="Agent Position", pos_observable=0):
        Condition.__init__(self)
        self.target = target
        self.agg_episode = agg_episode
        self.agg_horizon = agg_horizon

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
                horizon_agg = self.agg_horizon(torch.tensor(horizon_data, device=x.device, dtype=x.dtype))
                if isinstance(horizon_agg, tuple):
                    horizon_agg, _ = horizon_agg

                eps_data.append(horizon_agg)
            population_data.append(torch.stack(eps_data, dim=0))

        agg_cart_pos = torch.stack(population_data, dim=0)  # (size x n_episodes x features)
        agg_cart_pos = self.agg_episode(agg_cart_pos, dim=1)  # (size x features)
        if isinstance(agg_cart_pos, tuple):
            agg_cart_pos, _ = agg_cart_pos

        self.evaluation = agg_cart_pos[charles_instance._fitness_argsort]
        return self.evaluation

    def sample(self, charles_instance, num_samples):
        # sample around the target position, with STD related to the minimum distance to the target
        min_dist = torch.min(torch.abs(self.evaluation - self.target))
        self.sampling = self.target + torch.randn(num_samples) * torch.sqrt(min_dist)

        print(self.label, "evaluation (min, mean(STD), max):", self.evaluation.min().item(), self.evaluation.mean().item(), "(+-", self.evaluation.std().item(), ")", self.evaluation.max().item())
        print(self.label, "sampling   (min, mean(STD), max):", self.sampling.min().item(), self.sampling.mean().item(), "(+-", self.sampling.std().item(), ")", self.sampling.max().item())
        return self.sampling

    def __repr__(self):
        return f"{self.__class__.__name__}(target={self.target}, horizon={self.horizon})"


class Cartpole(ABCExperiment):
    GYM_ID = "CartPole-v1"

    def __init__(self, **kwargs):
        super().__init__(gym_id=self.GYM_ID, **kwargs)

    def get_position_condition(self, target=0.5, horizon=30, agg_episode="mean", agg_horizon="mean", label="Cart Position", pos_observable=0):
        if isinstance(agg_episode, str):
            agg_episode = getattr(torch, agg_episode)

        if isinstance(agg_horizon, str):
            agg_horizon = getattr(torch, agg_horizon)

        return PositionCondition(target=target, horizon=horizon, agg_episode=agg_episode, agg_horizon=agg_horizon, label=label, pos_observable=pos_observable)
