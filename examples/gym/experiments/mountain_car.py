""" Mountain Car Experiment

see: https://gymnasium.farama.org/environments/classic_control/mountain_car/
"""

from .abc_experiment import ABCExperiment, PositionCondition
import torch


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

    def get_position_condition(self, target=0.5, horizon=1, agg="mean", label="Cart Position", pos_observable=0):
        if isinstance(agg, str):
            agg = getattr(torch, agg)

        return PositionCondition(target=target, horizon=horizon, agg=agg, label=label, pos_observable=pos_observable)


class MountainCarContinuous(MountainCar):
    GYM_ID = "MountainCarContinuous-v0"
