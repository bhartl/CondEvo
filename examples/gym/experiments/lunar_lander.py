""" Lunar Lander Experiment

see: https://gymnasium.farama.org/environments/box2d/lunar_lander/
"""

from . import ABCExperiment


class LunarLander(ABCExperiment):
    GYM_ID = "LunarLander-v2"

    def __init__(self, **kwargs):
        super().__init__(gym_id=self.GYM_ID, **kwargs)

    @property
    def num_conditions(self):
        return 0


class LunarLanderContinuous(ABCExperiment):
    GYM_ID = "LunarLanderContinuous-v2"
    def __init__(self, **kwargs):
        super().__init__(gym_id=self.GYM_ID, **kwargs)
