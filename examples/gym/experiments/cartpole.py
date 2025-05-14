""" Cartpole Experiment

see: https://gymnasium.farama.org/environments/classic_control/cart_pole/
"""

from . import ABCExperiment


class Cartpole(ABCExperiment):
    GYM_ID = "CartPole-v1"

    def __init__(self, **kwargs):
        super().__init__(gym_id=self.GYM_ID, **kwargs)
