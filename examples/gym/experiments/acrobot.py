""" Acrobot Experiment

see: https://gymnasium.farama.org/environments/classic_control/acrobot/
"""

from .abc_experiment import ABCExperiment


class Acrobot(ABCExperiment):
    GYM_ID = "Acrobot-v1"
    def __init__(self, **kwargs):
        super().__init__(gym_id=self.GYM_ID, **kwargs)
