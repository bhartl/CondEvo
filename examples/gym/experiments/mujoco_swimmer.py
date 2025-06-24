""" MuJoCo Swimmer environment.

see: https://www.gymlibrary.dev/environments/mujoco/swimmer/
"""

from . import ABCExperiment


class MujocoSwimmer(ABCExperiment):
    GYM_ID = "Swimmer-v4"

    def __init__(self, **kwargs):
        super().__init__(gym_id=self.GYM_ID, **kwargs)
