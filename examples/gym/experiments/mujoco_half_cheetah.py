""" MuJoCo Half-Cheetah environment.

see: https://www.gymlibrary.dev/environments/mujoco/half_cheetah/
"""

from . import ABCExperiment


class MujocoHalfCheetah(ABCExperiment):
    GYM_ID = "HalfCheetah-v4"

    def __init__(self, **kwargs):
        super().__init__(gym_id=self.GYM_ID, **kwargs)
