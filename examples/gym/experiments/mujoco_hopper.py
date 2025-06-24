""" MuJoCo Hopper environment.

see: https://www.gymlibrary.dev/environments/mujoco/hopper/
"""

from . import ABCExperiment


class MujocoHopper(ABCExperiment):
    GYM_ID = "Hopper-v4"
    GMY_KWARGS = {
        "healthy_reward": 0.01,
        "terminate_when_unhealthy": True,
    }
    """ Default kwargs for the Ant environment, see https://www.gymlibrary.dev/environments/mujoco/hopper/#arguments """

    def __init__(self, **kwargs):
        env_kwargs = kwargs.pop("env_kwargs", {})
        gym_kwargs = env_kwargs.pop("gym_kwargs", {})
        gym_kwargs = {**self.GMY_KWARGS, **gym_kwargs}
        env_kwargs["gym_kwargs"] = gym_kwargs
        super().__init__(gym_id=self.GYM_ID, env_kwargs=env_kwargs, **kwargs)
