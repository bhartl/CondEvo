""" MuJoCo Ant environment.

see: https://www.gymlibrary.dev/environments/mujoco/ant/
"""

from . import ABCExperiment


class MujocoAnt(ABCExperiment):
    GYM_ID = "Ant-v4"
    GMY_KWARGS = {
        "healthy_reward": 0.01,
        "terminate_when_unhealthy": True,
    }
    """ Default kwargs for the Ant environment, see https://www.gymlibrary.dev/environments/mujoco/ant/#arguments    
    """

    DEFAULT = {
        "agent": "FF",
        "agent_kwargs": {
            "num_hidden": 32,
            "num_layers": 2,
        },
        "world_kwargs": {
            "n_episodes": 4,
            "log_fields": ("reward", "done",),
        }
    }

    def __init__(self, **kwargs):
        env_kwargs = kwargs.pop("env_kwargs", {})
        gym_kwargs = env_kwargs.pop("gym_kwargs", {})
        gym_kwargs = {**self.GMY_KWARGS, **gym_kwargs}
        env_kwargs["gym_kwargs"] = gym_kwargs
        super().__init__(gym_id=self.GYM_ID, env_kwargs=env_kwargs, **kwargs)

    @property
    def num_conditions(self):
        return 0
