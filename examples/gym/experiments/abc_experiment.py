import os
import json
import torch
from condevo.es.guidance import Condition, FitnessCondition, KNNNoveltyCondition
from condevo.es.guidance import Condition
from mindcraft import World
from mindcraft import Agent
from mindcraft import Env
from mindcraft.envs import GymWrapper
from mindcraft.agents import TorchAgent
from mindcraft.torch.module import FeedForward, Recurrent


class ABCExperiment:
    AGENT_ARCHITECTURES = ["FF", "RNN", "LSTM", "GRU"]
    AGENT_FILE = 'data/{ENV}/diff_{DIFF}/{DETAILS}/{ES}.yml'
    AGENT_DETAILS = 'agent_{ARCHITECTURE}-layers_{LAYERS}-hidden_{HIDDEN}'

    DEFAULT = {
        "agent": "FF",
        "agent_kwargs": {
            "num_hidden": 32,
            "num_layers": 1,
        },
        "world_kwargs": {
            "n_episodes": 8,
            "log_fields": ("reward", "done",),
        }
    }

    def __init__(self, gym_id, agent,
                 env_kwargs=None, agent_kwargs=None,
                 conditions=None,
                 world_kwargs=None):
        self.gym_id = gym_id
        self.env_kwargs = env_kwargs or {}

        self.agent = agent
        self.agent_kwargs = agent_kwargs or {}

        self.world_kwargs = world_kwargs or {}

        self.mindcraft_env = None
        self.mindcraft_agent = None
        self.agent_details = ""

        self.conditions = conditions or {}

    def __str__(self):
        return f"{self.__class__.__name__}({self.gym_id})"

    @property
    def num_conditions(self):
        return len(self.conditions)

    def get_conditions(self, **kwargs):
        conditions = []
        for key, value in self.conditions.items():
            if isinstance(value, str):
                value = json.loads(value)

            if isinstance(value, dict):
                foo = getattr(self, key, getattr(self, f"get_{key}"))

                try:  # try to unpack the kwargs
                    value = foo(**value, **kwargs)

                except TypeError:  # if the kwargs are not accepted (default conditions)
                    value = foo(**value)

            assert isinstance(value, Condition)
            conditions.append(value)

        return conditions

    def get_fitness_condition(self, **kwargs):
        return FitnessCondition(**kwargs)

    def get_knn_novelty_condition(self, **kwargs):
        return KNNNoveltyCondition(**kwargs)

    @property
    def env(self):
        return self.__class__.__name__.lower()

    @classmethod
    def load(cls, config, **kwargs):
        if isinstance(config, str):
            config = getattr(cls, config, cls.DEFAULT)

        assert isinstance(config, dict)
        kwargs = {**config, **kwargs}
        return cls(**kwargs)

    def get_env(self) -> Env:
        return GymWrapper(self.gym_id, **self.env_kwargs)

    def get_input_size(self):
        if self.mindcraft_env is None:
            self.mindcraft_env = self.get_env()
        return self.mindcraft_env.observation_space.shape[0]

    def get_output_size(self):
        if self.mindcraft_env is None:
            self.mindcraft_env = self.get_env()
        try:
            return self.mindcraft_env.action_space.shape[0]
        except (AttributeError, IndexError):
            return self.mindcraft_env.action_space.n

    def get_agent(self) -> Agent:
        if self.mindcraft_env is None:
            self.mindcraft_env = self.get_env()

        input_size = self.get_input_size()
        output_size = self.get_output_size()

        num_hidden = int(self.get_default("num_hidden"))
        num_layers = int(self.get_default("num_layers"))

        policy_module = None
        if self.agent == "FF":
            # define the feed-forward policy module
            policy_module = FeedForward(
                input_size=input_size,
                output_size=output_size,
                hidden_size=[num_hidden, ] * num_layers,
                activation="Tanh",
            )

        elif self.agent in ["RNN", "LSTM", "GRU", "RGRN"]:
            # define the recurrent policy module
            policy_module = Recurrent(
                input_size=input_size,
                output_size=output_size,
                hidden_size=num_hidden,
                num_layers=num_layers,
                layer_type=self.agent,
            )

        elif self.agent not in self.AGENT_ARCHITECTURES:
            raise NotImplementedError(f"Agent-architecture '{self.agent}' not implemented, "
                                      f"chose from {self.AGENT_ARCHITECTURES}.")

        from gym import spaces
        action_space = self.mindcraft_env.action_space
        clip = not isinstance(action_space, spaces.Discrete)
        default = [0] * output_size
        if not clip:
            default[0] = 1

        self.mindcraft_agent = TorchAgent(
            action_space=repr(action_space),
            default_action=default,
            clip=clip,
            retain_grad=False,
            policy_module=policy_module,
        )

        return self.mindcraft_agent

    def get_default(self, key):
        if hasattr(self, key):
            return getattr(self, key)
        if key in self.agent_kwargs:
            return self.agent_kwargs[key]
        return self.DEFAULT.get(key)

    def get_agent_file(self, diff_instance, es, timestamp=None):
        if diff_instance is None:
            diff = es
            es = "mindcraft"
        elif not isinstance(diff_instance, str):
            diff = diff_instance.__class__.__name__
        else:
            diff = diff_instance

        if timestamp:
            es = es + f"_{timestamp}"

        agent_details = self.AGENT_DETAILS.format(ARCHITECTURE=self.agent,
                                                  LAYERS=self.get_default("num_layers"),
                                                  HIDDEN=self.get_default("num_hidden"),
                                                  )

        if self.conditions:
            condition_keys = ["condition_" + (k.replace("_", "").replace("condition", ""))
                              for k in self.conditions.keys()]
            agent_details = "-".join(condition_keys) + "/" + agent_details

        agent_file = self.AGENT_FILE.format(ENV=self.env, DIFF=diff, DETAILS=agent_details, ES=es)
        return agent_file

    def get_agent_dir(self, diff_instance, es, timestamp=None):
        agent_details = self.get_agent_file(diff_instance, es, timestamp)
        return os.path.dirname(agent_details)

    def get_agent_filename(self, diff_instance, es, timestamp=None):
        agent_details = self.get_agent_file(diff_instance, es, timestamp)
        return os.path.basename(agent_details)

    def get_world(self) -> World:
        self.mindcraft_env = self.get_env()
        self.mindcraft_agent = self.get_agent()

        world_kwargs = self.world_kwargs.copy()
        world_kwargs["verbose"] = world_kwargs.get("verbose", False)
        world_kwargs["render"] = world_kwargs.get("render", False)

        import inspect
        kwargs = inspect.signature(World).parameters.keys()
        world_kwargs = {k: v for k, v in world_kwargs.items() if k in kwargs}

        world = World(env=self.mindcraft_env.to_dict(),
                      agent=self.mindcraft_agent.to_dict(),
                      **world_kwargs
                      )

        return world

    @property
    def num_params(self):
        if self.mindcraft_agent is None:
            self.mindcraft_agent = self.get_agent()
        return len(self.mindcraft_agent.get_parameters())


class PositionCondition(Condition):
    def __init__(self, target=0.5, horizon=1, agg=torch.mean, label="Agent Position", pos_observable=0):
        Condition.__init__(self)
        self.target = target
        self.agg = agg

        # helpers
        self.evaluation = None
        self.sampling = None
        self.label = label

        self.horizon = horizon
        self.pos_observable = pos_observable

    @torch.no_grad()
    def evaluate(self, charles_instance, x, f):
        # evaluate cart position from log history, array-shape (size x n_episodes)
        horizon_cart_pos = charles_instance.world_log["observation"][:, :, -self.horizon:,
                           self.pos_observable]  # (size x n_episodes x steps x features)
        horizon_cart_pos = torch.tensor(horizon_cart_pos, device=x.device, dtype=x.dtype)
        mean_cart_pos = self.agg(horizon_cart_pos.mean(dim=2), dim=1)  # mean features, agg over episodes
        self.evaluation = mean_cart_pos[charles_instance._fitness_argsort]
        return self.evaluation

    def sample(self, charles_instance, num_samples):
        sigma = (self.target - self.evaluation.mean()).abs().sqrt() * 0.5
        self.sampling = self.target + torch.randn(num_samples) * sigma

        print(self.label, "evaluation:", self.evaluation.min(), self.evaluation.mean(), self.evaluation.max())
        print(self.label, "sampling  :", self.sampling.min(), self.sampling.mean(), self.sampling.max())
        return self.sampling

    def __repr__(self):
        return f"{self.__class__.__name__}(target={self.target}, horizon={self.horizon})"
