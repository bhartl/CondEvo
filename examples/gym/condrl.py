import os
from typing import Optional
import datetime
import utils

def import_experiment(name):
    """ Imports an experiment """
    # import python file in experiments subdirectory
    import experiments
    experiment = getattr(experiments, name)
    return experiment


def list_experiments():
    """ Lists all experiments """
    files = [f for f in os.listdir("experiments") if f.endswith(".py") and f not in ["__init__.py", ]]
    for f in files:
        # import python file in experiments subdirectory
        name = f[:-3]
        experiment = import_experiment(name)
        print(f"{f[:-3]}: {experiment.__doc__}")


def get_timestamp(timestamp: bool = False):
    if timestamp:
        if isinstance(timestamp, bool):
            # get timestamp in milliseconds
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
        return timestamp
    return None


def train(experiment: str ="Cartpole",
          experiment_config: str = "DEFAULT",
          conditions: dict = None,
          es: str = "HADES",
          es_config: Optional[dict] = None,
          generations: int = 100,
          diff: str = "DDIM",
          diff_config: dict = None,
          nn: str = "MLP",
          nn_config: dict = None,
          es_logs: dict = None,
          new_model: bool = False,
          quiet: bool = False,
          timestamp: bool = False,
          ):
    """ Trains an RL agent on a specified experiment """

    # load experiment
    if not quiet:
        print(f"# Loading `{experiment}`-Experiment:")
    experiment_module = import_experiment(experiment)
    experiment_config = utils.load_experiment(experiment_config)
    conditions = utils.load_file(conditions) if (es == "CHARLES" and conditions) else {}
    experiment = experiment_module.load(experiment_config, conditions=conditions)

    if not quiet:
        print(f"- {experiment}")

    if es in ("HADES", "CHARLES"):
        # load Diffusion Model ES
        if es == "HADES":
            from condevo.es import HADES
            es = HADES

        elif es == "CHARLES":
            from condevo.es import CHARLES
            es = CHARLES

        nn_instance, diff_instance = None, None
        if nn is not None and diff is not None:
            # load diffusion config
            if not quiet:
                print(f"# Loading Neural Network:")
                print(f"- {nn}")
            nn_instance, nn_config = utils.load_nn(nn, nn_config, experiment.num_params, experiment.num_conditions)
            if not quiet:
                print(f"- {utils.to_json(nn_config)}")

            # load diffusion config
            if not quiet:
                print(f"# Loading Diffusion Model:")
                print(f"- {diff}")
            diff_instance, diff_config = utils.load_diffuser(diff, diff_config, nn_instance)
            if not quiet:
                print(f"- {utils.to_json(diff_config)}")

    else:
        nn_instance, diff_instance = None, None

    # load solver config
    es_name = es if isinstance(es, str) else es.__name__
    if not quiet:
        print(f"# Loading Evolutionary Strategy")
        print(f"-  {es_name}")
    solver, es_config = utils.load_es(es, es_config, diff_instance, experiment.num_params)
    if not quiet:
        print(f"- {utils.to_json(es_config)}")

    # load mindcraft env
    timestamp = get_timestamp(timestamp)
    agent_path = experiment.get_agent_dir(diff_instance=diff_instance, es=es_name, timestamp=timestamp)
    agent_filename = experiment.get_agent_filename(diff_instance=diff_instance, es=es_name, timestamp=timestamp)
    os.makedirs(agent_path, exist_ok=True)

    from mindcraft.script import train as train_mindcraft
    world = experiment.get_world()

    if es_name == "CHARLES":
        conditions = experiment.get_conditions()
        es_config["conditions"] = conditions

    if not new_model and os.path.exists(os.path.join(agent_path, agent_filename)):
        if not quiet:
            print(f"# Loading existing agent: {os.path.join(agent_path, agent_filename)}")
        world.agent = os.path.join(agent_path, agent_filename)

    return train_mindcraft(
        world_repr=world,
        es=es,
        generations=generations,
        verbose=not quiet,
        path=agent_path,
        file_name=agent_filename,
        opts=es_config,
        size=es_config["popsize"],
        new_model=new_model,
        static_world=True,
        **(es_logs or utils.configs.ES_LOGS)
    )


def rollout(experiment: str ="Cartpole", experiment_config: str = "DEFAULT", conditions=None,
            diff: str = "DDIM", es: str = "HADES", n_episodes: int = 10, timestamp: Optional[str] = None, **kwargs):
    """ Run HADES-trained rollouts in a gym environment

    :param n_episodes: Number of episodes to run.
    """

    # load experiment
    experiment_module = import_experiment(experiment)
    experiment_config = utils.load_experiment(experiment_config)
    conditions = utils.load_file(conditions) if (es == "CHARLES" and conditions) else {}
    experiment = experiment_module.load(experiment_config, conditions=conditions)

    experiment.env_kwargs["gym_kwargs"] = {**experiment.env_kwargs.get("gym_kwargs", {}), **{"render_mode": "human"}}
    env = experiment.get_env()

    agent = experiment.get_agent_file(diff_instance=diff, es=es, timestamp=timestamp)

    mindcraft_config = dict(world=dict(env=env, agent=agent, verbose=True, render=True,  n_episodes=n_episodes, delay=0.0),
                            rollout_kwargs=dict(verbose=True))
    from mindcraft import rollout as run
    return run(config=mindcraft_config)


def progress(experiment: str ="Cartpole", experiment_config: str = "DEFAULT", conditions=None,
             diff: str = "DDIM", es: str = "HADES", n_episodes: int = 10, timestamp: Optional[str] = None, **kwargs):

    experiment_module = import_experiment(experiment)
    experiment_config = utils.load_experiment(experiment_config)
    conditions = utils.load_file(conditions) if (es == "CHARLES" and conditions) else {}
    experiment = experiment_module.load(experiment_config, conditions=conditions)
    experiment.env_kwargs["gym_kwargs"] = {**experiment.env_kwargs.get("gym_kwargs", {}), **{"render_mode": "human"}}
    agent = experiment.get_agent_file(diff_instance=diff, es=es, timestamp=timestamp)

    from mindcraft.train import EvolutionaryStrategy
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    log_file = agent.replace(".yml", ".log")
    if not os.path.exists(log_file):
        raise FileNotFoundError(log_file)

    runs = EvolutionaryStrategy.load_log(log_file, to_pandas=True)
    print(f"Accessing Log-File `{log_file}`")
    print(f"History: {len(runs)} runs, {len([r for r in runs if r['tail'] != []])} of them finished")

    p_start = 0
    best = []
    cost = []
    mean = []
    std = []
    steps = []
    boosts = []
    for i in range(len(runs)):
        data = runs[i]['data']
        if not len(data):
            continue

        best.append(data['best'])
        cost.append(data['cost'])
        mean.append(data['mean'])
        std.append(data['std'])
        steps.append(data['step'] + p_start)

        boosts.append(max(data['step']))
        p_start += boosts[-1] + 1

    best = np.concatenate(best)
    cost = np.concatenate(cost)
    mean = np.concatenate(mean)
    std = np.concatenate(std)
    steps = np.concatenate(steps)

    plt.figure(figsize=(15, 5))
    ax = plt.gca()

    # plot_foo(steps, mean, label='mean')
    plt.plot(steps, best, label='Hist. Fittest', zorder=2, linewidth=2)
    plt.fill_between(steps, cost, best, label='Current Fittest', zorder=3, alpha=0.5)
    plt.plot(steps, mean, color="black", zorder=1, label="Pop. Mean", linewidth=2)
    plt.fill_between(steps, mean-std, np.minimum(best, mean+std), zorder=3, alpha=0.5, color="gray", label="Pop. Std.")

    ax.set_xlim([1, max(steps)])

    d_min = min([min(mean), min(cost)])
    d_max = max([max(best), max(cost)])
    offset = 0.05 * (d_max - d_min)

    # for b in np.cumsum(boosts):
    #     plot_foo([b, b], [d_min - offset, d_max + offset], 'k--')
    ax.set_ylim([d_min - offset, d_max + offset])

    ax.set_xlabel('Generations')
    ax.set_ylabel('Return (Cumulative Reward)')

    plt.grid()
    plt.legend()
    plt.show()


def demo(experiment: str ="Cartpole", experiment_config: str = "DEFAULT",):
    """ Run a simple demo """
    experiment_module = import_experiment(experiment)
    experiment_config = utils.load_experiment(experiment_config)
    experiment = experiment_module.load(experiment_config)

    experiment.env_kwargs["gym_kwargs"] = {**experiment.env_kwargs.get("gym_kwargs", {}), **{"render_mode": "human"}}
    env = experiment.get_env()
    env.reset()

    for i in range(1000):
        env.step(env.action_space.sample())
        env.render()


def run(config, method="train", timestamp=None):
    config = utils.load_file(config)
    method = globals()[method]
    if timestamp:
        config["timestamp"] = timestamp

    if config["es"] not in ("HADES", "CHARLES"):
        print(f"Assuming diff {config['es']}")
        config["diff"] = config["es"]
        config["es"] = "mindcraft"

    return method(**config)


if __name__ == '__main__':
    import argh
    argh.dispatch_commands([list_experiments,
                            train,
                            rollout,
                            demo,
                            run,
                            ])