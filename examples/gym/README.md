# Reinforcement Learning Examples
The `gym` experiments demonstrates how to use diffusion models and evolutionary strategies together for solving reinforcement learning problems.

## Installation
Requires the `mindcraft` packages. Please refer to the [mindcraft README](https://github.com/bhartl/NeurEvo/?tab=readme-ov-file#install) for installation instructions.

## Examples
### Classic Control Tasks in Gym
Via `mindcraft`, we provide APIs to some [classic control tasks](https://www.gymlibrary.dev/environments/classic_control/), such as [CartPole](experiments/cartpole.py)  or [MountanCar](experiments/mountain_car.py).

### Box2D Tasks in Gym
Via `mindcraft`, we provide APIs to some [Box2D tasks](https://www.gymlibrary.dev/environments/box2d/) such as
[LunarLander](experiments/lunar_lander.py)(Continuous).

## Gym Example Usage
### HADES, CMA-ES, and SimpleGA
Run a random agent **demo** experiment with the following command:
```bash
python -m condrl demo --experiment Cartpole
```
or use any other experiment name such as `Acrobot`, `LunarLanderContinuous` etc.

**Train** an agent with the following command:
```bash
python -m condrl train --experiment Cartpole --generations 10
```
or use the `MPI` framework if available:
```bash
mpirun -np 8 python -m condrl train --experiment Cartpole --generations 10
```

and **test** a trained agent with the following command:
```bash
python -m condrl rollout --experiment Cartpole
```

please note the `--es` flag to specify the evolutionary strategy, e.g., `--es HADES`, `--es CMAES`, or `--es SimpleGA`. This can be used for training and rollout.

Checkout the documentation for more details on the available options (such as using different solvers, etc).

### CHARLES
Conditional evolution can be run with the following command:
```bash
python -m condrl train --experiment Cartpole --generations 10 --es CHARLES
```

Or start from a config-file via
```bash
python -m condrl run config/cartpole_charles.yml --method train
```

and test with
```bash
python -m condrl run config/cartpole_charles.yml --method rollout
```

Per default, the `CHARLES` algorithm uses `FitnessCondition` to sample high-fitness individuals with greater probability. 
This can be changed by setting the `--condition` flag to any other specified condition implemented in the respective experiment file. 

## Train from config
Configs are located in the [./config](config) directory.
They are written in `yml` format, and define all details necessary to train a gym experiment.

To train e.g. a cartpole agent, simply run (below with `MPI` on 8 cores):
```bash
mpirun -n 8 python condrl.py run config/cartpole_charles_pos.yml
```

The `run` script in `condrl.py` loads the provided `config` file, and forwards the loaded dictionary as kwargs to the specified `method` (again implemented in `condrl.py`, e.g., `train`, `rollout`, etc).

A config, such as [config/cartpole_charles_pos.yml](config/cartpole_charles_pos.yml), contains all details for an experiment, such as
 - which experiment to use (e.g., Cartpole, MountainCar, etc.), see wrapped environments in [./experiments](experiments)
 - how to set up the agent (what ANN architecture to use, etc) and the world (how many episodes, state logs, etc)
 - what `es` to use (e.g., HADES) and all `es_config` settings (e.g., popsize, etc.)
 - what `diff` (diffusion model) to use (see `condevo.diffusion`)
 - what network to use for the diffusion model

Note: be aware of the `timestamp` argument -> that's used to support multiple evaluations in parallel, so file-names won't interfere. Disable it in the config for training, or use the timestamp as argument to run specific rollouts.

## Gym Wrapper
In general, Gym environments are wrapped via the [ABCExperiment](experiments/abc_experiment.py) class. It defines how an agent can be defined (e.g., which neural network architecture) and how a world is composed (number of episodes, max steps, state logs).

Also, it handles conditions for CHARLES training. 
- Conditions need to be passed as dictionaries, where the key specifies the method name (e.g., `"fitness_condition"`, or `"knn_novelty_condition"`), while the value specifies `kwargs` that are passed to the condition method of the derivate of the `ABCExperiment` gym wrapper. 
- The gym wrapper will evaluate the condition by calling a corresponding `get_<key>(**kwargs)` method (e.g., `get_fitness_condition`). 
- A custom example for a behavioral condition is given by the [MountainCar experiment](experiments/mountain_car.py), which implements a custom `XRangeCondition` condition, which is linked to the `get_x_range_condition(**kwargs)` method of the `MountainCar` class, and can be controlled via the `x_range_condition` in the respective [config/mountain_car_conditional.yml](config/mountain_car_conditional.yml) config. The condition tries to maximize the horizontal range a car is experiencing during an episode. This helps the agent to build up enough momentum to solve the task, without reward shaping.
- Another custom example is given by the [Cartpole experiment](experiments/cartpole.py), which implements a `PositionCondition` class (dervied from `condevo.es.guidance.Condition`), links to the `get_position_condition` method of the `Cartpole` class, and defines a condition/target for the resting position of the cart pole at the end of an episode. The condition can e.g. be controlled via the `position_condition` block in [config/cartpole_charles_pos.yml](config/cartpole_charles_pos.yml) config (please note that this condition can be brittle, and might depend on the chosen diffuser, agent ANN, MLP, etc.).
