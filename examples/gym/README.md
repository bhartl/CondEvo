# Reinforcement Learning Examples
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
