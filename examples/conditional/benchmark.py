import os
import numpy as np
import h5py
import pandas as pd
import json

# load benchmark utils
import utils
import configs


def run(objective="{\"foo\": \"rastrigin\", \"limits\": 3}",
        generations=20,
        nn="MLP",
        nn_config=None,
        diff="DDIM",
        diff_config=None,
        es="HADES",
        es_config=None,
        dst=utils.DST,
        quiet=False,
        timestamp=False,
        ) -> tuple:
    """ Test HADES on a ND objective function.

    :param objective: str or dict, representation of foo.benchmarks.Objective, objective function to optimize.
                      `objective` can be a *string*, in which case it is assumed to be a function in `foobench` module.
                      Alternatively, it can be a *dictionary* or *json string* specifying the kwargs of the
                      `foobench.Objective` class.
                      With the "foo" key, the objective function can be imported from the `foobench` module, with
                       optional kwargs passed as "foo_kwargs" dictionary.
                      **Note** that "foo" and "foo_module" keys can be used together to specify the python module of
                        the objective function, thus allowing to import custom objective functions.
                      If the objective function is implemented as callable class, the "foo_kwargs" are passed as
                        kwargs to the constructor of the "foo" class. Otherwise, the "foo_kwargs" are passed as
                        kwargs to the "foo" function call.
    :param generations: int, number of generations to run
    :param nn: str, name of the `condevo.nn` neural network class to use for the diffusion model, defaults to `MLP`.
    :param nn_config: str or dict representation of the kwargs of the neural network, may also be a file path
                        or json string. If not provided, the default `<nn>` configuration is used from `configs.py`.
    :param diff: str, name of the diffusion model to use (e.g. DDIM, RectFlow), defaults to `DDIM`.
    :param diff_config: str or dict, configuration of the diffusion model, may also be a file path or json string.
                          If not provided, the default `<diff>` configuration is used from `configs.py`.
    :param es: str, name of the evolutionary strategy to use (e.g. HADES, CHARLES, CMAES, [WIP: PEPG, OpenES, SimpleGA])
    :param es_config: str or dict, configuration of the evolutionary strategy, may also be a file path or json string.
                        If not provided, the default `<es>` configuration is used from `configs.py`.
    :param dst: str, destination folder to save the results, defaults to `utils.DST`.
                **Note: make sure to provide conflict-free destination folder in case of large scale runs.**
    :param quiet: bool, whether to suppress the output during runtime
    :return: tuple of (h5_filename, run_id, best_solution, best_fitness), the h5 data may be accessed with the
             `load_run` function.
    """

    from foobench import Objective

    # load objective
    if not quiet:
        print(f"# Loading Objective:")
    objective_instance = Objective.load(objective)
    if not quiet:
        print(f"-  {objective}")

    nn_instance, diff_instance = None, None

    if nn is not None and diff is not None:
        # load diffusion config
        if not quiet:
            print(f"# Loading Neural Network:")
            print(f"- {nn}")
        nn_instance, nn_config = utils.load_nn(nn, nn_config, objective_instance.dim)
        if not quiet:
            print(f"- {utils.to_json(nn_config)}")

        # load diffusion config
        if not quiet:
            print(f"# Loading Diffusion Model:")
            print(f"- {diff}")
        diff_instance, diff_config = utils.load_diffuser(diff, diff_config, nn_instance)
        if not quiet:
            print(f"- {utils.to_json(diff_config)}")

    # load solver config
    if not quiet:
        print(f"# Loading Evolutionary Strategy")
        print(f"-  {es}")
    solver, es_config = utils.load_es(es, es_config, diff_instance, objective_instance.dim)
    if not quiet:
        print(f"- {utils.to_json(es_config)}")

    # for logging: create h5 file
    es_name = es if isinstance(es, str) else es.__name__
    h5_filename = os.path.join(dst, utils.H5_FILE.format(ES=es_name, objective=objective_instance.foo_name))

    if timestamp:
        if not isinstance(timestamp, str):
            from datetime import datetime
            timestamp = datetime.utcnow().strftime('%Y%m%d%H%M%S%f')[:-3]
        h5_filename = h5_filename.replace(".h5", f"-{timestamp}.h5")

    os.makedirs(dst, exist_ok=True)
    with h5py.File(h5_filename, 'a') as f:
        runs = f.keys()
        num_run = len(np.unique([str(r) for r in runs]))
        print(f"# Results are saved \n- in folder `{dst}`\n- as `{h5_filename}`\n- with run_id `{num_run}`.")

        # create group for run
        run = f.create_group(f'run_{num_run}')

        # add attributes to run
        run.attrs['objective'] = repr(objective_instance)
        run.attrs['generations'] = generations
        if diff is not None:
            run.attrs['nn'] = nn if isinstance(nn, str) else nn.__name__
            run.attrs['nn_config'] = utils.to_json(nn_config)
            run.attrs['diff'] = diff if isinstance(diff, str) else diff.__name__
            run.attrs['diff_config'] = utils.to_json(diff_config)
        run.attrs['es'] = es if isinstance(es, str) else es.__name__
        run.attrs['es_config'] = utils.to_json(es_config)
        run.attrs['dst'] = dst
        run.attrs['quiet'] = quiet

    # run SOLVER
    i, fitness, model_loss = 0, [], [0]
    for i in range(generations):
        # sample solutions and evaluate them
        samples = solver.ask()
        fitness = objective_instance(samples)

        # train the diffusion model on the best solution
        model_loss = solver.tell(fitness)

        if not quiet and diff is not None:
            print("  {" + f" \"Generation\": {i}, \"Max-Fitness\": {fitness.max()}, \"Avg-Fitness\": {fitness.mean()}, \"Model-Loss\": {model_loss[-1]}" + "}")
        elif not quiet:
            print("  {" + f" \"Generation\": {i}, \"Max-Fitness\": {fitness.max()}, \"Avg-Fitness\": {fitness.mean()}" + "}")

        # convert to numpy arrays if necessary
        samples = samples.numpy() if not isinstance(samples, np.ndarray) else samples
        fitness = fitness.numpy() if not isinstance(fitness, np.ndarray) else fitness
        model_loss = np.array(model_loss)

        with h5py.File(h5_filename, 'a') as f:
            f.create_dataset(f'run_{num_run}/gen_{i}/samples', data=samples)
            f.create_dataset(f'run_{num_run}/gen_{i}/fitness', data=fitness)
            if diff is not None:
                f.create_dataset(f'run_{num_run}/gen_{i}/model_loss', data=model_loss)

    if not quiet and diff is not None:
        print("  {" + f" \"Generation\": {i}, \"Max-Fitness\": {fitness.max()}, \"Avg-Fitness\": {fitness.mean()}, \"Model-Loss\": {model_loss[-1]}" + "}")
    elif not quiet:
        print("  {" + f" \"Generation\": {i}, \"Max-Fitness\": {fitness.max()}, \"Avg-Fitness\": {fitness.mean()}" + "}")

    return h5_filename, num_run, solver.result()[0], solver.result()[1]


#### BENCHMARKS ####

from foobench import Objective
from foobench.dynamic import OscillatingDoubleDip
from ast import literal_eval

# define objective function shortcuts
FOOS = {
    "DoublePeak": Objective(foo="double_dip", foo_kwargs={"m": 1.0}, maximize=True),
    "Rastrigin": Objective(foo="rastrigin", maximize=False, limits=4, limit_val=0., apply_limits=True),
    # "rosenbrock": Objective(foo="rosenbrock", maximize=False, limits=4, limit_val=0., apply_limits=True),
    "Himmelblau": Objective(foo="himmelblau", maximize=True, limits=4, limit_val=-100., apply_limits=True),
    "DynamicDoublePeak": Objective(foo=OscillatingDoubleDip(omega=np.pi*0.2, phi=np.pi), foo_kwargs=dict(m=0.625), maximize=True),
}

def objective_names():
    return list(FOOS.keys())


def objectives():
    return [f"{k}: {repr(FOOS[k])}" for k in FOOS]


# benchmark default parameters
NUM_RUNS = 20
GENERATIONS = 40
NN = "MLP"
NN_CONFIG = None
DIFF = "DDIM"
DIFF_CONFIG = None
ES = "HADES"
ES_CONFIG = None
QUIET = False


DST_param = "data/benchmark/"

def eval_es_parameter(foo,
                      param_names,
                      param_vals,
                      elitism=False,
                      adaptive_selection_pressure=False,
                      readaptation=False,
                      num_runs=NUM_RUNS,
                      generations=GENERATIONS,
                      nn=NN,
                      nn_config=NN_CONFIG,
                      diff=DIFF,
                      diff_config=DIFF_CONFIG,
                      es=ES,
                      es_config=ES_CONFIG,
                      dst=DST_param,
                      quiet=QUIET,
                      ):
    """ run benchmark with particular es-config parameter

    :param foo: key of the objective function in `FOOS`
    :param param_names: list of names of the ES-config parameter for the benchmark
    :param param_vals: corresponding values of the ES-config parameters for the benchmark, will be evaluated via `literal_eval`
    :param elitism: bool, whether to forget the best solution, or keep the elites for the next generation
    :param num_runs: number of runs for the benchmark, will be stored in the output file

    the remaining parameters are the same as in `benchmark.run`
    """

    # load objective
    objective = FOOS[foo]

    # check arguments
    if not isinstance(param_names, (list, tuple)):
        param_names = [param_names]

    if not isinstance(param_vals, (list, tuple)):
        param_vals = [param_vals]

    if "adaptive_selection_pressure" not in param_names:
        param_names.append("adaptive_selection_pressure")
        param_vals.append(adaptive_selection_pressure)

    if "forget_best" not in param_names:
        param_names.append("forget_best")
        param_vals.append(not elitism)

    if "readaptation" not in param_names:
        param_names.append("readaptation")
        param_vals.append(readaptation)

    # set es config
    es_config = {**getattr(configs, es)} if es_config is None else {**es_config}
    path = []
    for param_name, param_val in zip(param_names, param_vals):
        path.append(f"{param_name.replace('_', '')}_{param_val}")
        try:
            es_config[param_name] = literal_eval(param_val)
        except:
            es_config[param_name] = param_val
    dst = os.path.join(dst, '-'.join(path))

    # check if conditioning solver is used
    if nn_config is None and es == "CHARLES":
        # add condition for FitnessCondition (default)
        nn_config = getattr(configs, nn)
        nn_config = {"num_conditions": 1, **nn_config}

    # run experiments
    runs = []
    for r in range(num_runs):
        rc = run(objective=objective,
                 generations=generations,
                 nn=nn,
                 nn_config=nn_config,
                 diff=diff,
                 diff_config=diff_config,
                 es=es,
                 es_config=es_config,
                 quiet=quiet,
                 dst=dst)
        runs.append(rc)

    return runs


def eval_buffer_size(buffer_size=3,
                     elitism=False,
                     adaptive_selection_pressure=False,
                     foo="DoublePeak",
                     num_runs=NUM_RUNS,
                     generations=GENERATIONS,
                     nn=NN,
                     nn_config=NN_CONFIG,
                     diff=DIFF,
                     diff_config=DIFF_CONFIG,
                     es=ES,
                     es_config=ES_CONFIG,
                     quiet=QUIET,
                     ):
    """ run buffer-size benchmark for different problems by calling `benchmark_es_parameter`
        but with `param_name="buffer_size"` and `param_val=buffer_size`

    :param buffer_size: int, the buffer size for the benchmark (try 1 - 10)

    The rest of the parameter is the same as in `benchmark_es_parameter`
    """
    kwargs = locals()
    kwargs["param_names"] = ["buffer_size"]
    kwargs["param_vals"] = [kwargs.pop("buffer_size")]
    return eval_es_parameter(**kwargs)


def eval_mutation_rate(mutation_rate=0.05,
                       readaptation=False,
                       elitism=False,
                       adaptive_selection_pressure=False,
                       foo="DoublePeak",
                       num_runs=NUM_RUNS,
                       generations=GENERATIONS,
                       nn=NN,
                       nn_config=NN_CONFIG,
                       diff=DIFF,
                       diff_config=DIFF_CONFIG,
                       es=ES,
                       es_config=ES_CONFIG,
                       quiet=QUIET,
                       dst=DST_param,
                       ):
    """ run mutation-rate benchmark for different problems by calling `benchmark_es_parameter`
        but with `param_name="mutation_rate"` and `param_val=mutation_rate`

    :param mutation_rate: float, the mutation rate for the benchmark (try 0.0 - 1.0)
    :param readaptation: bool, whether to use mutation adaptation (denoising steps) or not.
                                Only valid for `HADES` and `CHARLES` evolutionary strategies.

    The rest of the parameter is the same as in `benchmark_es_parameter` """
    kwargs = locals()
    kwargs["param_names"] = ["mutation_rate"]
    kwargs["param_vals"] = [kwargs.pop("mutation_rate")]
    return eval_es_parameter(**kwargs)


def eval_crossover_ratio(crossover_ratio=0.0,
                         elitism=False,
                         adaptive_selection_pressure=False,
                         foo="DoublePeak",
                         num_runs=NUM_RUNS,
                         generations=GENERATIONS,
                         nn=NN,
                         nn_config=NN_CONFIG,
                         diff=DIFF,
                         diff_config=DIFF_CONFIG,
                         es=ES,
                         es_config=ES_CONFIG,
                         quiet=QUIET,
                         dst=DST_param,
                         ):
    """ run crossover-rate benchmark for different problems by calling `benchmark_es_parameter`
        but with `param_name="crossover_rate"` and `param_val=crossover_rate`

    :param crossover_rate: float, the crossover rate [0., 1.] for the benchmark
    """
    kwargs = locals()
    kwargs["param_names"] = ["crossover_ratio"]
    kwargs["param_vals"] = [kwargs.pop("crossover_ratio")]
    return eval_es_parameter(**kwargs)


def eval_selection_pressure(selection_pressure=10.,
                            elitism=False,
                            foo="DoublePeak",
                            num_runs=NUM_RUNS,
                            generations=GENERATIONS,
                            nn=NN,
                            nn_config=NN_CONFIG,
                            diff=DIFF,
                            diff_config=DIFF_CONFIG,
                            es=ES,
                            es_config=ES_CONFIG,
                            quiet=QUIET,
                            dst=DST_param,
                            ):
        """ run selection-pressure benchmark for different problems by calling `benchmark_es_parameter`
            but with `param_name="selection_pressure"` and `param_val=selection_pressure`

        :param selection_pressure: float, the selection pressure for the benchmark (try values between 2.0 and 30.0)

        The rest of the parameter is the same as in `benchmark_es_parameter`
        """
        kwargs = locals()
        kwargs["param_names"] = ["selection_pressure"]
        kwargs["param_vals"] = [kwargs.pop("selection_pressure")]
        kwargs["adaptive_selection_pressure"] = False
        return eval_es_parameter(**kwargs)


def eval_elite_ratio(elite_ratio=0.15,
                     elitism=False,
                     adaptive_selection_pressure=False,
                     readaptation=False,
                     foo="DoublePeak",
                     num_runs=NUM_RUNS,
                     generations=GENERATIONS,
                     nn=NN,
                     nn_config=NN_CONFIG,
                     diff=DIFF,
                     diff_config=DIFF_CONFIG,
                     es=ES,
                     es_config=ES_CONFIG,
                     quiet=QUIET,
                     dst=DST_param,
                     ):
    """ run elite-ratio benchmark for different problems by calling `benchmark_es_parameter`
        but with `param_name="elite_ratio"` and `param_val=elite_ratio`

    :param elite_ratio: float, the elite ratio for the benchmark (try 0.0 - 0.5)
    """
    kwargs = locals()
    kwargs["param_names"] = ["elite_ratio"]
    kwargs["param_vals"] = [kwargs.pop("elite_ratio")]
    return eval_es_parameter(**kwargs)


if __name__ == "__main__":
    import argh
    argh.dispatch_commands([run,
                            objective_names,
                            objectives,
                            eval_es_parameter,
                            eval_buffer_size,
                            eval_mutation_rate,
                            eval_crossover_ratio,
                            eval_selection_pressure,
                            eval_elite_ratio
                            ])