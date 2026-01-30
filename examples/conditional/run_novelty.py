import numpy as np
from condevo.es.guidance import Condition, KNNNoveltyCondition, FitnessCondition
from benchmark import utils, configs, run
from foobench import Objective


DST = utils.DST + '/paper_HADES/experiment_novelty/'


def get_objective(objective):
    if objective == "double_dip":
        return Objective(foo="double_dip", maximize=True, foo_kwargs=dict(m=1.0))

    if objective == "himmelblau":
        return Objective(foo="himmelblau", maximize=True, limits=5, apply_limits=True, limit_val=-200)

    if objective == "rastrigin":
        return Objective(foo="rastrigin", maximize=False, limits=4, apply_limits=True, limit_val=0)

    if objective == "helix_rastrigin":
        return Objective(foo="helix_rastrigin", maximize=False, limits=None, apply_limits=False, foo_kwargs={"rmax": 4})
    
    if objective == "spiral_peaks":
        foo_kwargs = foo_kwargs={"npeaks": 3, "nbranches": 3, "distance": 1.0, "sigma": 0.5, "omega": 5.0/2./np.pi}
        return Objective(foo="spiral_peaks", maximize=False, limits=None, apply_limits=False, foo_kwargs=foo_kwargs)

    raise NotImplementedError(objective)


def run_experiments(conditions="", dst=DST, niter=2, objective="double_dip", adaptive_selection_pressure=False, generations=101, es="CHARLES", param_range=10, popsize=256, sigma_init=0.2):
    es_conditions = []
    conditions = conditions.split("-")
    for c, dash in zip(conditions, (["-"] * len(conditions))[:-1] + [""]):
        try:
            c = int(c)
        except:
            pass

        if isinstance(c, int):
            es_conditions.append(KNNNoveltyCondition(k=c, weight_by_fitness=True))
            dst += f"novelty_{c}"

        elif c == "fisher":
            es_conditions.append(FitnessCondition(greedy=False))
            dst += f"charles_{c}"

        elif c == "":
            dst += "baseline"

        elif c == "greedy": 
            es_conditions.append(FitnessCondition(greedy=True))
            dst += f"charles_{c}"

        else:
            raise ValueError(c)


        dst += dash


    kwargs = dict(
        objective=get_objective(objective),
        generations=generations,
        es=es,
    )

    if es == "CHARLES":
        es_config = {**configs.CHARLES, "conditions": es_conditions}
        es_config["selection_pressure"] = 12.
        es_config["adaptive_selection_pressure"] = adaptive_selection_pressure 
        es_config["elite_ratio"] = 0.25
        es_config["crossover_ratio"] = 0.
        es_config["mutation_rate"] = 0.2
        es_config["unbiased_mutation_ratio"] = 0.1
        es_config["readaptation"] = False
        es_config["forget_best"] = True
        es_config["diff_lr"] = 1e-2
        es_config["diff_max_epoch"] = 200
        es_config["diff_weight_decay"] = 1e-8
        es_config["buffer_size"] = 10
        es_config["sigma_init"] = sigma_init
        es_config["is_genetic_algorithm"] = False  # True

    
        kwargs["diff"] = "DDIM"
        kwargs["nn"] = "MLP"
        kwargs["nn_config"] = {
                "num_conditions": len(es_conditions), 
                "num_hidden": 64,
                "num_layers": 2,
                "activation": "ReLU",
                # "time_embedding": 32,
                # "batch_norm": False,
                # "layer_norm": True,
        }
        kwargs["es_config"] = es_config
        kwargs["diff_config"] = {"num_steps": 1000, } # "param_range": param_range, "lambda_range": 1e-4}
        # kwargs["diff_config"]["scaler"] = None  # "StandardScaler"
        # kwargs["diff_config"]["diff_range"] = param_range  # True

    else:
        es_config = {**getattr(configs, es)}
        if es == "CMAES":
            #es_config["inopts"] = {"AdaptSigma": False}
            es_config["inopts"] = {"CMA_elitist": 0}
            
        elif es == "SimpleGA":
            #es_config["sigma_decay"] = 1.0
            es_config["forget_best"] = True

        elif es == "MultistartCMA":
            kwargs["es"] = "MultistartES"

        es_config["sigma_init"] = sigma_init
        kwargs["diff"] = None
        kwargs["nn"] = None
        kwargs["es_config"] = es_config

    kwargs["es_config"]["popsize"] = popsize

    from datetime import datetime
    timestamp = datetime.utcnow().strftime('%Y%m%d%H%M%S%f')[:-3]

    return [run(dst=dst, timestamp=str(timestamp), **kwargs) for _ in range(niter)]


if __name__ == "__main__":
    import argh
    argh.dispatch_command(run_experiments)


