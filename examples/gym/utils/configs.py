ES_LOGS = dict(
    log_fields = ("x", "reward", "duration", "done"),
    log_foos = {},
    checkpoint_interval = 1,
)

ES_COMMONS = dict(
    popsize=256,                       # population size
    sigma_init=1.0,                    # initial standard deviation
)


HADES = dict(
    **ES_COMMONS,
    is_genetic_algorithm=False,        # whether to use as genetic algorithm or as evolutionary algorithm (in the former case, solutions of the buffer are selected for training via roulette wheel selection, in the latter case, the solution are weighted by their fittnes for training the diffusion model)
    selection_pressure=8.0,            # selection pressure for roulette wheel selection
    adaptive_selection_pressure=True,  # whether to adapt the selection pressure
    elite_ratio=0.1,                   # ratio of the population to keep as elite. If crossover is used, an additional copy of the elite solutions might be added to the population if `improvement_steps` is defined.
    crossover_ratio=0.4,               # ratio of the population to perform crossover (if genetic algorithm, otherwise ignored)
    mutation_rate=0.2,                 # mutation rate after crossover and sampling
    unbiased_mutation_ratio=0.1,       # ratio of the population to perform unbiased mutation (if genetic algorithm, otherwise ignored)
    readaptation=False,                # whether to use mutation adaptation (denoising steps) or not. Only valid for `HADES` and `CHARLES` evolutionary strategies.
    forget_best=False,                 # whether to forget the best solution, or keep the elites for the next generation
    weight_decay=0,                    # L2 weight decay for the parameters of the evolutionary strategy (NOT of the diffusion model)
    diff_optim="Adam",                 # DM: optimizer for training the diffusion model
    diff_lr=3e-3,                      # DM: learning rate for training the diffusion model
    diff_weight_decay=1e-8,            # DM: weight decay for training the diffusion model
    diff_batch_size=256,               # DM: batch size for training the diffusion model
    diff_max_epoch=300,                # DM: number of training steps for the diffusion model at each generation
    diff_continuous_training=False,    # DM: continuous training, or retrain diffusion model at each generation
    buffer_size=10,                    # DM: buffer size for training the diffusion model, enabling to store (elites of) multiple generations. New solutions replace old weaker ones.
)
""" Default-Configuration for the HADES evolutionary strategy. """


CHARLES = {k: v for k, v in HADES.items()}
""" Default-Configuration for the CHARLES evolutionary strategy. """


CMAES = dict(
    **ES_COMMONS,
    weight_decay=0.0,                # weight decay for the parameters of the evolutionary strategy
    inopts=None,                     # additional options for the CMA-ES optimizer, see `cma.CMAOptions`
)


SimpleGA = dict(
    **ES_COMMONS,
    sigma_decay=0.999,               # decay of the standard deviation
    sigma_limit=0.01,                # limit of the standard deviation
    elite_ratio=0.1,                 # ratio of the population to keep as elite
    forget_best=False,               # whether to forget the best solution, or keep the elites for the next generation
    weight_decay=0.0,                # weight decay for the parameters of the evolutionary strategy
    reg='l2',                        # regularization for the weight decay
)


OpenES = dict(
    **ES_COMMONS,
    sigma_decay=0.999,
    sigma_limit=0.01,
    learning_rate=0.01,
    learning_rate_decay=0.9999,
    learning_rate_limit=0.001,
    antithetic=False,
    weight_decay=0.01,
    reg='l2',
    rank_fitness=True,
    forget_best=True,
)


PEPG = dict(
    **ES_COMMONS,
    sigma_alpha=0.20,
    sigma_decay=0.999,
    sigma_limit=0.01,
    sigma_max_change=0.2,
    learning_rate=0.01,
    learning_rate_decay=0.9999,
    learning_rate_limit=0.01,
    elite_ratio=0,
    average_baseline=True,
    weight_decay=0.01,
    reg='l2',
    rank_fitness=True,
    forget_best=True,
)


DDIM = dict(
    num_steps=1000,
)


VPred = {k: v for k, v in DDIM.items()}
XPred = {k: v for k, v in DDIM.items()}

RectFlow = dict(
    num_steps=100,
)


MLP = dict(
    num_hidden=24,
    num_layers=3,
    activation="LeakyReLU",
)

UNet = dict(
    num_hidden=[24, 12,],
    activation="LeakyReLU,"
)