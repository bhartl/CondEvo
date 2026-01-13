HADES = dict(
    popsize=256,                       # population size
    sigma_init=2,                      # initial standard deviation, also used for mutation
    x0=None,                           # initial guess of the solution
    is_genetic_algorithm=True,         # whether to use as genetic algorithm or as evolutionary algorithm (in the former case, solutions of the buffer are selected for training via roulette wheel selection, in the latter case, the solution are weighted by their fittnes for training the diffusion model)
    selection_pressure=5.0,            # selection pressure for roulette wheel selection
    adaptive_selection_pressure=True,  # whether to adapt the selection pressure
    elite_ratio=0.15,                  # ratio of the population to keep as elite. If crossover is used, an additional copy of the elite solutions might be added to the population if `improvement_steps` is defined.
    crossover_ratio=0.125,             # ratio of the population to perform crossover (if genetic algorithm, otherwise ignored)
    mutation_rate=0.05,                # mutation rate after crossover and sampling
    readaptation=False,                # whether to use mutation adaptation (denoising steps) or not. Only valid for `HADES` and `CHARLES` evolutionary strategies.
    forget_best=True,                  # whether to forget the best solution, or keep the elites for the next generation
    weight_decay=0,                    # L2 weight decay for the parameters of the evolutionary strategy (NOT of the diffusion model)
    diff_optim="Adam",                 # DM: optimizer for training the diffusion model
    diff_lr=1e-2,                      # DM: learning rate for training the diffusion model
    diff_weight_decay=1e-5,            # DM: weight decay for training the diffusion model
    diff_batch_size=256,               # DM: batch size for training the diffusion model
    diff_max_epoch=200,                # DM: number of training steps for the diffusion model at each generation
    diff_continuous_training=False,    # DM: continuous training, or retrain diffusion model at each generation
    buffer_size=3,                     # DM: buffer size for training the diffusion model, enabling to store (elites of) multiple generations. New solutions replace old weaker ones.
)
""" Default-Configuration for the HADES evolutionary strategy. """

CHARLES = {k: v for k, v in HADES.items()}
""" Default-Configuration for the CHARLES evolutionary strategy. """

CMAES = dict(
    sigma_init=1.0,                  # initial standard deviation
    popsize=256,                     # population size
    weight_decay=0.0,                # weight decay for the parameters of the evolutionary strategy
    x0=None,                         # initial guess of the solution
    inopts=None,                     # additional options for the CMA-ES optimizer, see `cma.CMAOptions`
)

MultistartCMA = {k: v for k, v in CMAES.items()}
""" Default-Configuration for the Multistart-CMA evolutionary strategy. """
MultistartCMA["num_starts"] = 5      # number of restarts
MultistartCMA["cls"] = "CMAES"

SimpleGA = dict(
    sigma_init=1.0,                  # initial standard deviation
    sigma_decay=0.999,               # decay of the standard deviation
    sigma_limit=0.01,                # limit of the standard deviation
    popsize=256,                     # population size
    elite_ratio=0.1,                 # ratio of the population to keep as elite
    forget_best=False,               # whether to forget the best solution, or keep the elites for the next generation
    weight_decay=0.0,                # weight decay for the parameters of the evolutionary strategy
    reg='l2',                        # regularization for the weight decay
    x0=None,                         # initial guess of the solution
)

OpenES = dict(
    sigma_init=1.0,
    sigma_decay=0.999,
    sigma_limit=0.01,
    learning_rate=0.01,
    learning_rate_decay=0.9999,
    learning_rate_limit=0.001,
    popsize=256,
    antithetic=False,
    weight_decay=0.01,
    reg='l2',
    rank_fitness=True,
    forget_best=True,
    x0=None,
)

PEPG = dict(
    sigma_init=1.0,
    sigma_alpha=0.20,
    sigma_decay=0.999,
    sigma_limit=0.01,
    sigma_max_change=0.2,
    learning_rate=0.01,
    learning_rate_decay=0.9999,
    learning_rate_limit=0.01,
    elite_ratio=0,
    popsize=256,
    average_baseline=True,
    weight_decay=0.01,
    reg='l2',
    rank_fitness=True,
    forget_best=True,
    x0=None,
)

DDIM = dict(
    num_steps=1000,
)

RectFlow = dict(
    num_steps=100,
)

MLP = dict(
    num_hidden=24,
    num_layers=3,
    activation="LeakyReLU",
)

SelfAttention = dict(
    num_hidden=8,
    num_heads=2,
    num_mlp=1,
    num_layers=3,
    num_channels=1,
    activation="GELU",
    positional_embedding_size=3,
)
