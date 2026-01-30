"""
nd_parameter_search.py

Experimental suite for conditional parameter-search using diffusion-based generative samplers combined with evolutionary strategies.

Purpose
- Provide experiments that combine learned conditional samplers (DDIM, RectFlow, VPred, ...) with evolutionary search algorithms (HADES, CHARLES) to explore multimodal fitness landscapes and quality-diversity strategies.
- Include simple multimodal fitness benchmark (`foo`), plotting helpers (`plot_2d`, `plot_3d`), a small QD buffer implementation, and several experiment entry points: `hades`, `charles`, `hades_GA_refined`, and `hades_score_refined`.
- Provide two custom conditioning strategies: `UniformFitnessCondition` and `RouletteFitnessCondition`.

Usage
- Run experiments from the command line via the provided argh dispatch: `python -m nd_targets hades --generations 100`
- Adjust hyperparameters such as `popsize`, `sigma_init`, `elite_ratio`, `matthew_factor`, `diff_range`, `scaler`, and buffer strategies to study trade-offs between exploration and exploitation.

Dependencies
- torch, numpy, matplotlib
- package-local modules: `condevo.es`, `condevo.diffusion`, `condevo.nn`, `condevo.es.data`, `condevo.stats`

Notes
- Designed for small-dimensional parameter spaces (2D/3D visualizations).
- Modify the neural backbone and diffuser settings to adapt to larger/complex domains.
"""


import torch
import numpy as np
from condevo.es import HADES, CHARLES
from condevo.es.guidance import FitnessCondition, KNNNoveltyCondition
from condevo.es.data import DataBuffer
from condevo.nn import MLP
from condevo.diffusion import DDIM, RectFlow, VPred
from condevo.diffusion.x_prediction import XPred
from condevo.stats import diversity


def foo(x, targets, metric='euclidean', amplitude=5.):
    f = torch.zeros(x.shape[0], len(targets), device=x.device)
    for i, target in enumerate(targets):
        if metric == 'euclidean':
            f[:, i] = torch.linalg.norm(x - target, dim=-1)

        elif metric == 'exponential':
            f[:, i] = torch.exp(-torch.linalg.norm(x - target, dim=-1) / amplitude)

        else:
            raise NotImplementedError(f"metric {metric} not implemented")

    if metric == 'euclidean':
        return -f.min(dim=-1).values * amplitude

    elif metric == 'exponential':
        return f.max(dim=-1).values

    else:
        raise NotImplementedError(f"metric {metric} not implemented")


def plot_3d(x, f, targets=None):
    # Plotting: 3D scatter plot of the parameters (colored by fitness) across generations
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.animation import FuncAnimation

    if targets is not None:
        dists = []
        for i, target in enumerate(targets):
            dists.append(torch.stack([torch.linalg.norm(xi - target, dim=-1).min() for xi in x]))

        for i, d in enumerate(dists):
            plt.plot(d, label=f"target {i}", marker=".", linewidth=0)

        plt.xlabel("Generation")
        plt.ylabel("Distance to target")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Initialize the scatter plot
    scatter = ax.scatter([], [], [], c='b', marker='o', s=10)

    # Set axis limits
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_zlim(-10, 10)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Update function for animation
    def update(frame):
        # set the data for the scatter plot
        x_t, y_t, z_t = x[frame].T
        scatter._offsets3d = (x_t.numpy(), y_t.numpy(), z_t.numpy())

        # color by fitness
        f_frame = f[frame]
        scatter.set_array(f_frame.numpy())

        ax.set_title(f"Frame {frame}")
        return scatter,

    # Create the animation
    ani = FuncAnimation(fig, update, frames=len(x), interval=100, blit=False)

    # Show the animation
    plt.show()


def plot_2d(x, f, targets=None):
    # Plotting: 3D scatter plot of the parameters (colored by fitness) across generations
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    if targets is not None:
        dists = []
        for i, target in enumerate(targets):
            dists.append(torch.stack([torch.linalg.norm(xi - target, dim=-1).min() for xi in x]))

        for i, d in enumerate(dists):
            plt.plot(d, label=f"target {i}", marker=".", linewidth=0)

        plt.xlabel("Generation")
        plt.ylabel("Distance to target")

    fig = plt.figure()
    ax = plt.gca()
    # Initialize the scatter plot
    scatter = ax.scatter([], [], c='b', marker='o')

    # Set axis limits
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    # Update function for animation
    def update(frame):
        # set the data for the scatter plot
        x_t, y_t = x[frame].T
        scatter.set_offsets(np.c_[x_t.numpy(), y_t.numpy()])

        # color by fitness
        f_frame = f[frame]
        scatter.set_array(f_frame.numpy())

        ax.set_title(f"Frame {frame}")
        return scatter,

    # Create the animation
    ani = FuncAnimation(fig, update, frames=len(x), interval=100, blit=False)

    # Show the animation
    plt.show()


def get_diffuser(diffuser, mlp, scaler, tensorboard, no_skip_connection):
    if diffuser == "DDIM":
        # define the diffusion model
        diffuser = DDIM(nn=mlp,
                        num_steps=100,
                        noise_level=1.,
                        scaler=scaler,
                        alpha_schedule="cosine_nichol",
                        matthew_factor=np.sqrt(0.5),
                        diff_range=10.0,
                        log_dir="data/logs/hades_DDIM" * tensorboard,
                        skip_connection=not no_skip_connection,
                        )

    elif diffuser == "RectFlow":
        # define the fect flow
        diffuser = RectFlow(nn=mlp,
                            num_steps=100,
                            noise_level=0.5,
                            scaler=scaler,
                            matthew_factor=1,#np.sqrt(0.5),
                            diff_range=10.0,
                            log_dir="data/logs/hades_RF" * tensorboard,
                            )

    elif diffuser == "VPred":
        diffuser = VPred(nn=mlp,
                         num_steps=100,
                         noise_level=1.,
                         scaler=scaler,
                         alpha_schedule="cosine_nichol",
                         matthew_factor=1,
                         diff_range=10.0,
                         log_dir="data/logs/hades_VPred" * tensorboard,
                         skip_connection=not no_skip_connection,
                         )

    elif diffuser == "XPred":
        diffuser = XPred(nn=mlp,
                         num_steps=100,
                         noise_level=1.,
                         scaler=scaler,
                         alpha_schedule="cosine_nichol",
                         matthew_factor=1,
                         diff_range=10.0,
                         log_dir="data/logs/hades_XPred" * tensorboard,
                        )

    else:
        raise NotImplementedError(diffuser)

    return diffuser


def hades(generations=100, popsize=512, scaler="StandardScaler",
          is_genetic=False, diffuser="DDIM", tensorboard=False, sharpen_sampling=25,
          reload=False, no_skip_connection=False,
          ):
    # define the fitness function
    targets = [[0.1, 4.0, -3.0],
               [-2., 0.5, -0.25],
               [1.0, -1., 1.4],
               ]

    # targets = [[0.1, 4.0, ],
    #            [-2., 0.5, ],
    #            [1.0, -1., ],
    #            ]

    # targets = [
    #     [10.332, 20.044, 10.399, ],
    #     [-10.418, 10.795, -20.232, ],
    #     [0.897, -10.847, -20.126,],
    # ]
    targets = torch.tensor(targets)

    # define the neural network
    num_params = len(targets[0])
    mlp = MLP(num_params=num_params, num_hidden=32, num_layers=6, activation='SiLU',
              layer_norm=True, time_embedding=32)

    # from condevo.nn.self_attention import SelfAttentionMLP
    # mlp = SelfAttentionMLP(num_params=num_params, num_hidden=32, num_layers=6, activation='ReLU',
    #                         batch_norm=True, dropout=0.1, num_heads=4, num_conditions=0)

    # mlp = mlp.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    diffuser = get_diffuser(diffuser, mlp, scaler, tensorboard, no_skip_connection)

    # define the evolutionary strategy
    solver = HADES(num_params=num_params,
                   model=diffuser,
                   popsize=popsize,
                   sigma_init=3.0,
                   is_genetic_algorithm=is_genetic,
                   selection_pressure=10,
                   adaptive_selection_pressure=False,            # chose seletion_pressure such that `elite_ratio` individuals have cumulative probability
                   ###########################################
                   ## free sampling
                   # elite_ratio=0.,                 # 0.1 if not is_genetic else 0.4,
                   # mutation_rate=0.,               # 0.05
                   # unbiased_mutation_ratio=0.,     # 0.1
                   # readaptation=False,
                   ###########################################
                   ###########################################
                   ## protect locals
                   elite_ratio=0.,#2 if not is_genetic else 0.4,
                   mutation_rate=0.05,
                   unbiased_mutation_ratio=0.01,
                   readaptation=True,
                   ###########################################
                   random_mutation_ratio=0.125,
                   crossover_ratio=0.0,
                   forget_best=True,
                   diff_lr=0.003,
                   diff_optim="Adam",
                   diff_max_epoch=10,
                   diff_batch_size=256,
                   diff_weight_decay=1e-6,
                   buffer_size=10,  # don't restrict buffer size
                   diff_continuous_training=reload,
                   model_path="data/models/hades.pt" * reload,
                   training_interval=10,
                   )

    # evolutionary loop
    x, f = [], []
    for g in range(generations):
        x_g = solver.ask()          # sample new parameters
        f_g = foo(x_g, targets)     # evaluate fitness
        print(f"Generation {g} -> fitness: {f_g.max()}, diversity: {diversity(x_g)}")
        solver.tell(f_g)            # tell the solver the fitness of the parameters
        x.append(x_g)
        f.append(f_g)

        if g > sharpen_sampling:
            solver.model.matthew_factor = 1.0  # first go for diversity, then sharpen sampling after `N` generations

    if num_params == 3:
        # plotting results
        plot_3d(x, f, targets=targets)

    else:
        # plotting results
        plot_2d(x, f, targets=targets)


def charles(generations=15, popsize=512, scaler=None,
            is_genetic=False, diffuser="DDIM", tensorboard=False, sharpen_sampling=10,
            disable_knn_condition=False, disable_fitness_condition=False,
            diff_continuous_training=False, no_skip_connection=False,
            ):
    # define the fitness function
    targets = [[0.1, 4.0, -3.0],
               [-2., 0.5, -0.25],
               [1.0, -1., 1.4],
               ]

    # targets = [[0.1, 4.0, ],
    #            [-2., 0.5, ],
    #            [1.0, -1., ],
    #            ]

    # targets = [
    #     [10.332, 20.044, 10.399, ],
    #     [-10.418, 10.795, -20.232, ],
    #     [0.897, -10.847, -20.126,],
    # ]
    targets = torch.tensor(targets)

    conditions = []
    # define the KNN novelty condition
    if not disable_knn_condition:
        print("Using KNN novelty condition")
        knn_condition = KNNNoveltyCondition(k=popsize//len(targets), metric=2, beta=10., weight_by_fitness=True, eps=1e-8)
        conditions.append(knn_condition)

    if not disable_fitness_condition:
        print("Using fitness condition")
        fitness_condition = FitnessCondition(scale=1.0, greedy=False)
        conditions.append(fitness_condition)

    # define the neural network
    num_params = len(targets[0])
    mlp = MLP(num_params=num_params, num_hidden=32, num_layers=3, activation='SiLU',
              batch_norm=True, # dropout=0.1,
              num_conditions=len(conditions))
    # mlp = mlp.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    diffuser = get_diffuser(diffuser, mlp, scaler, tensorboard, no_skip_connection)

    # define the evolutionary strategy
    solver = CHARLES(num_params=num_params,
                     model=diffuser,
                     popsize=popsize,
                     sigma_init=2.0,
                     is_genetic_algorithm=is_genetic,
                     selection_pressure=10,
                     adaptive_selection_pressure=False,            # chose seletion_pressure such that `elite_ratio` individuals have cumulative probability
                     elite_ratio=0.2 if not is_genetic else 0.4,
                     mutation_rate=0.05,
                     unbiased_mutation_ratio=0.1,
                     crossover_ratio=0.0,
                     readaptation=False,
                     forget_best=True,
                     diff_lr=0.003,
                     diff_optim="Adam",
                     diff_max_epoch=100,
                     diff_batch_size=256,
                     diff_weight_decay=1e-6,
                     buffer_size=10,  # don't restrict buffer size
                     diff_continuous_training=diff_continuous_training,
                     conditions=conditions,
                     training_interval=10,
                    )

    # evolutionary loop
    x, f = [], []
    for g in range(generations):
        x_g = solver.ask()          # sample new parameters
        f_g = foo(x_g, targets)     # evaluate fitness
        print(f"Generation {g} -> fitness: {f_g.max()}, diversity: {diversity(x_g)}")
        solver.tell(f_g)            # tell the solver the fitness of the parameters
        x.append(x_g)
        f.append(f_g)

        if g > sharpen_sampling:
            solver.model.matthew_factor = 1.0  # first go for diversity, then sharpen sampling after `N` generations

    if num_params == 3:
        # plotting results
        plot_3d(x, f, targets=targets)

    else:
        # plotting results
        plot_2d(x, f, targets=targets)


def hades_GA_refined(generations=20, popsize=512, scaler="StandardScaler", tensorboard=False):
    """ use HADES alternatingly with and without genetic algorithm training, i.e.,
        at every generation, first
        train the diffusion model weighted by fitness (increases diversity)
        and train the diffusion model on the best individuals (increases convergence).
        After the weight-based training, data are included into the DataBuffer following the "quality" criterion,
        and after the genetic algorithm training, data are included following the "diversity" criterion.
        """
    # define the fitness function
    targets = [[0.1, 4.0, -3.0],
               [-2., 0.5, -0.25],
               [1.0, -1., 1.4],
               ]
    targets = torch.tensor(targets)

    # define the neural network
    num_params = len(targets[0])
    mlp = MLP(num_params=num_params, num_hidden=32, num_layers=6, activation='SiLU')
    mlp = mlp.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # define the diffusion model
    diffuser = DDIM(nn=mlp,
                    num_steps=100,
                    noise_level=1.0,
                    scaler=scaler,
                    alpha_schedule="cosine_nichol",
                    matthew_factor=np.sqrt(0.5),
                    diff_range=10.0,
                    log_dir="data/logs/refined/hades" * tensorboard,
                    )

    # define the evolutionary strategy
    solver = HADES(num_params=num_params,
                   model=diffuser,
                   popsize=popsize,
                   sigma_init=2.0,
                   is_genetic_algorithm=False,
                   selection_pressure=10,
                   elite_ratio=0.9,
                   mutation_rate=0.05,
                   unbiased_mutation_ratio=0.1,
                   crossover_ratio=0.0,
                   readaptation=True,
                   forget_best=True,
                   diff_lr=0.001,
                   diff_optim="Adam",
                   diff_max_epoch=100,
                   diff_batch_size=256,
                   diff_weight_decay=1e-6,
                   buffer_size=5,
                   )

    # evolutionary loop
    x, f = [], []
    for g in range(generations):
        x_g = solver.ask()          # sample new parameters
        f_g = foo(x_g, targets)     # evaluate fitness

        x.append(x_g)
        f.append(f_g)

        # train as a genetic algorithm, i.e., select training dataset from the best individuals
        solver.buffer.pop_type = DataBuffer.POP_QUALITY
        solver.is_genetic_algorithm = True   # train as a genetic algorithm
        r, solver.elite_ratio = solver.elite_ratio, 0.4  # select 40% of the best individuals
        if tensorboard:
            solver.model.logger.log_dir = "data/logs/refined/genetic"
        solver.tell(f_g)                     # tell the solver the fitness of the parameters, and train DM

        x_g = solver.ask()                   # sample new parameters from GA-HADES
        f_g = foo(x_g, targets)              # evaluate fitness
        solver.is_genetic_algorithm = False  # restore to HADES (train DM on fitness-weighted parameters)
        solver.buffer.pop_type = DataBuffer.POP_DIVERSITY    # use diversity criteria for dataset selection
        solver.elite_ratio = r               # restore to original elite ratio
        if tensorboard:
            solver.model.logger.log_dir = "data/logs/refined/hades"
        solver.tell(f_g)                     # tell the solver the fitness of the parameters

        # logging
        print(f"Generation {g} -> fitness: {f_g.max()}, diversity: {diversity(x_g)}")

    # plotting results
    plot_3d(x, f, targets=targets)


def hades_score_refined(generations=20, popsize=512, scaler="StandardScaler", t_score=0.05, n_refine=100, refine_interval=1, tensorboard=False):
    # define the fitness function
    targets = [[0.1, 4.0, -3.0],
               [-2., 0.5, -0.25],
               [1.0, -1., 1.4],
               ]
    targets = torch.tensor(targets)

    # define the neural network
    num_params = len(targets[0])
    mlp = MLP(num_params=num_params, num_hidden=64, num_layers=3, activation='SiLU', batch_norm=True)
    mlp = mlp.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # define the diffusion model
    diffuser = DDIM(nn=mlp,
                    num_steps=100,
                    noise_level=1.0,
                    scaler=scaler,
                    alpha_schedule="cosine_nichol",
                    matthew_factor=1., #0.8,  # np.sqrt(0.5),
                    diff_range=10.0,
                    log_dir="data/logs/score" * tensorboard,
                    )

    # define the evolutionary strategy
    solver = HADES(num_params=num_params,
                   model=diffuser,
                   popsize=popsize,
                   sigma_init=2.0,
                   is_genetic_algorithm=False,
                   selection_pressure=10,
                   elite_ratio=0.1,
                   mutation_rate=0.05,
                   unbiased_mutation_ratio=0.1,
                   crossover_ratio=0.0,
                   readaptation=True,
                   forget_best=True,
                   diff_lr=0.001,
                   diff_optim="Adam",
                   diff_max_epoch=100,
                   diff_batch_size=256,
                   diff_weight_decay=1e-6,
                   buffer_size=5,
                   )

    # evolutionary loop
    x, f = [], []
    for g in range(generations):
        x_g = solver.ask()          # sample new parameters
        f_g = foo(x_g, targets)     # evaluate fitness

        x.append(x_g)
        f.append(f_g)

        print(f"Generation {g} -> fitness: {f_g.max()}, diversity: {diversity(x_g)}")
        solver.diff_continuous_training = False
        solver.buffer.pop_type = DataBuffer.POP_QUALITY
        solver.tell(f_g, x_g)

        if not (g + 1) % refine_interval:
            model = solver.model
            x_g_refined = x_g.clone()
            f_g_refined = f_g.clone()
            sigma, model.sigma = model.sigma, torch.zeros_like(model.sigma)
            for t in np.linspace(t_score * model.num_steps, 1, n_refine):
                x_g_refined = model.sample(x_source=x_g_refined, shape=x_g_refined.shape[1:],
                                           t_start=int(t), **solver.diff_sample_kwargs)
                _f_g = foo(x_g_refined, targets)  # evaluate fitness
                improved = _f_g > f_g
                x_g_refined[improved] = x_g_refined[improved]
                f_g_refined[improved] = _f_g[improved]

            model.sigma = sigma
            solver.diff_continuous_training = True
            solver.buffer.pop_type = DataBuffer.POP_DIVERSITY
            print(f"               -> annealed: {f_g_refined.max()}, diversity: {diversity(x_g_refined)}")
            solver.tell(f_g_refined, x_g_refined)

    # plotting results
    plot_3d(x, f, targets=targets)


if __name__ == "__main__":
    import argh
    argh.dispatch_commands([hades,
                            charles,
                            hades_GA_refined,
                            hades_score_refined,
                            ])
