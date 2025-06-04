import torch
import numpy as np
from condevo.es import HADES, CHARLES
from condevo.es.guidance import FitnessCondition, KNNNoveltyCondition
from condevo.es.data import DataBuffer
from condevo.nn import MLP
from condevo.diffusion import DDIM, RectFlow
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
    scatter = ax.scatter([], [], [], c='b', marker='o')

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


def hades(generations=15, popsize=512, autoscaling=True, sample_uniform=True,
          is_genetic=False, diffuser="DDIM", tensorboard=False,
          score_refinement=10, refinement_generation_threshold=10,
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
              batch_norm=True, dropout=0.1)
    # mlp = mlp.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    if diffuser == "DDIM":
        # define the diffusion model
        diffuser = DDIM(nn=mlp,
                        num_steps=100,
                        noise_level=1.0,
                        autoscaling=autoscaling,
                        sample_uniform=sample_uniform,
                        alpha_schedule="cosine",
                        matthew_factor=np.sqrt(0.5),
                        diff_range=10.0,
                        # predict_eps_t=True,
                        log_dir="data/logs/hades" * tensorboard,
                        normalize_steps=False,
                        )

    else:
        # define the fect flow
        diffuser = RectFlow(nn=mlp,
                            num_steps=300,
                            noise_level=1.0,
                            autoscaling=autoscaling,
                            sample_uniform=sample_uniform,
                            matthew_factor=np.sqrt(0.5),
                            diff_range=10.0,
                            log_dir="data/logs/hades_RF" * tensorboard,
                            )

    # define the evolutionary strategy
    solver = HADES(num_params=num_params,
                   model=diffuser,
                   popsize=popsize,
                   sigma_init=2.0,
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
                   elite_ratio=0.1 if not is_genetic else 0.4,
                   mutation_rate=0.05,
                   unbiased_mutation_ratio=0.1,
                   readaptation=True,
                   ###########################################
                   random_mutation_ratio=0.0625,
                   crossover_ratio=0.0,
                   forget_best=True,
                   diff_lr=0.003,
                   diff_optim="Adam",
                   diff_max_epoch=100,
                   diff_batch_size=256,
                   diff_weight_decay=1e-6,
                   buffer_size=0,
                   diff_continuous_training=False,
                   )

    # evolutionary loop
    x, f = [], []
    for g in range(generations):
        if g > refinement_generation_threshold and score_refinement:
            solver.unbiased_mutation_ratio = 1e-6
            solver.readaptation = score_refinement

        x_g = solver.ask()          # sample new parameters
        f_g = foo(x_g, targets)     # evaluate fitness
        print(f"Generation {g} -> fitness: {f_g.max()}, diversity: {diversity(x_g)}")
        solver.tell(f_g)            # tell the solver the fitness of the parameters
        x.append(x_g)
        f.append(f_g)

    if num_params == 3:
        # plotting results
        plot_3d(x, f, targets=targets)

    else:
        # plotting results
        plot_2d(x, f, targets=targets)


def charles(generations=15, popsize=512, autoscaling=True, sample_uniform=True,
            is_genetic=False, diffuser="DDIM", tensorboard=False,
            score_refinement=0, refinement_generation_threshold=10,
            disable_knn_condition=False, disable_fitness_condition=False,
            diff_continuous_training=False,
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
        knn_condition = KNNNoveltyCondition(k=popsize//len(targets), metric=2, beta=10., weight_by_fitness=True, eps=1e-8)
        conditions.append(knn_condition)

    if not disable_fitness_condition:
        fitness_condition = FitnessCondition(scale=1.0, greedy=False)
        conditions.append(fitness_condition)

    # define the neural network
    num_params = len(targets[0])
    mlp = MLP(num_params=num_params, num_hidden=32, num_layers=6, activation='SiLU',
              batch_norm=True, dropout=0.1,
              num_conditions=len(conditions))
    # mlp = mlp.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    if diffuser == "DDIM":
        # define the diffusion model
        diffuser = DDIM(nn=mlp,
                        num_steps=100,
                        noise_level=1.0,
                        autoscaling=autoscaling,
                        sample_uniform=sample_uniform,
                        alpha_schedule="linear",
                        matthew_factor=np.sqrt(0.5),
                        diff_range=10.0,
                        # predict_eps_t=True,
                        log_dir="data/logs/charles" * tensorboard,
                        normalize_steps=True,
                        )

    else:
        # define the fect flow
        diffuser = RectFlow(nn=mlp,
                            num_steps=300,
                            noise_level=0.2,
                            autoscaling=autoscaling,
                            sample_uniform=sample_uniform,
                            matthew_factor=np.sqrt(0.5),
                            diff_range=10.0,
                            )

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
                   buffer_size=0,
                   diff_continuous_training=diff_continuous_training,
                   conditions=conditions
                   )

    # evolutionary loop
    x, f = [], []
    for g in range(generations):
        if g > refinement_generation_threshold and score_refinement:
            solver.unbiased_mutation_ratio = 1e-6
            solver.readaptation = score_refinement

        x_g = solver.ask()          # sample new parameters
        f_g = foo(x_g, targets)     # evaluate fitness
        print(f"Generation {g} -> fitness: {f_g.max()}, diversity: {diversity(x_g)}")
        solver.tell(f_g)            # tell the solver the fitness of the parameters
        x.append(x_g)
        f.append(f_g)

    if num_params == 3:
        # plotting results
        plot_3d(x, f, targets=targets)

    else:
        # plotting results
        plot_2d(x, f, targets=targets)


def hades_GA_refined(generations=20, popsize=512, autoscaling=True, sample_uniform=True, tensorboard=False):
    # define the fitness function
    targets = [[0.1, 4.0, -3.0],
               [-2., 0.5, -0.25],
               [1.0, -1., 1.4],
               ]
    targets = torch.tensor(targets)

    # define the neural network
    num_params = len(targets[0])
    mlp = MLP(num_params=num_params, num_hidden=32, num_layers=6, activation='SiLU', batch_norm=True)
    mlp = mlp.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # define the diffusion model
    diffuser = DDIM(nn=mlp,
                    num_steps=100,
                    noise_level=1.0,
                    autoscaling=autoscaling,
                    sample_uniform=sample_uniform,
                    alpha_schedule="cosine",
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


def hades_score_refined(generations=20, popsize=512, autoscaling=True, sample_uniform=True, t_score=0.05, n_refine=100, refine_interval=1, tensorboard=False):
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
                    num_steps=1000,
                    noise_level=0.1,
                    autoscaling=autoscaling,
                    sample_uniform=sample_uniform,
                    alpha_schedule="cosine",
                    matthew_factor=0.8,  # np.sqrt(0.5),
                    diff_range=10.0,
                    # predict_eps_t=True,
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



from condevo.es.guidance import FitnessCondition

class UniformFitnessCondition(FitnessCondition):
    def __init__(self, scale: float = 1.0, greedy=False):
        """ Constructor of the UniformFitnessCondition. """
        super().__init__(scale=scale, greedy=greedy)
        self.f = None

    def evaluate(self, charles_instance, x: torch.Tensor, f: torch.Tensor):
        self.f = f.clone()
        c = FitnessCondition.evaluate(self, charles_instance, x, f)
        c = c.to(f.device)
        return c

    def sample(self, charles_instance, num_samples: int):
        min_f, max_f = self.f.min(), self.f.max()
        f_sample = min_f + (max_f - min_f) * torch.rand(num_samples, device=self.f.device) * 1.1
        return f_sample / self.scale


class RouletteFitnessCondition(FitnessCondition):
    def __init__(self, scale: float = 1.0, greedy=False):
        """ Constructor of the UniformFitnessCondition. """
        super().__init__(scale=scale, greedy=greedy)
        self.f = None
        self.prob = None

    def evaluate(self, charles_instance, x: torch.Tensor, f: torch.Tensor):
        self.f = f.clone()
        c = FitnessCondition.evaluate(self, charles_instance, x, f)
        c = c.to(f.device)
        return c

    def sample(self, charles_instance, num_samples: int):
        # choice of self.f with probability self.prob
        selected_fitness = torch.multinomial(self.prob.flatten() / self.prob.sum(), len(self.f), replacement=True)
        selected_fitness = selected_fitness.to(self.f.device)

        min_f, max_f = self.f.min(), self.f.max()
        f_sample = min_f + (max_f - min_f) * torch.rand(num_samples, device=self.f.device)
        f_sample = torch.sort(f_sample, descending=False).values  # sort the samples in descending order

        f_sample = f_sample[selected_fitness]
        return f_sample / self.scale


def half_hades(generations=20, popsize=512, autoscaling=False, sample_uniform=False, tensorboard=False):
    # define the fitness function
    targets = [[0.1, 4.0, -3.0],
               [-2., 0.5, -0.25],
               [1.0, -1., 1.4],
               ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    targets = torch.tensor(targets, device=device)

    def init_diffuser():
        # define the neural network
        num_params = len(targets[0])
        mlp = MLP(num_params=num_params, num_hidden=64, num_layers=6, activation='SiLU', batch_norm=True, num_conditions=1)
        mlp = mlp.to(device=device)

        # define the diffusion model
        diffuser = DDIM(nn=mlp,
                        num_steps=1000,
                        noise_level=0.1,
                        autoscaling=autoscaling,
                        sample_uniform=sample_uniform,
                        alpha_schedule="cosine",
                        matthew_factor=0.8,  # np.sqrt(0.5),
                        diff_range=5.0,
                        log_dir="data/logs/half_hades" * tensorboard,
                        clip_gradients=1.
                        )

        diffuser = diffuser.to(device=device)
        return diffuser


    # evolutionary loop
    x, f = [], []
    samples = torch.randn(popsize, len(targets[0]), device=device) * 5.0  # initial random samples
    from condevo.es.utils import roulette_wheel

    # fitness_condition = UniformFitnessCondition(scale=1.0, greedy=False)
    fitness_condition = RouletteFitnessCondition()

    for g in range(generations):
        fx = foo(samples.clone(), targets)  # evaluate fitness
        print(f"Generation {g} -> fitness: {fx.clone().detach().max()}, diversity: {diversity(torch.tensor(samples.clone().detach()))}")

        argsort_f = fx.clone().detach().argsort()
        samples = samples[argsort_f]  # sort samples by fitness
        fx = fx[argsort_f]  # sort fitness by fitness

        x.append(samples.clone().detach().cpu())
        f.append(fx.clone().detach().cpu())

        fx_plain = fx.cpu().clone().detach()
        fx = roulette_wheel(f=fx.view(-1,1), s=10, assume_sorted=True, threshold=0.5) # higher S mean more greedy selection

        fitness_condition_evaluation = fitness_condition.evaluate(None, samples, fx_plain)
        fitness_condition_evaluation = fitness_condition_evaluation.view(-1, 1).to(device)
        fitness_condition.prob = fx.clone()

        fitness_samples = fitness_condition.sample(None, num_samples=popsize)
        fitness_samples = fitness_samples.to(device).view(-1, 1)  # sample fitness values for the diffuser

        import matplotlib.pyplot as plt
        plt.plot(fx_plain.numpy(), fx.cpu().numpy(), label=f"Generation {g}", marker=".", linewidth=0)

        # histogram of fitness
        plt.hist(fx_plain.cpu().numpy(), bins=50, alpha=0.5, label=f"Generation {g} histogram")
        plt.hist(fitness_samples.cpu().numpy(), bins=50, alpha=0.5, label=f"Sampling {g} histogram")
        plt.legend()
        plt.show()


        diffuser = init_diffuser()
        loss = diffuser.fit(samples, fitness_condition_evaluation, weights=fx.view(-1,1),
                            max_epoch=500, batch_size=256, weight_decay=1e-5,
                            )  # train the diffuser on the sampled parameters and their fitness
        print(f"Loss: {np.mean(loss):.4f}")

        samples = diffuser.sample(num=popsize, shape=(len(targets[0],),), conditions=[fitness_samples,])  # sample from the diffuser

    # plotting results
    plot_3d(x, f, targets=targets.clone().detach().cpu())


def almost_hades(generations=20, popsize=512, autoscaling=True, sample_uniform=True, tensorboard=False):
    # define the fitness function
    targets = [[0.1, 4.0, -3.0],
               [-2., 0.5, -0.25],
               [1.0, -1., 1.4],
               ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    targets = torch.tensor(targets, device=device)

    def init_diffuser():
        from condevo.es.data import DataBuffer

        # data buffer, potentially synchronized across multiple processes
        data_buffer = DataBuffer(max_size=0)

        # define the neural network
        num_params = len(targets[0])
        mlp = MLP(num_params=num_params, num_hidden=64, num_layers=3, activation='SiLU', batch_norm=False, )
        mlp = mlp.to(device=device)

        # define the diffusion model
        diffuser = DDIM(nn=mlp,
                        num_steps=1000,
                        noise_level=0.1,
                        autoscaling=autoscaling,
                        sample_uniform=sample_uniform,
                        alpha_schedule="cosine",
                        matthew_factor=0.8,  # np.sqrt(0.5),
                        diff_range=5.0,
                        log_dir="data/logs/half_hades" * tensorboard,
                        device=device,
                        )

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
                       buffer_size=data_buffer,
                       device=device,
                       )

        return solver, data_buffer


    # evolutionary loop
    x, f = [], []
    samples = torch.randn(popsize, len(targets[0]), device=device) * 5.0  # initial random samples
    from condevo.es.utils import roulette_wheel

    for g in range(generations):
        fx = foo(samples, targets)  # evaluate fitness
        print(f"Generation {g} -> fitness: {fx.clone().detach().max()}, diversity: {diversity(torch.tensor(samples.clone().detach()))}")
        x.append(samples.clone().detach().cpu())
        f.append(fx.clone().detach().cpu())

        # fx = roulette_wheel(f=fx.view(-1,1), s=10, device=device) # higher S mean more gready selection

        # # normalize the samples
        # fx  = (fx - fx.min()) / (fx.max() - fx.min() + 1e-8)

        solver, data_buffer = init_diffuser()
        data_buffer.push(x=samples, fitness=fx)
        parent_dataset, survival_weights = solver.selection()
        loss = solver.train_model(dataset=parent_dataset, weights=survival_weights)

        print(f"Loss: {np.mean(loss):.4f}")

        # if np.random.rand(1) < solver.elite_ratio:
        # draw high-quality from data buffer
        parent_dataset, fitness_probability = solver.selection()
        # could also directly access `data_buffer.x` and `data_buffer.fitness` for high quality/diversity sample

        # pick from parent_dataset with probability weights
        selected_genotypes = torch.multinomial(fitness_probability.flatten() / fitness_probability.sum(), samples.shape[0], replacement=True)
        x_g = parent_dataset[selected_genotypes]

        # refine the sample by denoising with retrained DM
        x_g = solver.model.sample(x_source=x_g, shape=(samples.shape[1],),)# t_start=50) # t_start = criticality mutation, how much we should refind on the criticality
        # else:
        # # sample from diffusion model
        # x_g = solver.model.sample(num=popsize, shape=(samples.shape[1],))  # t_start = criticality mutation, how much we should refind on the criticality

        samples = x_g


    # plotting results
    plot_3d(x, f, targets=targets.clone().detach().cpu())



if __name__ == "__main__":
    import argh
    argh.dispatch_commands([hades,
                            charles,
                            hades_GA_refined,
                            hades_score_refined,
                            half_hades,
                            almost_hades,
                            ])
