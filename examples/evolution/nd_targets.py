import torch
import numpy as np
from condevo.es import HADES, CHARLES
from condevo.es.guidance import FitnessCondition, KNNNoveltyCondition
from condevo.nn import MLP
from condevo.diffusion import DDIM, RectFlow
from condevo.stats import diversity


def foo(x, targets, metric='euclidean', amplitude=5.):
    f = torch.zeros(x.shape[0], len(targets), device=x.device)
    for i, target in enumerate(targets):
        if metric == 'euclidean':
            f[:, i] = torch.linalg.norm(x - target, dim=-1)

        elif metric == 'exponential':
            f[:, i] = torch.exp(-torch.linalg.norm(x - target, dim=-1))

        else:
            raise NotImplementedError(f"metric {metric} not implemented")

    if metric == 'euclidean':
        return -f.min(dim=-1).values * amplitude

    elif metric == 'exponential':
        return f.max(dim=-1).values * amplitude

    else:
        raise NotImplementedError(f"metric {metric} not implemented")


def plot_3d(x, f):
    # Plotting: 3D scatter plot of the parameters (colored by fitness) across generations
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.animation import FuncAnimation

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


def hades(generations=20, popsize=512, autoscaling=True, sample_uniform=True):
    # define the fitness function
    targets = [[0.1, 4.0, -3.0],
               [-2., 0.5, -0.25],
               [1.0, -1., 1.4],
               ]
    targets = torch.tensor(targets)

    # define the neural network
    num_params = len(targets[0])
    mlp = MLP(num_params=num_params, num_hidden=48, num_layers=3, activation='ELU', batch_norm=True)
    mlp = mlp.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # define the diffusion model
    diffuser = DDIM(nn=mlp,
                    num_steps=100,
                    noise_level=1.0,
                    autoscaling=autoscaling,
                    sample_uniform=sample_uniform,
                    alpha_schedule="cosine",
                    matthew_factor=np.sqrt(0.5),
                    diff_range=20.0,
                    # predict_eps_t=True,
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
                   diff_max_epoch=300,
                   diff_batch_size=256,
                   diff_weight_decay=1e-6,
                   buffer_size=5,
                   diversity_selection=False,
                   )

    # evolutionary loop
    x, f = [], []
    for g in range(generations):
        x_g = solver.ask()          # sample new parameters
        f_g = foo(x_g, targets)     # evaluate fitness
        solver.tell(f_g)            # tell the solver the fitness of the parameters

        # logging
        print(f"Generation {g} -> fitness: {f_g.max()}, diversity: {diversity(x_g)}")
        x.append(x_g)
        f.append(f_g)

    # plotting results
    plot_3d(x, f)


if __name__ == "__main__":
    import argh
    argh.dispatch_commands([hades])

