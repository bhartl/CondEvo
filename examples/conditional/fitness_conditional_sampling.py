import os
import torch
import numpy as np
from torch.distributions import MultivariateNormal
from condevo.nn import MLP
from condevo.diffusion import DDIM, RectFlow
import matplotlib.pyplot as plt


def two_peak_density(x, mu1=None, mu2=None, std=0.25):
    if mu1 is None:
        mu1 = torch.tensor([-1., -1.])
    if mu2 is None:
        mu2 = torch.tensor([1., 1.])

    # Checking if the input tensor x has shape (2,) and unsqueeze to make it (*N, 2)
    if len(x.shape) == 1:
        x = x.unsqueeze(0)

    # Covariance matrix for the Gaussian distributions (identity matrix, since it's a standard Gaussian)
    covariance_matrix = torch.eye(2) * std

    # Create two multivariate normal distributions
    dist1 = MultivariateNormal(mu1, covariance_matrix)
    dist2 = MultivariateNormal(mu2, covariance_matrix)

    # Evaluate the density functions for each distribution and sum them up
    density = dist1.log_prob(x).exp() + dist2.log_prob(x).exp()

    return density


def discretize(f, discrete, fmax=None):
    if discrete > 0:
        max_f = torch.max(f) if fmax is None else fmax
        f = torch.round((f / max_f) * discrete) / discrete * max_f

    return f


def train_model(model="DDIM", epochs=500, lr=1e-3, discrete=-1):
    std = 0.5
    samples = [std * torch.randn(5000, 2) + torch.Tensor([[-1, -1]]), std * torch.randn(5000, 2) + torch.Tensor([[1, 1]])]
    samples = torch.cat(samples, dim=0)

    max_value = two_peak_density(torch.Tensor([[-1, -1]])).item()
    fitness = two_peak_density(samples, std=std).reshape(-1, 1) / max_value
    fitness = discretize(fitness, discrete, fmax=max_value)

    mlp = MLP(num_params=2, num_hidden=64, num_layers=2, num_conditions=1, activation="ReLU")
    if model == "RectFlow":
        model = RectFlow(nn=mlp, num_steps=100, )

    elif model == "DDIM":
        model = DDIM(nn=mlp, num_steps=1000, noise_level=0.1)

    else:
        raise ValueError(f"Model {model} not recognized.")

    loss_count = model.fit( samples, fitness, max_epoch=epochs, lr=lr, weight_decay=1e-8, batch_size=64)
    return model, loss_count, samples, max_value


def main(model:str = "DDIM", epochs:int = 500, lr: float = 1e-3, discrete:int = -1):
    """ Trains the model conditional on fitness values of Gaussian two-peak function.
        Plots the loss and samples in `./img/` directory.

    :param model: str, Model to be used for training. Defaults to "DDIM".
    :param epochs: int, Maximum number of epochs for training. Defaults to 500.
    :param lr: float, Learning rate for the optimizer. Defaults to 1e-3.
    :param discrete: int, Number of discrete values for the fitness. Defaults to -1 (continuous).
    """
    
    # Create the model
    dm, loss_count, samples, max_value = train_model(model=model, epochs=epochs, lr=lr, discrete=discrete)

    # plot the loss
    img_dir = "local/fitness_conditional_sampling/img"
    plt.plot(loss_count)
    plt.title("Diffusion Model Training")

    discrete_label = f"-discrete_{discrete}" if discrete > 0 else ""
    os.makedirs(img_dir, exist_ok=True)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(f"{img_dir}/{model}-loss{discrete_label}.png")
    plt.close()

    # sample
    fig = plt.figure(figsize=(5, 5))
    plt.plot(samples[:, 0], samples[:, 1], '.', alpha=0.1, color='gray')
    ax = plt.gca()
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # def plot gradient on uniform, equally spaced grid
    res_x, res_y = 41, 41
    x = np.linspace(-3, 3, res_x)
    y = np.linspace(-3, 3, res_y)
    X, Y = np.meshgrid(x, y)
    Z = two_peak_density(torch.Tensor(np.dstack((X, Y)).reshape(-1, 2))).reshape(res_x, res_y) / max_value
    Z = discretize(Z, discrete, fmax=1.)
    if discrete > 0:
        print("discrete fitness values:", Z.unique())
        target_fitness = sorted(Z.unique())
    else:
        target_fitness = [0.0, 0.25, 0.5, 0.75, 1.0]

    XY = torch.Tensor(np.dstack((X, Y)).reshape(-1, 2))

    fig_grad, ax_gradient = plt.subplots(1, len(target_fitness), figsize=(5 * len(target_fitness), 5), sharex=True, sharey=True)
    sampled_fitness = []
    for f, ax_grad in zip(target_fitness, ax_gradient):
        conditions = (torch.Tensor([[f]]*512),)
        sampled = dm.sample((2,), 512, conditions=conditions)
        fitness = two_peak_density(sampled) / max_value
        sampled_fitness.append(fitness)
        ax.plot(sampled[:, 0], sampled[:, 1], '.', alpha=1, label=f'fitness={f:.2f}')

        conditions = (torch.Tensor([[f]] * len(XY)),)
        t_start = 100 if model == "DDIM" else 90
        refined = dm.sample(shape=(2,),  x_source=XY, conditions=conditions, t_start=t_start)

        gradient = (refined - XY)
        # plot the gradient field
        ax_grad.quiver(X, Y, gradient[:, 0].reshape(res_x, res_y), gradient[:, 1].reshape(res_x, res_y), zorder=1)
        ax_grad.set_title(f'fitness={f:.2f}')

        magnitude = torch.norm(gradient, dim=-1).reshape(res_x, res_y)
        ax_grad.imshow(magnitude, extent=(-3, 3, -3, 3), origin='lower', cmap='magma_r', zorder=0, alpha=0.4)
        ax_grad.set_xlim(-3, 3)
        ax_grad.set_ylim(-3, 3)
        ax_grad.set_aspect('equal', adjustable='box')
        ax_grad.set_xlabel("x")

    ax_gradient[0].set_ylabel("y")
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal', adjustable='box')
    ax.legend()

    fig.savefig(f"{img_dir}/{model}-sample{discrete_label}.png")
    fig_grad.savefig(f"{img_dir}/{model}-gradient{discrete_label}.pdf")
    plt.close()


if __name__ == "__main__":
    import argh
    argh.dispatch_command(main)
