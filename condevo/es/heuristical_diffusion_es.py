import torch
from torch import rand, randn, optim, tensor, zeros, Tensor
import numpy as np
from scipy.optimize import minimize
from ..diffusion import DM, DDIM, RectFlow, get_default_model
from ..es import utils
from .data import DataBuffer
from typing import Optional, Union


class HADES:
    """ HADES optimizer, which uses a diffusion model to either (i) sample high-fitness solutions or to (ii) learn
    a distribution of the fitness landscape.

    At each generation, HADES samples solutions from the diffusion model, evaluates them, and trains the diffusion model
    (i) on the best solution, (ii) weighted by the fitness values. The diffusion model is trained to predict samples
    that are similar to (i) the best solutions / (ii) distribution of solutions in the backward diffusion process.
    The diffusion model is trained with a Gaussian diffusion process in the forward direction, which is a stochastic
    process that starts with a high-fitness solution and diffuses it to a random noise.

    Note the analogy to the CMA-ES optimizer, which adapts the covariance matrix of a Gaussian distribution to sample
    high-fitness solutions. HADES uses a diffusion model to learn a more complex distribution of high-fitness solutions.
    """

    def __init__(self,
                 num_params,
                 model: Union[DM, str] = "DDIM",
                 popsize=256,
                 sigma_init=1.0,
                 x0=None,
                 is_genetic_algorithm=False,
                 selection_pressure=3.,
                 adaptive_selection_pressure=False,
                 elite_ratio=0.1,
                 forget_best=True,
                 crossover_ratio=0.0,
                 mutation_rate=0.0,
                 unbiased_mutation_ratio=0.0,
                 random_mutation_ratio=0.0625,
                 readaptation=False,
                 weight_decay=0.0,
                 reg='l2',
                 diff_optim: Union[torch.optim.Optimizer, str] = optim.Adam,
                 diff_lr: float = 1e-3,
                 diff_weight_decay: float = 1e-5,
                 diff_batch_size: int = 32,
                 diff_max_epoch: int = 100,
                 diff_continuous_training: bool = False,
                 diff_sample_kwargs: dict = None,
                 to_numpy=False,
                 eps: float = 1e-12,
                 buffer_size: Optional[Union[int, dict, DataBuffer]] = 4,
                 training_interval: int = 1,
                 ):
        """ Constructs a SHADES optimizer.

        :param num_params: int, number of parameters to es
        :param model: hades.diffusion.DM, model to use for diffusion, which learns to draw samples of high fitness values.
                      If None is provided, the `get_default_model` function provides a default model, if any of the names
                      ('DDIM', 'RectFlow') is used, a corresponding default-model is generated.
        :param popsize: int, population size
        :param sigma_init: float, initial standard deviation
        :param x0: array, initial solution
        :param is_genetic_algorithm: bool, whether to use the genetic algorithm variant of HADES, i.e., to use the
                                     evaluated fitness values as probabilities for the selection of the dataset to train the
                                     diffusion model.
                                     In other words: if used as genetic algorithm, solutions of the buffer are selected
                                     for training the DM via roulette wheel selection of the solution's fitness.
                                     In the case of the algorithm being used as evolutionary, the solution are weighted by
                                     their fitness when training the diffusion model.
        :param selection_pressure: float, selection pressure for the `roulette_wheel` fitness transformation
        :param adaptive_selection_pressure: bool, whether to adapt the selection pressure, so the elite solutions
                                            have a probability of (1-elite_ratio) to be selected. If False, the
                                            selection pressure is fixed, otherwise the selection pressure is the
                                            initial value.
        :param elite_ratio: float, ratio of the population considered as elite. The elites are used for crossover
                            operations, and for adaptive selection pressure. If also the `forget_best` parameter is
                            set to False, the elites are kept for the next generation. Otherwise, they are
                            subjected to mutation.
        :param crossover_ratio: The ratio of non-elite samples `crossover_ratio * popsize * (1 - elite_ratio)` that are
                                reproduced using genetic crossover, defaults to 0. The remaining samples are drawn from
                                the diffusion model.
        :param mutation_rate: float, mutation rate after crossover and sampling. The mutation rate specifies the ratio
                              of diffusion steps `model.num_steps * mutation_rate` that are applied to the samples.
        :param unbiased_mutation_ratio: float, ratio of the population that is mutated without applying the diffusion,
                                        while no mutations with diffusion are applied to any subset of the population.
                                        If this value is non-zero, a total of `popsize * unbiased_mutation_ratio`
                                        individuals are mutated, where for each individual, `num_parameters * mutation_rate`
                                        genes are mutated by Gaussian noise of scale `sigma_init`.
                                        This option is compatible with `readaptation`.
        :param random_mutation_ratio: float, ratio of the population that is mutated by adding random noise over
                                      the entire parameter range (this is not subject to any potential readaptation).
        :param readaptation: bool, whether to refine the mutated samples by applying denoising for
                             `model.num_steps * mutation_rate` steps (i.e., try to revert the mutations).
        :param forget_best: bool, whether to protect the elite solution from being replaced and mutated.
        :param weight_decay: float, weight decay coefficient for the population parameters in the fitness evaluation.
        :param reg: str, weight-decay regularization type, either 'l1' or 'l2'
        :param diff_optim: torch.optim.Optimizer or str, optimizer to use for training the diffusion model
        :param diff_lr: float, learning rate for training the diffusion model
        :param diff_weight_decay: float, weight decay coefficient for the diffusion model parameters
        :param diff_batch_size: int, batch size for training the diffusion model
        :param diff_max_epoch: int, maximum number of training epochs for the diffusion model
        :param diff_continuous_training: bool, whether to continue training the diffusion model based on the previous model parameters (of the previous generation)
        :param diff_sample_kwargs: dict, kwargs for sampling from the diffusion model. For RectFlow, this can include
                                   num_v: int, clip: float, normalize: bool.
        :param to_numpy: bool, whether to return the solutions as numpy arrays
        :param buffer_size: int, size of the buffer to store the solutions, fitness and probabilities for training the diffusion model.
        :param training_interval: int, number of sampled generations between retraining the diffusion model.
        """

        self.num_params = num_params
        self.popsize = popsize
        self.weight_decay = weight_decay
        self.reg = reg

        self.sigma_init = sigma_init
        if x0 is None:
            x0 = zeros(self.num_params)
        assert x0.shape == (self.num_params,), x0.shape
        self.x0 = x0
        self.selection_pressure = selection_pressure  # selection_pressure initial value
        self.adaptive_selection_pressure = adaptive_selection_pressure
        self.is_genetic_algorithm = is_genetic_algorithm

        # genetic algorithm parameters
        self.elite_ratio = elite_ratio  # ratio of the population considered as elite
        self.forget_best = forget_best  # whether to forget the best (elite) solutions

        self.crossover_ratio = crossover_ratio  # ratio of the population to perform crossover

        self.mutation_rate = mutation_rate  # mutation rate, intrinsically performed via DM
        self.unbiased_mutation_ratio = unbiased_mutation_ratio
        self.random_mutation_ratio = random_mutation_ratio
        self.readaptation = readaptation

        if model is None or isinstance(model, str):
            model = get_default_model(dm_cls=model, num_params=num_params)
        self.model = model
        self.diff_optim = diff_optim
        self.diff_lr = diff_lr
        self.diff_weight_decay = diff_weight_decay
        self.diff_batch_size = diff_batch_size
        self.diff_max_epoch = diff_max_epoch
        self.diff_continuous_training = diff_continuous_training
        self.eps = eps
        self.diff_sample_kwargs = diff_sample_kwargs or dict()

        self.to_numpy = to_numpy

        self.solutions = None
        self.fitness = None
        self._fitness_argsort = None  # helper
        self.regularization = None

        self.best_solution = None
        self.best_fitness = -torch.inf

        self._buffer = None
        self.buffer = buffer_size

        self.training_interval = training_interval
        self._asked = 0

    @property
    def buffer(self):
        """ Buffer for the solutions, fitness and conditions for training the diffusion model. """
        return self._buffer

    @buffer.setter
    def buffer(self, value: Union[int, dict, DataBuffer]):
        if isinstance(value, int):
            self._buffer = DataBuffer(max_size=value * self.popsize, num_conditions=self.model.num_conditions)

        elif isinstance(value, dict):
            kwargs = dict()
            if 'max_size' in value:
                kwargs['max_size'] = value.pop('max_size') * self.popsize

            if "num_conditions" not in value:
                value["num_conditions"] = self.model.num_conditions

            self._buffer = DataBuffer(**kwargs, **value)

        elif isinstance(value, DataBuffer):
            self._buffer = value

        else:
            raise ValueError("Buffer must be an int, dict or DataBuffer instance.")

    @property
    def num_elite(self):
        """ Number of elite solutions. """
        return int(self.popsize * self.elite_ratio)

    @property
    def elites(self):
        """ Elite solutions. """
        if not self.num_elite:
            return self.solutions
        return self.solutions[:self.num_elite]

    @property
    def elite_fitness(self):
        """ Fitness of elite solutions. """
        if not self.num_elite:
            return self.fitness
        return self.fitness[:self.num_elite]

    @property
    def is_evolutionary_strategy(self):
        return not self.is_genetic_algorithm

    @is_evolutionary_strategy.setter
    def is_evolutionary_strategy(self, value):
        self.is_genetic_algorithm = not value

    def flush(self, solutions):
        """ Override the current population with the given solutions. Flush the fitness. """
        if not isinstance(solutions, Tensor):
            solutions = tensor(solutions)
        self.solutions = solutions.type(torch.FloatTensor)
        self.fitness = None
        self.buffer = {}

    @property
    def is_initial_population(self):
        return self.solutions is None

    def ask(self):
        """ Sample solutions from the diffusion model.

        Initially, sample from a Gaussian distribution with mean x0 and standard deviation sigma_init.
        After the first generation, sample from the diffusion model.
        If a `mutation_rate` is specified, add random Gaussian noise to non-elite solutions at a rate of
        `r=popsize*mutation_rate`.

        :return: torch.Tensor of shape (popsize, num_params), sampled solutions
        """

        # update population with new samples
        if self.is_initial_population:
            # initial population
            samples = randn(self.popsize, self.num_params)
            self.solutions = samples * self.sigma_init + self.x0

        else:
            if self.num_elite < self.popsize:
                # keep the elite solutions
                self.solutions[self.num_elite:] = self.sample(self.popsize - self.num_elite)

            # mutate the solutions
            if self.forget_best:
                self.solutions[:] = self.mutate(self.solutions)

            else:
                # keep elites
                self.solutions[self.num_elite:] = self.mutate(self.solutions[self.num_elite:])

        if self.to_numpy:
            return utils.tensor_to_numpy(self.solutions)

        return self.solutions

    def tell(self, reward_table_result, parameters=None):
        """ Train the diffusion model on the best solution.

        :param reward_table_result: torch.Tensor of shape (popsize,), fitness values of the sampled solutions
        :param parameters: torch.Tensor of shape (popsize, num_params), optional parameters to update the solutions

        """
        # sort the solutions by fitness, starting with the best solution
        fitness, fitness_argsort = self.get_fitness_argsort(reward_table_result)
        self.fitness = fitness
        if parameters is not None:
            # update the solutions with the given parameters
            if self.solutions is None:
                self.solutions = parameters
            else:
                self.solutions[:] = parameters

        self._fitness_argsort = fitness_argsort
        self.solutions = self.solutions[fitness_argsort]
        self.selection_pressure = self.get_selection_pressure(fitness)

        # update the best solution
        if self.fitness[0] > self.best_fitness:
            self.best_fitness = self.fitness[0].clone()
            self.best_solution = self.solutions[0].clone()

        parent_dataset, survival_weights = self.selection()
        if not self._asked % self.training_interval:
            self.loss = self.train_model(dataset=parent_dataset, weights=survival_weights)

            # import matplotlib.pyplot as plt
            # plt.plot(self.loss)
            # plt.xlabel("Epochs")
            # plt.ylabel("Loss")
            # plt.show()

        self._asked += 1
        return self.loss

    def log(self):
        logger = self.model.logger
        if logger is None:
            return

        logger.log_scalar("evo/population/best", self.fitness.max(), logger.generation)
        logger.log_scalar("evo/population/mean", self.fitness.mean(), logger.generation)
        logger.log_scalar("evo/population/std", self.fitness.std(), logger.generation)

        logger.log_scalar("evo/buffer/best", self.buffer.fitness.max(), logger.generation)
        logger.log_scalar("evo/buffer/mean", self.buffer.fitness.mean(), logger.generation)
        logger.log_scalar("evo/buffer/std", self.buffer.fitness.std(), logger.generation)

    def sample(self, num_samples=None, *conditions):
        """ Sample solutions from the diffusion model and from crossover.

        A total number of `num_samples * self.crossover_ratio` solutions are sampled via crossover,
        the rest is sampled from the diffusion model.

        :param num_samples: int, number of samples to draw from the diffusion model, defaults to popsize if None.
        :param conditions: optional tuple, conditions for the diffusion model.
        :return: torch.Tensor of shape (num_samples, num_params), sampled solutions. In case of crossover,
                 the newly drawn samples and crossover_samples are concatenated (in that order).
        """
        # sample without eval probs via Jacobian estimation
        if num_samples is None:
            num_samples = self.popsize

        num_crossover = int(self.crossover_ratio * num_samples)
        num_new_samples = num_samples - num_crossover

        # sample `num_new_samples` from the diffusion model - unbiased, from Gaussian noise
        samples = []
        if num_new_samples:
            new_conditions = tuple(c[:num_new_samples] for c in conditions)
            samples = self.model.sample(num=num_new_samples, shape=(self.num_params,),
                                        conditions=new_conditions, **self.diff_sample_kwargs)

        if num_crossover:
            # get `num_crossover` samples via crossover operations
            x_conditions = tuple(c[num_new_samples:] for c in conditions)
            x_samples = self.get_crossover(num_crossover, *x_conditions)
            if len(samples):
                samples = torch.concatenate((samples, x_samples), dim=0)
            else:
                samples = x_samples

        return samples

    def get_crossover(self, num_crossover, *conditions):
        # roulette_wheel probability for selection
        p = utils.roulette_wheel(utils.tensor_to_numpy(self.elite_fitness), s=self.selection_pressure, normalize=True)

        # sample crossover solutions
        xt_crossover = []
        for i in range(num_crossover):
            # select two parents via roulette wheel selection, either from elites or from all if elite_ratio = 0
            j, k = np.random.choice(np.arange(0, self.num_elite or self.popsize), size=2, p=p, replace=False)
            p1 = self.elites[j]

            # eval distance of j to all solutions via torch euclidean distance
            distances = torch.cdist(p1.reshape(1, -1), self.elites).flatten()

            try:
                raise AssertionError("All to All mutations.")
                valid_distances = torch.zeros([])
                fitness_std = self.fitness.std()
                if isinstance(self.model, DDIM):
                    # use the bar_alpha value to determine the valid distances: only consider distances that are
                    # smaller than the bar_alpha value of the current step, i.e., reachable by the diffusion model
                    diff_speciation_steps = int(np.rint(self.mutation_rate * (self.model.num_steps - 1)))
                    valid_distances = distances <= (1.-self.model.alpha[diff_speciation_steps]).sqrt() * fitness_std

                elif isinstance(self.model, RectFlow):
                    # use the linear interpolation by the time parameter to determine the valid distances
                    valid_distances = distances <= self.mutation_rate * fitness_std

                if not torch.any(valid_distances):
                    raise AssertionError("No valid distances found.")

                assert torch.any(valid_distances)
                p_p2 = p * utils.tensor_to_numpy(valid_distances)
                p_p2 /= p_p2.sum()
                k = np.random.choice(np.arange(0, self.num_elite), p=p_p2)
                p2 = self.elites[k]

            except (AssertionError, AttributeError):
                p2 = self.elites[k]

            # recombine the two parents' genes via random crossover
            crossover_indices = np.random.rand(self.num_params) < 0.5
            xt_crossover.append(p1 * crossover_indices + p2 * (~crossover_indices))

        return torch.stack(xt_crossover, dim=0)

    def mutate(self, samples, *conditions):
        """ Mutate the non-elite solutions by adding random Gaussian noise via the diffusion-model forward process.
            If `readaptation` is True, refine the mutated solutions by applying denoising """

        if self.mutation_rate is None or self.is_initial_population:
            return samples

        t_diffuse = tensor(self.mutation_rate, device=samples.device)
        if not self.unbiased_mutation_ratio:
            # "Mutate" crossover samples by adding noise (applying diffusion for a few steps)
            # -> would be interesting to find number of steps based on points of
            #    spontaneous symmetry breaking in the diffusion process.  https://arxiv.org/abs/2305.19693
            samples, _ = self.model.diffuse(samples, t=t_diffuse)

        else:
            mutate_indices = torch.rand(samples.shape[0]) < self.unbiased_mutation_ratio
            mutated_samples = samples[mutate_indices]
            mutated_samples, _ = self.model.diffuse(mutated_samples, t=t_diffuse)

        if self.readaptation:
            # refine the diffused/mutated samples by applying denoising for a few steps
            adaptive_diff_steps = int(np.rint(t_diffuse * (self.model.num_steps - 1)))  # be careful about T max
            samples = self.model.sample(x_source=samples, shape=(self.num_params,),
                                        t_start=adaptive_diff_steps, conditions=conditions,
                                        **self.diff_sample_kwargs)

        if self.random_mutation_ratio:
            # add random Gaussian noise to the samples
            random_indices = torch.rand(samples.shape[0]) < self.random_mutation_ratio
            for i in torch.where(random_indices)[0]:
                std = torch.maximum(self.model.param_std.flatten(), tensor(self.sigma_init, dtype=samples.dtype, device=samples.device))
                random_noise = (rand(self.num_params) - 0.5) * std * 3.
                samples[i, :] += random_noise

            if self.model.diff_range is not None:
                # clip the samples to the parameter range
                samples[:] = samples.clamp(-self.model.diff_range, self.model.diff_range)

        return samples

    def get_fitness_argsort(self, reward_table):
        """ Return the fitness values of the sampled solutions. """
        if not isinstance(reward_table, Tensor):
            reward_table = tensor(reward_table)

        if self.weight_decay > 0:
            reg = utils.compute_weight_decay(self.weight_decay, self.solutions, reg=self.reg)
            reward_table += reg

        argsort_reward = reward_table.argsort(dim=0, descending=True)
        return reward_table[argsort_reward], argsort_reward

    def get_selection_pressure(self, fitness):
        """ Return the selection pressure for the roulette wheel selection, so the current populations
            elites make up (1-elite_ratio) . """

        if not self.adaptive_selection_pressure or not self.elite_ratio:
            return self.selection_pressure

        num_elites = int(self.popsize * self.elite_ratio)
        def foo(s):
            p = utils.roulette_wheel(f=utils.tensor_to_numpy(fitness), s=s, normalize=True)
            elite_weight = (1 - self.elite_ratio)
            return np.abs(np.cumsum(sorted(p, reverse=True))[num_elites-1] - elite_weight)

        result = minimize(foo, x0=self.selection_pressure, bounds=[(0.1, 30.)], tol=1e-3)
        selection_pressure = result.x.item()
        return selection_pressure

    def selection(self):
        """ Perform roulette_wheel selection step of current population data. """
        x = self.solutions
        fitness = self.fitness
        if x is not None:
            self.buffer.push(x, fitness)

        # get buffer dataset
        x_dataset = self.buffer.x.clone()

        # evaluate roulette wheel selection for buffer samples
        f_dataset = self.buffer.fitness.flatten()

        # check for nans (e.g. runaway parameters)
        if torch.isinf(f_dataset).any() or torch.isnan(f_dataset).any():
            infty = torch.isinf(f_dataset) | torch.isnan(f_dataset)
            f_dataset = f_dataset[~infty]
            x_dataset = x_dataset[~infty]

        weights_dataset = utils.roulette_wheel(f=f_dataset, s=self.selection_pressure, normalize=False)
        weights_dataset = weights_dataset.reshape(-1, 1)
        self.buffer.info['selection_probability'] = weights_dataset.clone().flatten()

        if self.is_genetic_algorithm:
            # select samples from buffer based on roulette wheel selection
            selected_genotypes = torch.multinomial(weights_dataset.flatten() / weights_dataset.sum(), len(x_dataset), replacement=True)
            x_dataset = x_dataset[selected_genotypes]
            weights_dataset = None  # disable weights for DM training

        # else:
        #     # renormalize weights to max. value of 1.
        #     weights_dataset = weights_dataset / weights_dataset.max()

        return x_dataset, weights_dataset

    def train_model(self, dataset, weights=None):
        """ Train the diffusion model on the given dataset.

        :param weights: torch.Tensor of shape (num_elite,), weights for each sample in the dataset
        """
        if not self.diff_continuous_training:
            # reinitialize the diffusion model
            self.model.init_nn()

        loss = self.model.fit(dataset, weights=weights, **self.diff_kwargs)
        return loss

    @property
    def diff_kwargs(self):
        """ Diffusion model kwargs. """
        return dict(optimizer=self.diff_optim,
                    max_epoch=self.diff_max_epoch,
                    lr=self.diff_lr,
                    weight_decay=self.diff_weight_decay,
                    batch_size=self.diff_batch_size)

    def optimize(self,
                 objective_function: callable,
                 num_steps: int = 20,
                 track_history: bool = True,
                 temperature_range: tuple = (1.0, 0.5),
                 ):

        """ A wrapper to optimize the objective function. """

        history = {"samples": [], "fitness": []}
        temperatures = np.linspace(*temperature_range, num_steps)
        for temperature in temperatures:
            solutions = self.ask()
            fitness = objective_function(solutions)
            self.fitness_temperature = temperature
            self.tell(fitness)
            if track_history:
                history["samples"].append(self.solutions.clone())
                history["fitness"].append(self.fitness.clone())

        if track_history:
            return history

        return self.solutions[0].clone(), self.fitness[0].clone()

    def result(self):
        """ return best params so far, along with historically best reward, curr reward, curr reward STD. """
        best_solution = utils.tensor_to_numpy(self.best_solution)
        best_fitness = utils.tensor_to_numpy(self.best_fitness)
        fitness = utils.tensor_to_numpy(self.fitness[0])
        sigma = utils.tensor_to_numpy(self.fitness).std()
        return best_solution, best_fitness, fitness, sigma
