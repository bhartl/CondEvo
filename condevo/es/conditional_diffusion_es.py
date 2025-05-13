import torch
from torch import randn, optim
import numpy as np
from typing import Union, Optional, Tuple
from ..diffusion import DM, get_default_model
from ..es import utils
from ..es import HADES
from ..es.guidance import Condition, FitnessCondition


class CHARLES(HADES):
    """ Conditional CHARLES optimizer for targeted sampling with the CHARLES solver."""
    def __init__(self,
                 num_params: int,
                 conditions: Tuple[Condition] = (FitnessCondition(),),
                 model: Union[DM, str] = "DDIM",
                 popsize: int = 256,
                 sigma_init: float = 1.0,
                 x0: Optional[np.ndarray] = None,
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
                 weight_decay: float = 0.0,
                 reg: str = 'l2',
                 diff_optim: Union[torch.optim.Optimizer, str] = optim.Adam,
                 diff_lr: float = 1e-3,
                 diff_weight_decay: float = 1e-5,
                 diff_batch_size: int = 32,
                 diff_max_epoch: int = 100,
                 diff_continuous_training: bool = False,
                 diff_sample_kwargs: dict = None,
                 to_numpy: bool = False,
                 buffer_size: int = 4,
                 training_interval: int = 1,
                 diversity_selection: bool = False,
                 ):
        """ Constructs a CHARLES-Diffusion optimizer.

        :param num_params: int, number of parameters to es
        :param conditions: tuple or list of (str, scale) `Condition` instances.
        :param model: hades.diffusion.ConditionDDIM, model to use for conditional diffusion, which learns to draw
                      samples of high fitness values, conditioned on the highest fitness values and on the fitness
                      distribution of the current population.
        :param popsize: int, population size
        :param sigma_init: float, initial standard deviation
        :param x0: array, initial solution
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
                                        individuals are mutated, where for each individual, `model.num_steps * mutation_rate`
                                        genes are mutated by Gaussian noise of scale `sigma_init`.
                                        This option is compatible with `readaptation`.
        :param random_mutation_ratio: float, ratio of the population that is mutated by adding random noise over
                                      the entire parameter range (this is not subject to any potential readaptation).
        :param readaptation: bool, whether to refine the mutated samples by applying denoising for
                                    `model.num_steps * mutation_rate` steps (i.e., try to revert the mutations).
        :param forget_best: bool, whether to protect the elite solution from being replaced and mutated.
        :param weight_decay: float, weight decay coefficient for the population parameters in the fitness evaluation
        :param reg: str, weight-decay regularization type, either 'l1' or 'l2'
        :param diff_optim: torch.optim.Optimizer or str, optimizer to use for training the diffusion model
        :param diff_lr: float, learning rate for training the diffusion model
        :param diff_weight_decay: float, weight decay coefficient for the diffusion model parameters
        :param diff_batch_size: int, batch size for training the diffusion model
        :param diff_max_epoch: int, maximum number of training epochs for the diffusion model
        :param diff_continuous_training: bool, whether to continue training the diffusion model based on the previous
                                         model parameters (of the previous generation)
        :param diff_sample_kwargs: dict, kwargs for sampling from the diffusion model. For RectFlow, this can include
                                   num_v: int, clip: float, normalize: bool.
        :param to_numpy: bool, whether to return the solutions as numpy arrays
        :param buffer_size: int, size of the buffer to store the solutions and conditions for training the diffusion
                            model.
        :param training_interval: int, number of generations between training the diffusion model
        :param diversity_selection: bool, whether to use diversity selection for the buffer. If True, the buffer is
                                    updated with the best samples from the population by replacing the closest samples
                                    in the buffer with the new samples. If False, the buffer is updated with the best
                                    samples from the population by replacing the worst samples in the buffer with the
                                    new samples.
        """

        if model is None or isinstance(model, str):
            model = get_default_model(dm_cls=model, num_params=num_params, num_conditions=len(conditions))

        super().__init__(num_params=num_params,
                         model=model,
                         popsize=popsize,
                         sigma_init=sigma_init,
                         x0=x0,
                         is_genetic_algorithm=is_genetic_algorithm,
                         selection_pressure=selection_pressure,
                         adaptive_selection_pressure=adaptive_selection_pressure,
                         elite_ratio=elite_ratio,
                         crossover_ratio=crossover_ratio,
                         mutation_rate=mutation_rate,
                         unbiased_mutation_ratio=unbiased_mutation_ratio,
                         random_mutation_ratio=random_mutation_ratio,
                         readaptation=readaptation,
                         forget_best=forget_best,
                         weight_decay=weight_decay,
                         reg=reg,
                         diff_optim=diff_optim,
                         diff_lr=diff_lr,
                         diff_weight_decay=diff_weight_decay,
                         diff_batch_size=diff_batch_size,
                         diff_max_epoch=diff_max_epoch,
                         diff_continuous_training=diff_continuous_training,
                         diff_sample_kwargs=diff_sample_kwargs,
                         to_numpy=to_numpy,
                         buffer_size=buffer_size,
                         training_interval=training_interval,
                         diversity_selection=diversity_selection,
                         )

        self.conditions = conditions
        self.condition_values = None

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
            conditions = self.sample_conditions(self.popsize)
            non_elite_conditions = tuple([c[self.num_elite:] for c in conditions])

            if self.num_elite < self.popsize:
                # keep the elite solutions
                samples = self.sample(self.popsize - self.num_elite, *non_elite_conditions)
                self.solutions[self.num_elite:] = samples

            # mutate the solutions
            if self.forget_best:
                self.solutions[:] = self.mutate(self.solutions, *conditions)

            else:
                # keep elites
                samples = self.mutate(self.solutions[self.num_elite:], *non_elite_conditions)
                self.solutions[self.num_elite:] = samples

        if self.to_numpy:
            return utils.tensor_to_numpy(self.solutions)

        return self.solutions

    def selection(self):
        x = self.solutions
        fitness = self.fitness
        conditions = self.evaluate_conditions(x, fitness)
        self.buffer.push(x, fitness, *conditions)

        # get buffer dataset
        x_dataset = self.buffer.x
        conditions = self.buffer.conditions

        # evaluate roulette wheel selection for buffer samples
        f_dataset = self.buffer['fitness'].flatten()

        # check for nans (e.g. runaway parameters)
        if torch.isinf(f_dataset).any() or torch.isnan(f_dataset).any():
            infty = torch.isinf(f_dataset) | torch.isnan(f_dataset)
            f_dataset = f_dataset[~infty]
            x_dataset = x_dataset[~infty]
            conditions = tuple(c[~infty] for c in conditions)

        weights_dataset = utils.roulette_wheel(f=f_dataset, s=self.selection_pressure, normalize=False)
        weights_dataset = weights_dataset.reshape(-1, 1)
        self.buffer.info['selection_probability'] = weights_dataset.clone().flatten()

        # evaluate potential regularization term to reinforce conditions by discounting deviations from the conditions
        if self.is_genetic_algorithm:
            # select samples from buffer based on roulette wheel selection
            selected_genotypes = torch.multinomial(weights_dataset.flatten(), len(x), replacement=True)
            x_dataset = x_dataset[selected_genotypes]
            conditions = tuple(c[selected_genotypes] for c in conditions)
            weights_dataset = None  # disable weights for DM training

        return (x_dataset, conditions), weights_dataset

    def update_buffer(self, x, fitness):
        conditions = self.evaluate_conditions(x, fitness)

        if torch.isnan(fitness).any() or torch.isinf(fitness).any():
            fitness[torch.isnan(fitness)] = -np.infty

        if 'x' not in self.buffer:
            self.buffer['x'] = x.clone()
            self.buffer['fitness'] = fitness.clone()
            self.buffer['conditions'] = tuple(c.clone() for c in conditions)

        else:
            if self.buffer['x'].shape[0] >= self.buffer_size * self.popsize:
                # remove nan values from buffer
                is_nan = torch.isnan(self.buffer['fitness'])
                num_nan = torch.sum(is_nan)
                if num_nan:
                    self.buffer['x'] = self.buffer['x'][~is_nan]
                    self.buffer['fitness'] = self.buffer['fitness'][~is_nan]
                    self.buffer['conditions'] = tuple(c[~is_nan] for c in self.buffer['conditions'])

                if not self.diversity_selection:
                    # remove old samples from buffer with lowest fitness
                    num_replace = self.popsize - num_nan
                    indices = self.buffer['fitness'].flatten().argsort()
                    self.buffer['x'] = self.buffer['x'][indices[num_replace:]]
                    self.buffer['fitness'] = self.buffer['fitness'][indices[num_replace:]]
                    self.buffer['conditions'] = tuple(c[indices[num_replace:]] for c in self.buffer['conditions'])

                else:
                    # replace novel samples by maximizing the diversity of the buffer
                    indices = []
                    for xi, fi, *ci in zip(x, fitness, *conditions):
                        # find the most similar sample in the buffer
                        distances = torch.cdist(xi.reshape(1, -1), self.buffer['x']).flatten()
                        # get the index of the most similar sample
                        for index in distances.argsort()[:3]:
                            # replace the most similar sample with the new sample, if it is better
                            if self.buffer['fitness'][index] < fi and index not in indices:
                                # replace the sample in the buffer
                                self.buffer['x'][index] = xi
                                self.buffer['fitness'][index] = fi
                                for j in range(len(self.buffer['conditions'])):
                                    # replace the condition in the buffer
                                    self.buffer['conditions'][j][index] = ci[j]

                                indices += [index]
                                break

                    self.buffer['replaced'] = torch.tensor(indices)

            self.buffer['x'] = torch.cat((self.buffer['x'], x), dim=0)
            self.buffer['fitness'] = torch.cat((self.buffer['fitness'], self.fitness), dim=0)
            self.buffer['conditions'] = tuple(torch.cat((self.buffer['conditions'][i], c), dim=0) for i, c in enumerate(conditions))

        self.log()

    def sample_conditions(self, num_samples):
        samples = [c.sample(charles_instance=self, num_samples=num_samples) for c in self.conditions]
        for i in range(len(samples)):
            if samples[i].dim() == 1:
                samples[i] = samples[i].reshape(-1, 1)
        return tuple(samples)

    def evaluate_conditions(self, x, f):
        conditions = [c.evaluate(charles_instance=self, x=x, f=f) for c in self.conditions]
        for i in range(len(conditions)):
            if conditions[i].dim() == 1:
                conditions[i] = conditions[i].reshape(-1, 1)

        self.condition_values = tuple(conditions)
        return self.condition_values

    @property
    def elite_conditions(self):
        """ Fitness of elite solutions. """
        return self.condition_values[:self.num_elite] if self.num_elite else self.condition_values

    def train_model(self, dataset, weights=None):
        """ Train the diffusion model on the given dataset.

        :param dataset: tuple of torch.Tensor comprising the dataset and the conditions for training the diffusion model
        :param weights: torch.Tensor of shape (num_elite,), weights for each sample in the dataset
        """

        if not self.diff_continuous_training:
            self.model.init_nn()

        x, conditions = dataset
        return self.model.fit(x, *conditions, weights=weights, **self.diff_kwargs)
