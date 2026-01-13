""" Classes in this module are based on David Ha's `estools`, and adapted under the MIT licence (2024).
    You can find the original implementation at https://github.com/hardmaru/estool.
"""
from torch import Tensor, tensor, stack, cat
import numpy as np
from ..es import utils
from typing import Union


class CMAES:
    """
    Covariance Matrix Adaptation Evolutionary Strategy (CMAES)
    """

    def __init__(self, num_params,
                 sigma_init=1.0,
                 popsize=255,
                 weight_decay=0.01,
                 reg='l2',
                 x0=None,
                 inopts=None
                 ):
        """Constructs a CMA-ES solver, based on Hansen's `cma` module.

        :param num_params: number of model parameters.
        :param sigma_init: initial standard deviation.
        :param popsize: population size.
        :param weight_decay: weight decay coefficient.
        :param reg: Choice between 'l2' or 'l1' norm for weight decay regularization.
        :param inopts: dict-like CMAOptions, forwarded to cma.CMAEvolutionStrategy constructor).
        :param x0: (Optional) either (i) a single or (ii) several initial guesses for a good solution,
                   defaults to None (initialize via `np.zeros(num_parameters)`).
                   In case (i), the population is seeded with x0.
                   In case (ii), the population is seeded with mean(x0, axis=0) and x0 is subsequently injected.
        """

        self.popsize = popsize

        inopts = inopts or {}
        inopts['popsize'] = self.popsize

        self.num_params = num_params
        self.sigma_init = sigma_init
        self.weight_decay = weight_decay
        self.reg = reg
        self.solutions = None
        self.fitness = None

        # HANDLE INITIAL SOLUTIONS
        inject_solutions = None
        if x0 is None:
            x0 = np.zeros(self.num_params)

        elif isinstance(x0, np.ndarray):
            x0 = np.atleast_2d(x0)
            inject_solutions = x0
            x0 = np.mean(x0, axis=0)

        # INITIALIZE
        import cma
        self.cma = cma.CMAEvolutionStrategy(x0, self.sigma_init, inopts)

        if inject_solutions is not None:
            if len(inject_solutions) == self.popsize:
                self.flush(inject_solutions)
            else:
                self.inject(inject_solutions)  # INJECT POTENTIALLY PROVIDED SOLUTIONS

    def inject(self, solutions=None):
        if solutions is not None:
            self.cma.inject(solutions, force=True)

    def flush(self, solutions):
        self.cma.ary = solutions
        self.solutions = solutions

    def rms_stdev(self):
        sigma = self.cma.result[6]
        return np.mean(np.sqrt(sigma * sigma))

    def ask(self):
        '''returns a list of parameters'''
        self.solutions = np.array(self.cma.ask())
        return tensor(self.solutions)

    def tell(self, reward_table_result):
        if not isinstance(reward_table_result, Tensor):
            reward_table = tensor(reward_table_result)
        else:
            reward_table = reward_table_result.clone()

        if self.weight_decay > 0:
            reg = utils.compute_weight_decay(self.weight_decay, self.solutions, reg=self.reg)
            reward_table += reg

        try:
            reward_table = reward_table.numpy()
        except:
            reward_table = reward_table.cpu().numpy()

        self.cma.tell(self.solutions, (-reward_table).tolist())  # convert minimizer to maximizer.

        fitness_argsort = np.argsort(reward_table)[::-1]  # sort in descending order
        self.fitness = reward_table[fitness_argsort]
        self.solutions = self.solutions[fitness_argsort]

    def current_param(self):
        return self.cma.result[5]  # mean solution, presumably better with noise

    def set_mu(self, mu):
        pass

    def best_param(self):
        return self.cma.result[0]  # best evaluated solution

    def result(self):  # return best params so far, along with historically best reward, curr reward, sigma
        r = self.cma.result
        return r[0], -r[1], -r[1], r[6]


class SimpleGA:
    """
    Simple Genetic Algorithm.
    """
    def __init__(self, num_params,
                 sigma_init=1.0,
                 sigma_decay=0.999,
                 sigma_limit=0.01,
                 popsize=256,
                 elite_ratio=0.1,
                 forget_best=False,
                 weight_decay=0.01,
                 reg='l2',
                 x0=None,
                 ):
        """ Constructs a simple genetic algorithm instance.

        :param num_params: number of model parameters.
        :param sigma_init: initial standard deviation.
        :param sigma_decay: anneal standard deviation.
        :param sigma_limit: stop annealing if less than this.
        :param popsize: population size.
        :param elite_ratio: percentage of the elites.
        :param forget_best: forget the historical best elites.
        :param weight_decay: weight decay coefficient.
        :param reg: Choice between 'l2' or 'l1' norm for weight decay regularization.
        :param x0: initial guess for a good solution, defaults to None (initialize via np.zeros(num_parameters)).
        """

        self.num_params = num_params
        self.sigma_init = sigma_init
        self.sigma_decay = sigma_decay
        self.sigma_limit = sigma_limit
        self.popsize = popsize

        self.elite_ratio = elite_ratio
        self.elite_popsize = int(np.ceil(self.popsize * self.elite_ratio))
        self.curr_best_reward = np.nan

        # ADDING option to start from prior solution
        x0 = np.zeros(self.num_params) if x0 is None else np.asarray(x0)
        self.elite_params = np.stack([x0] * self.elite_popsize)
        # self.elite_params = np.zeros((self.elite_popsize, self.num_params))
        self.best_param = np.copy(self.elite_params[0])  # np.zeros(self.num_params)

        self.sigma = self.sigma_init
        self.elite_rewards = np.zeros(self.elite_popsize)
        self.best_reward = 0
        self.first_iteration = True
        self.forget_best = forget_best
        self.weight_decay = weight_decay
        self.reg = reg

    def rms_stdev(self):
        return self.sigma  # same sigma for all parameters.

    def ask(self):
        '''returns a list of parameters'''
        self.epsilon = np.random.randn(self.popsize, self.num_params) * self.sigma
        solutions = []

        def mate(a, b):
            c = np.copy(a)
            idx = np.where(np.random.rand((c.size)) > 0.5)
            c[idx] = b[idx]
            return c

        elite_range = range(self.elite_popsize)
        for i in range(self.popsize):
            idx_a = np.random.choice(elite_range)
            idx_b = np.random.choice(elite_range)
            child_params = mate(self.elite_params[idx_a], self.elite_params[idx_b])
            solutions.append(child_params + self.epsilon[i])

        solutions = np.array(solutions)
        self.solutions = solutions

        return solutions

    def tell(self, reward_table_result):
        # input must be a numpy float array
        assert (len(reward_table_result) == self.popsize), "Inconsistent reward_table size reported."

        reward_table = np.array(reward_table_result)

        if self.weight_decay > 0:
            reg = utils.compute_weight_decay(self.weight_decay, self.solutions, reg=self.reg)
            reward_table += reg

        if self.forget_best or self.first_iteration:
            reward = reward_table
            solution = self.solutions
        else:
            reward = np.concatenate([reward_table, self.elite_rewards])
            solution = np.concatenate([self.solutions, self.elite_params])

        idx = np.argsort(reward)[::-1][0:self.elite_popsize]

        self.elite_rewards = reward[idx]
        self.elite_params = solution[idx]

        self.curr_best_reward = self.elite_rewards[0]

        if self.first_iteration or (self.curr_best_reward > self.best_reward):
            self.first_iteration = False
            self.best_reward = self.elite_rewards[0]
            self.best_param = np.copy(self.elite_params[0])

        if self.sigma > self.sigma_limit:
            self.sigma *= self.sigma_decay

    def current_param(self):
        return self.elite_params[0]

    def set_mu(self, mu):
        pass

    def best_param(self):
        return self.best_param

    def flush(self, solutions):
        self.elite_params = solutions

    def result(self):  # return best params so far, along with historically best reward, curr reward, sigma
        return self.best_param, self.best_reward, self.curr_best_reward, self.sigma


class OpenES:
    ''' Basic Version of OpenAI Evolution Strategies.
    '''
    def __init__(self, num_params,
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
                 ):
        """ Constructs an evolutionary solver instance following the OpenAI algorithm.

        :param num_params: number of model parameters.
        :param sigma_init: initial standard deviation.
        :param sigma_decay: anneal standard deviation.
        :param sigma_limit: stop annealing if less than this.
        :param learning_rate: learning rate for standard deviation.
        :param learning_rate_decay: annealing the learning rate.
        :param learning_rate_limit: stop annealing learning rate.
        :param popsize: population size.
        :param antithetic: whether to use antithetic sampling.
        :param weight_decay: weight decay coefficient.
        :param reg: Choice between 'l2' or 'l1' norm for weight decay regularization.
        :param rank_fitness: use rank rather than fitness numbers.
        :param forget_best: forget historical best
        :param x0: initial guess for a good solution, defaults to None (initialize via np.zeros(num_parameters)).
        """

        self.num_params = num_params
        self.sigma_decay = sigma_decay
        self.sigma = sigma_init
        self.sigma_init = sigma_init
        self.sigma_limit = sigma_limit
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.learning_rate_limit = learning_rate_limit
        self.popsize = popsize
        self.antithetic = antithetic
        if self.antithetic:
            assert (self.popsize % 2 == 0), "Population size must be even"
            self.half_popsize = int(self.popsize / 2)

        self.reward = np.zeros(self.popsize)

        # BH: ADDING option to start from prior solution
        self.mu = np.zeros(self.num_params) if x0 is None else np.asarray(x0)  # np.zeros(self.num_params)
        self.best_mu = np.copy(self.mu[0])  # np.zeros(self.num_params)

        self.best_reward = 0
        self.first_interation = True
        self.forget_best = forget_best
        self.weight_decay = weight_decay
        self.reg = reg
        self.rank_fitness = rank_fitness
        if self.rank_fitness:
            self.forget_best = True  # always forget the best one if we rank
        # choose optimizer
        self.optimizer = utils.Adam(mu=self.best_mu, num_params=num_params, stepsize=learning_rate)

    def rms_stdev(self):
        sigma = self.sigma
        return np.mean(np.sqrt(sigma * sigma))

    def ask(self):
        '''returns a list of parameters'''
        # antithetic sampling
        if self.antithetic:
            self.epsilon_half = np.random.randn(self.half_popsize, self.num_params)
            self.epsilon = np.concatenate([self.epsilon_half, - self.epsilon_half])
        else:
            self.epsilon = np.random.randn(self.popsize, self.num_params)

        self.solutions = self.mu.reshape(1, self.num_params) + self.epsilon * self.sigma

        return self.solutions

    def tell(self, reward_table_result):
        # input must be a numpy float array
        assert (len(reward_table_result) == self.popsize), "Inconsistent reward_table size reported."

        reward = np.array(reward_table_result)

        if self.rank_fitness:
            reward = utils.compute_centered_ranks(reward)

        if self.weight_decay > 0:
            reg = utils.compute_weight_decay(self.weight_decay, self.solutions, reg=self.reg)
            reward += reg

        idx = np.argsort(reward)[::-1]

        best_reward = reward[idx[0]]
        best_mu = self.solutions[idx[0]]

        self.curr_best_reward = best_reward
        self.curr_best_mu = best_mu

        if self.first_interation:
            self.first_interation = False
            self.best_reward = self.curr_best_reward
            self.best_mu = best_mu
        else:
            if self.forget_best or (self.curr_best_reward > self.best_reward):
                self.best_mu = best_mu
                self.best_reward = self.curr_best_reward

        # main bit:
        # standardize the rewards to have a gaussian distribution
        normalized_reward = (reward - np.mean(reward)) / np.std(reward)
        change_mu = 1. / (self.popsize * self.sigma) * np.dot(self.epsilon.T, normalized_reward)

        # self.mu += self.learning_rate * change_mu

        self.optimizer.stepsize = self.learning_rate
        update_ratio = self.optimizer.update(-change_mu)

        # adjust sigma according to the adaptive sigma calculation
        if (self.sigma > self.sigma_limit):
            self.sigma *= self.sigma_decay

        if (self.learning_rate > self.learning_rate_limit):
            self.learning_rate *= self.learning_rate_decay

    def current_param(self):
        return self.curr_best_mu

    def set_mu(self, mu):
        self.mu = np.array(mu)

    def best_param(self):
        return self.best_mu

    def flush(self, solutions):
        self.solutions = solutions

    def result(self):  # return best params so far, along with historically best reward, curr reward, sigma
        return (self.best_mu, self.best_reward, self.curr_best_reward, self.sigma)


class PEPG:
    '''
    Extension of PEPG with bells and whistles.
    '''
    def __init__(self, num_params,
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
                 ):  #
        """ Constructs a `PEPG` solver instance.

        :param num_params: number of model parameters.
        :param sigma_init: initial standard deviation.
        :param sigma_alpha: learning rate for standard deviation.
        :param sigma_decay: anneal standard deviation.
        :param sigma_limit: stop annealing if less than this.
        :param sigma_max_change: clips adaptive sigma to 20%.
        :param learning_rate: learning rate for standard deviation.
        :param learning_rate_decay: annealing the learning rate.
        :param learning_rate_limit: stop annealing learning rate.
        :param elite_ratio: if > 0, then ignore learning_rate.
        :param popsize: population size.
        :param average_baseline: set baseline to average of batch.
        :param weight_decay: weight decay coefficient.
        :param reg: Choice between 'l2' or 'l1' norm for weight decay regularization.
        :param rank_fitness: use rank rather than fitness numbers.
        :param forget_best: don't keep the historical best solution.
        :param x0: initial guess for a good solution, defaults to None (initialize via np.zeros(num_parameters)).
        """

        self.num_params = num_params
        self.sigma_init = sigma_init
        self.sigma_alpha = sigma_alpha
        self.sigma_decay = sigma_decay
        self.sigma_limit = sigma_limit
        self.sigma_max_change = sigma_max_change
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.learning_rate_limit = learning_rate_limit
        self.popsize = popsize
        self.average_baseline = average_baseline
        if self.average_baseline:
            assert (self.popsize % 2 == 0), "Population size must be even"
            self.batch_size = int(self.popsize / 2)
        else:
            assert (self.popsize & 1), "Population size must be odd"
            self.batch_size = int((self.popsize - 1) / 2)

        # option to use greedy es method to select next mu, rather than using drift param
        self.elite_ratio = elite_ratio
        self.elite_popsize = int(self.popsize * self.elite_ratio)
        self.use_elite = False
        if self.elite_popsize > 0:
            self.use_elite = True

        self.forget_best = forget_best
        self.batch_reward = np.zeros(self.batch_size * 2)

        # BH: ADDING option to start from prior solution
        self.mu = np.zeros(self.num_params) if x0 is None else np.asarray(x0)  # np.zeros(self.num_params)
        self.best_mu = np.copy(self.mu[0])  # np.zeros(self.num_params)
        self.curr_best_mu = np.copy(self.mu[0])  # np.zeros(self.num_params)

        self.sigma = np.ones(self.num_params) * self.sigma_init
        self.best_reward = 0
        self.first_interation = True
        self.weight_decay = weight_decay
        self.reg = reg
        self.rank_fitness = rank_fitness
        if self.rank_fitness:
            self.forget_best = True  # always forget the best one if we rank
        # choose optimizer
        self.optimizer = utils.Adam(mu=self.best_mu, num_params=num_params, stepsize=learning_rate)

    def rms_stdev(self):
        sigma = self.sigma
        return np.mean(np.sqrt(sigma * sigma))

    def ask(self):
        '''returns a list of parameters'''
        # antithetic sampling
        self.epsilon = np.random.randn(self.batch_size, self.num_params) * self.sigma.reshape(1, self.num_params)
        self.epsilon_full = np.concatenate([self.epsilon, - self.epsilon])
        if self.average_baseline:
            epsilon = self.epsilon_full
        else:
            # first population is mu, then positive epsilon, then negative epsilon
            epsilon = np.concatenate([np.zeros((1, self.num_params)), self.epsilon_full])
        solutions = self.mu.reshape(1, self.num_params) + epsilon
        self.solutions = solutions
        return solutions

    def tell(self, reward_table_result):
        # input must be a numpy float array
        assert (len(reward_table_result) == self.popsize), "Inconsistent reward_table size reported."

        reward_table = np.array(reward_table_result)

        if self.rank_fitness:
            reward_table = utils.compute_centered_ranks(reward_table)

        if self.weight_decay > 0:
            reg = utils.compute_weight_decay(self.weight_decay, self.solutions, reg=self.reg)
            reward_table += reg

        reward_offset = 1
        if self.average_baseline:
            b = np.mean(reward_table)
            reward_offset = 0
        else:
            b = reward_table[0]  # baseline

        reward = reward_table[reward_offset:]
        if self.use_elite:
            idx = np.argsort(reward)[::-1][0:self.elite_popsize]
        else:
            idx = np.argsort(reward)[::-1]

        best_reward = reward[idx[0]]
        if (best_reward > b or self.average_baseline):
            best_mu = self.mu + self.epsilon_full[idx[0]]
            best_reward = reward[idx[0]]
        else:
            best_mu = self.mu
            best_reward = b

        self.curr_best_reward = best_reward
        self.curr_best_mu = best_mu

        if self.first_interation:
            self.sigma = np.ones(self.num_params) * self.sigma_init
            self.first_interation = False
            self.best_reward = self.curr_best_reward
            self.best_mu = best_mu
        else:
            if self.forget_best or (self.curr_best_reward > self.best_reward):
                self.best_mu = best_mu
                self.best_reward = self.curr_best_reward

        # short hand
        epsilon = self.epsilon
        sigma = self.sigma

        # update the mean

        # move mean to the average of the best idx means
        if self.use_elite:
            self.mu += self.epsilon_full[idx].mean(axis=0)
        else:
            rT = (reward[:self.batch_size] - reward[self.batch_size:])
            change_mu = np.dot(rT, epsilon)
            self.optimizer.stepsize = self.learning_rate
            update_ratio = self.optimizer.update(-change_mu)  # adam, rmsprop, momentum, etc.
            # self.mu += (change_mu * self.learning_rate) # normal SGD method

        # adaptive sigma
        # normalization
        if (self.sigma_alpha > 0):
            stdev_reward = 1.0
            if not self.rank_fitness:
                stdev_reward = reward.std()
            S = ((epsilon * epsilon - (sigma * sigma).reshape(1, self.num_params)) / sigma.reshape(1, self.num_params))
            reward_avg = (reward[:self.batch_size] + reward[self.batch_size:]) / 2.0
            rS = reward_avg - b
            delta_sigma = (np.dot(rS, S)) / (2 * self.batch_size * stdev_reward)

            # adjust sigma according to the adaptive sigma calculation
            # for stability, don't let sigma move more than 10% of orig value
            change_sigma = self.sigma_alpha * delta_sigma
            change_sigma = np.minimum(change_sigma, self.sigma_max_change * self.sigma)
            change_sigma = np.maximum(change_sigma, - self.sigma_max_change * self.sigma)
            self.sigma += change_sigma

        if (self.sigma_decay < 1):
            self.sigma[self.sigma > self.sigma_limit] *= self.sigma_decay

        if (self.learning_rate_decay < 1 and self.learning_rate > self.learning_rate_limit):
            self.learning_rate *= self.learning_rate_decay

    def flush(self, solutions):
        self.solutions = solutions

    def current_param(self):
        return self.curr_best_mu

    def set_mu(self, mu):
        self.mu = np.array(mu)

    def best_param(self):
        return self.best_mu

    def result(self):  # return best params so far, along with historically best reward, curr reward, sigma
        return (self.best_mu, self.best_reward, self.curr_best_reward, self.sigma)


class MultistartES:
    def __init__(self, cls: Union[str, type]=CMAES, num_starts=1, popsize=256, **kwargs):
        self.num_starts = num_starts
        popsizes = [popsize // num_starts for i in range(num_starts)]
        for i in range(popsize % num_starts):
            popsizes[i] += 1  # add reminders evenly

        Cls = self._get_cls(cls)
        self.es = [Cls(popsize=pi, **kwargs) for pi in popsizes]

    def _get_cls(self, cls):
        if isinstance(cls, type):
            return cls

        assert cls in globals().keys(), f"cls {cls} not found in globals {globals()}"
        return globals()[cls]

    def inject(self, solutions=None):
        if solutions is not None:
            [es.inject(solutions) for es in self.es]

    def flush(self, solutions):
        [es.flush(solutions) for es in self.es]

    def rms_stdev(self):
        [es.rms_stdev() for es in self.es]

    def ask(self):
        '''returns a list of parameters'''
        self.solutions = [es.ask() for es in self.es]
        return cat(self.solutions)

    def tell(self, reward_table_result):
        solution_slize = [len(s) for s in self.solutions]
        reward_tables = []
        for i, s in enumerate(solution_slize):
            reward_tables.append(reward_table_result[sum(solution_slize[:i]):sum(solution_slize[:i + 1])])

        [es.tell(r) for es, r in zip(self.es, reward_tables)]

    def current_param(self):
        return [es.current_parma() for es in self.es][self.best_es_id]

    def set_mu(self, mu):
        return [es.set_mu(mu) for es in self.es]

    def best_param(self):
        return [es.best_param() for es in self.es][self.best_es_id]

    def result(self):
        ri = [es.result() for es in self.es] # for every es: (best_param, hist_best_rward, current_reward, sigma)
        return ri[self.best_es_id]

    @property
    def best_es_id(self):
        ri = [es.result() for es in self.es]  # for every es: (best_param, hist_best_rward, current_reward, sigma)
        hist_best = [b for _, b, _, _ in ri]
        return np.argmin(hist_best)
