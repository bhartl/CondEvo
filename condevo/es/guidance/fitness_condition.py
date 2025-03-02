from torch import abs, randn, Tensor, float32
from condevo.es.guidance import Condition


class FitnessCondition(Condition):
    """ Fitness condition based on Fisher's fundamental theorem of natural selection. """
    def __init__(self, scale: float = 1.0, greedy=False):
        """ Constructor of the FitnessCondition. """
        super().__init__()
        self.scale = scale
        self.greedy = greedy

    def evaluate(self, charles_instance, x: Tensor, f: Tensor) -> Tensor:
        """ Evaluate the condition on the given Charles instance and input x.

        :param charles_instance: CHARLES instance
        :param x: torch.Tensor, input tensor representing the genotype parameters
        :param f: torch.Tensor, fitness values of the population
        :return: torch.Tensor, evaluation of the condition for each parameter input x
        """
        return (f / self.scale).type(float32)

    def sample(self, charles_instance, num_samples: int) -> Tensor:
        """ Sample fitness values based on Fisher's fundamental theorem of natural selection (https://en.wikipedia.org/wiki/Fisher%27s_fundamental_theorem_of_natural_selection).

        It states:
            "The rate of increase in fitness of any organism at any time is equal to its genetic variance in fitness at that time."[4]

        Or in more modern terminology:
            "The rate of increase in the mean fitness of any organism, at any time, that is ascribable to natural selection acting through changes in gene frequencies, is exactly equal to its genetic variance in fitness at that time".

        :param charles_instance: CHARLES instance
        :param num_samples: int, number of fitness samples to draw from the temperature-dependent fitness distribution.
        :return: torch.Tensor of shape (num_samples, num_params), sampled solutions
        """

        # evaluate variance of the fitness values
        var_W = charles_instance.fitness.var()

        # fitness increase due to genetic variance
        dW = var_W * 0.5 * randn(num_samples, 1)

        if not self.greedy:
            # sample fitness values as Gaussian noise around the current fitness values
            fitness_condition = charles_instance.elite_fitness.mean() + abs(dW)

        else:
            # greedily sample the best fitness values
            fitness_condition = charles_instance.elite_fitness.max() + dW

        return fitness_condition / self.scale

    def to_dict(self):
        """ Convert the condition to a dictionary. """
        return dict(scale=self.scale, **Condition.to_dict(self))

    def __str__(self):
        """ Return the string representation of the condition. """
        return f"FitnessCondition(scale={self.scale}, greedy={self.greedy})"

    def __repr__(self):
        """ Return the string representation of the condition. """
        return self.__str__()
