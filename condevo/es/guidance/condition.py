from torch import Tensor


class Condition:
    """ Base class for implementing conditions for targeted sampling with the CHARLES solver. """
    def __init__(self, ):
        pass

    def evaluate(self, charles_instance, x: Tensor, f: Tensor) -> Tensor:
        """ Evaluate the condition on the given Charles instance and input x.

        :param charles_instance: CHARLES instance
        :param x: torch.Tensor, input tensor representing the genotype parameters
        :param f: torch.Tensor, fitness values of the population
        :return: torch.Tensor, evaluation of the condition for each parameter input x
        """
        raise NotImplementedError("Evaluation method not implemented.")

    def transform(self, charles_instance, values: Tensor) -> Tensor:
        """ Transform the evaluated input condition (defaults to identity transformation).

        :param charles_instance: CHARLES instance
        :param values: torch.Tensor, input tensor representing the condition values
        :return: torch.Tensor, transformed condition values.
        """
        return values

    def sample(self, charles_instance, num_samples: int) -> Tensor:
        """ Sample from the condition.

        :param charles_instance: CHARLES instance
        :param num_samples: int, number of samples to generate
        :return: torch.Tensor, samples of condition values.
        """
        raise NotImplementedError("Sampling method not implemented.")

    def to_dict(self):
        """ Convert the condition to a dictionary. """
        return dict()

    def get_index(self, charles_instance):
        """ Get the index of the condition in the CHARLES instance.

        :param charles_instance: CHARLES instance
        :return: int, index of the condition in the CHARLES instance.
        """
        for i, c in enumerate(charles_instance.conditions):
            if c == self:
                return i

        raise ValueError("Condition not found in CHARLES instance.")
