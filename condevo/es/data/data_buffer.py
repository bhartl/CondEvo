from typing import Union
from torch import Tensor, stack, randperm, cdist, concatenate


class DataBuffer:
    """
    A class to manage a buffer of data with a maximum size.
    """

    # Types of population management
    POP_QUALITY = "quality"
    POP_DIVERSITY = "diversity"
    POP_TYPES = [POP_QUALITY, POP_DIVERSITY]

    def __init__(self, max_size: int = 0, num_conditions=0, avoid_nans=True, pop_type=POP_QUALITY, shuffle=True):
        """
        Initialize the DataBuffer with a maximum size.

        :param max_size: The maximum size of the buffer. No limit if 0.
        """
        self.max_size = max_size
        self.avoid_nans = avoid_nans
        self.pop_type = pop_type
        self.shuffle = shuffle
        self.buffer = {"x": [], "fitness": [], }
        if num_conditions:
            self.buffer["conditions"] = [[] for _ in range(num_conditions)]

        self.info = {}

    def __getitem__(self, key):
        """ Access the buffer with a key. """
        return self.buffer[key]

    def push(self, x: Tensor, fitness: Tensor, *conditions: Tensor):
        """
        Push new data into the buffer.

        :param x: The data to be added to the buffer.
        :param fitness: The fitness scores associated with the data.
        :param conditions: Additional conditions to be added to the buffer.
        """

        if self.avoid_nans:
            # find indices in x and fitness that are not NaN
            x_nans = x.isnan().any(dim=1)
            fitness_nans = fitness.isnan()
            condition_nans = [condition.isnan() for condition in conditions]

            nan_indices = x_nans | fitness_nans
            for condition_nan in condition_nans:
                condition_nan = condition_nan.any(dim=1) if condition_nan.ndim > 1 else condition_nan
                nan_indices = nan_indices | condition_nan

            # remove NaN values from x, fitness, and conditions
            x = x[~nan_indices]
            fitness = fitness[~nan_indices]
            conditions = [condition[~nan_indices] for condition in conditions]

        if self.max_size and (len(self.buffer["x"]) + len(x) > self.max_size):
            _, (x, fitness, *conditions) = self.pop(x, fitness, *conditions)

        self.buffer["x"].extend([xi for xi in x.clone()])
        self.buffer["fitness"].extend([fi for fi in fitness.clone()])

        for i, condition in enumerate(conditions):
            self.buffer["conditions"][i].extend([ci for ci in condition.clone()])

    def  pop(self, x: Tensor, fitness: Tensor, *conditions: Tensor) -> Union[tuple, None]:
        if not self.max_size:
            return (None, None, None), (x, fitness, *conditions)

        if self.pop_type == self.POP_QUALITY:
            remove, replace = self.pop_quality(x, fitness, *conditions)

        elif self.pop_type == self.POP_DIVERSITY:
            remove, replace = self.pop_diversity(x, fitness, *conditions)

        else:
            if not hasattr(self, f"pop_{self.pop_type}"):
                raise ValueError(f"Unknown population management `pop_type`: {self.pop_type}")

            pop_method = getattr(self, f"pop_{self.pop_type}")
            remove, replace = pop_method(x, fitness, *conditions)

        x_pop = self.x[remove]
        fitness_pop = self.fitness[remove]
        conditions_pop = ()
        if self.has_conditions:
            conditions_pop = []
            for i, condition in enumerate(self.buffer["conditions"]):
                conditions_pop.append(stack([condition[j] for j in remove]))

        # remove old samples from buffer
        self.buffer["x"] = [self.buffer["x"][i] for i in range(len(self.buffer["x"])) if i not in remove]
        self.buffer["fitness"] = [self.buffer["fitness"][i] for i in range(len(self.buffer["fitness"])) if
                                  i not in remove]
        if self.has_conditions:
            for i, condition in enumerate(self.buffer["conditions"]):
                self.buffer["conditions"][i] = [condition[j] for j in range(len(condition)) if j not in remove]

        x = x[replace]
        fitness = fitness[replace]
        conditions = [condition[replace] for condition in conditions]

        return (x_pop, fitness_pop, *conditions_pop), (x, fitness, *conditions)

    def pop_quality(self, x, fitness, *conditions):
        """
        Remove the worst samples from the buffer based on fitness.

        :param x: The data to be added to the buffer.
        :param fitness: The fitness scores associated with the data.
        :param conditions: Additional conditions to be added to the buffer.
        """
        num_replace = max([self.size + len(x) - self.max_size, 0])

        if self.shuffle:
            size = self.size
            shuffle_indices = randperm(size)
            self.buffer["x"] = [self.buffer["x"][i] for i in shuffle_indices]
            self.buffer["fitness"] = [self.buffer["fitness"][i] for i in shuffle_indices]
            if self.has_conditions:
                for i, condition in enumerate(self.buffer["conditions"]):
                    self.buffer["conditions"][i] = [condition[j] for j in shuffle_indices]

        # remove old samples from buffer with the lowest fitness
        indices = self.fitness.flatten().argsort()
        remove = indices[:num_replace]  # remove the worst samples
        replace = [True] * len(x)  # replace all samples, i.e., all indices in x
        return remove, replace

    def pop_diversity(self, x, fitness, *conditions):
        buffered_x = self.x.clone()
        buffered_fitness = self.fitness.clone()

        indices = {}
        for i, (xi, fi) in enumerate(zip(reversed(x), reversed(fitness))):
            # find the most similar sample in the buffer
            distances = cdist(xi.reshape(1, -1), buffered_x).flatten()
            # get the index of the most similar sample
            for index in distances.argsort()[:5]:
                # replace the most similar sample with the new sample, if it is better
                if buffered_fitness[index] < fi:
                    # replace the nearest sample in the buffer with a better one
                    buffered_x[index] = xi
                    buffered_fitness[index] = fi
                    indices[index] = len(x) - 1 - i  # reverse index
                    break

        remove = list(indices.keys())
        replace = list(indices.values())
        return remove, replace

    @property
    def has_conditions(self):
        """
        Check if the buffer has conditions.

        :return: True if the buffer has conditions, False otherwise.
        """
        return "conditions" in self.buffer

    @property
    def size(self):
        """
        Get the current size of the buffer.

        :return: The current size of the buffer.
        """
        return len(self.buffer["x"])

    @property
    def x(self) -> Tensor:
        """
        Get the data in the buffer.

        :return: The data in the buffer.
        """
        if not len(self.buffer["x"]):
            return ()

        return stack(self.buffer["x"])

    @x.setter
    def x(self, value):
        """
        Set the data in the buffer.

        :param value: The data to be set in the buffer.
        """
        self.buffer["x"] = value

    @property
    def fitness(self) -> Tensor:
        """
        Get the fitness scores in the buffer.

        :return: The fitness scores in the buffer.
        """
        if not len(self.buffer["fitness"]):
            return ()

        return stack(self.buffer["fitness"])

    @fitness.setter
    def fitness(self, value):
        """
        Set the fitness scores in the buffer.

        :param value: The fitness scores to be set in the buffer.
        """
        self.buffer["fitness"] = value

    @property
    def conditions(self) -> Union[tuple, None]:
        """
        Get the conditions in the buffer.

        :return: The conditions in the buffer.
        """
        if "conditions" not in self.buffer:
            return None

        return tuple((stack(condition) if len(condition) else Tensor())
                     for condition in self.buffer["conditions"] if condition)

    def get_condition(self, index):
        """
        Get the condition at a specific index.

        :param index: The index of the condition to be retrieved.
        :return: The condition at the specified index.
        """
        if "conditions" not in self.buffer:
            return None

        return stack(self.buffer["conditions"][index]) if len(self.buffer["conditions"][index]) else Tensor()

    @conditions.setter
    def conditions(self, value: Union[list, tuple]):
        """
        Set the conditions in the buffer.

        :param value: The conditions to be set in the buffer.
        """
        self.buffer["conditions"] = value

    def clear(self):
        """
        Clear the buffer.
        """
        self.buffer = {"x": [], "fitness": []}
        if self.has_conditions:
            self.buffer["conditions"] = [[] for _ in range(len(self.buffer["conditions"]))]

        self.info = {}
