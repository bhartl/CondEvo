from torch import Tensor, cat, randn, multinomial, cdist, sort, stack, exp, log
from . import Condition


class KNNNoveltyCondition(Condition):
    def __init__(self, k=10, metric=2.0, beta=10., weight_by_fitness=True, eps=1e-8):
        Condition.__init__(self)
        self.k = k
        self.metric = metric
        self.beta = beta
        self.weight_by_fitness = weight_by_fitness
        self.eps = eps

    def novelty_score(self, x, buffer):
        assert self.k < len(buffer), f"k (={self.k}) must be smaller than the buffer size (={len(buffer)})"

        novelty_scores = []
        for xi in x:
            distance_to_buffer = cdist(xi[None, :], buffer, self.metric)[0]
            nearest_neigh_dist, _ = sort(distance_to_buffer, dim=0)  # val, idx
            novelty_score = nearest_neigh_dist[self.k:].nanmean()
            # novelty_scores.append(novelty_score)
            log_novelty_score = log(novelty_score + self.eps)
            novelty_scores.append(log_novelty_score)

        return stack(novelty_scores)

    def evaluate(self, charles_instance, x, f):
        has_buffer = "x" in charles_instance.buffer
        all_x = x
        if has_buffer:
            buffered_x = charles_instance.buffer["x"]
            all_x = cat([x, buffered_x])

        # evaluate novelty score of all x to all x (of new samples and buffer solutions)
        novelty_scores = self.novelty_score(all_x, all_x)

        if has_buffer:
            # workaround: update old conditions, should be done within CHARLES, TODO
            condition_num = [i for i, c in enumerate(charles_instance.conditions) if c is self][0]
            charles_instance.buffer["conditions"][condition_num][:] = novelty_scores[len(x):, None]

        # return new scores
        novelty_scores = novelty_scores[:len(x)]
        return novelty_scores

    def fitness_weight(self, fitness_scores):
        f = 1.

        if self.weight_by_fitness:
            nans = fitness_scores.isnan()
            nan_min = fitness_scores[~nans].min()
            nan_max = fitness_scores[~nans].max()
            f = 1. - (fitness_scores - nan_min) / (nan_max - nan_min)

            if f.isnan().any():
                f[f.isnan()] = 0.

        return f

    def boltzmann_selection(self, novelty_scores, fitness_scores, beta=1.0):
        # Compute the Boltzmann probabilities
        nan_min = novelty_scores[~novelty_scores.isnan()].min()
        novelty_scores = novelty_scores + nan_min.abs()

        exp_scores = exp(-beta * self.fitness_weight(fitness_scores) / (novelty_scores + self.eps)) + self.eps
        if exp_scores.isnan().any():
            exp_scores[exp_scores.isnan()] = 0.

        if exp_scores.sum() == 0.:
            exp_scores[:] = 1.

        probabilities = exp_scores / exp_scores.sum()
        return probabilities

    def sample(self, charles_instance, num_samples):
        # sample target quadrant
        c = [charles_instance.buffer["conditions"][i] for i, c in enumerate(charles_instance.conditions) if c is self][0]
        f = charles_instance.buffer["fitness"]

        # draw from Boltzmann distribution
        p = self.boltzmann_selection(novelty_scores=c.flatten(), fitness_scores=f, beta=self.beta)

        try:
            idx = multinomial(p, num_samples, replacement=True)
        except:
            print("here")
            print(f)
            print(p)
            raise

        # add Gaussian noise to drawn samples
        samples = randn(num_samples)[:, None] * (0.5 * (c[idx].std())**2) + c[idx]
        return samples

    def to_dict(self):
        return {"k": self.k, "metric": self.metric, "beta": self.beta, **Condition.to_dict(self)}

    def __str__(self):
        return f"KNNNoveltyCondition(k={self.k}, metric={self.metric}, beta={self.beta}, weight_by_fitness={self.weight_by_fitness})"

    def __repr__(self):
        return self.__str__()