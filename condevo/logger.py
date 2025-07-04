import os
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None


class TensorboardLogger:
    def __init__(self, log_dir=None, model=None):
        self.log_dir = log_dir
        self.writer = None
        self.model = None
        self.generation = 0
        self.run = None

    # def log_hparams(self, hparams: dict, metrics: dict):
    #     if not self.log_dir:
    #         return
    #
    #     self.writer.add_hparams(hparams, metrics)
    #     self.generation += 1

    # def log_dataset(self, x, weights, *conditions):
    #     if not self.log_dir:
    #         return
    #
    #     # first, log fitness weights: (N, 1) tensor
    #     self.log_histogram("fitness_weights", weights.flatten(), self.generation)
    #
    #     # log dataset: (N, d) tensor
    #     self.writer.add_embedding(mat=x, metadata=weights, global_step=self.generation, tag="dataset")
    #
    #     # log conditions: list of (N, k) tensors
    #     for i, c in enumerate(conditions):
    #         if len(c.shape) == 2 and c.shape[1] == 1:
    #             c = c.flatten()
    #
    #         if len(c.shape) == 1:
    #             self.log_histogram(f"condition_{i}", c, self.generation)
    #
    #         elif len(c.shape) == 2:
    #             self.writer.add_embedding(mat=c, metadata=weights, global_step=self.generation, tag=f"condition_{i}")
    #
    # def log_metrics(self, metrics: dict, step: int):
    #     if not self.log_dir:
    #         return
    #
    #     for k, v in metrics.items():
    #         k = k + "/gen_" + str(self.generation)
    #         self.log_scalar(k, v, step)

    def log_scalar(self, tag, value, step):
        if not self.log_dir or not self.writer:
            return

        self.writer.add_scalar(tag + f"_{self.run}", value, step)
        self.writer.flush()

    # def log_histogram(self, tag, values, step):
    #     if not self.log_dir:
    #         return
    #
    #     self.writer.add_histogram(tag, values, step)

    def next(self):
        self.generation += 1

        if self.log_dir and SummaryWriter is not None:
            run = os.path.join(self.log_dir, f"run_{self.generation}")
            if os.path.exists(run):
                if self.run is None:
                    self.run = len([f for f in os.listdir(self.log_dir) if f.startswith("events.out")])

            else:
                os.makedirs(run, exist_ok=True)

            if self.run is None:
                self.run = 0

            if self.writer:
                self.writer.close()

            self.writer = SummaryWriter(run)

    def __del__(self):
        try:
            self.writer.close()
        except Exception:
            pass
