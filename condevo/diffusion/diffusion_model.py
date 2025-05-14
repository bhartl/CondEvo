from torch import cat, optim, ones, zeros_like, Tensor, no_grad, vmap, randn, rand
from torch.nn import MSELoss, Module, ReLU
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from typing import Union
import torch
from ..logger import TensorboardLogger


class DM(Module):
    """ Diffusion Model base-class for condevo package. """

    def __init__(self, nn, num_steps=100, diff_range=None, lambda_range=0., param_mean=0.0, param_std=1.0,
                 epsilon=1e-8, log_dir="", sample_uniform=True, autoscaling=True):
        """ Initialize the Diffusion Model

        :param nn: torch.nn.Module, Neural network to be used for the diffusion model. Needs to have a `_build_model` method.
                   The dimension of the input tensor should be (batch_size, num_params + num_conditions + 1).
        :param num_steps: int, Number of steps for the diffusion model. Defaults to 100.
        :param diff_range: float, Parameter range for generated samples of the diffusion model. Defaults to None.
                           The diffusion model will also scale the input tensor to the standard normal distribution
                           of the given training data upon calling the `fit` method.
        :param lambda_range: float, Magnitude of loss if denoised parameters exceed parameter range. Defaults to 0.
        :param param_mean: float, Initial mean of the training data for the standard scaler. Defaults to 0.
        :param param_std: float, Initial standard deviation of the training data for the standard scaler. Defaults to 1.
        :param epsilon: float, Small value to avoid division by zero in the standard scaler and fitness weight. Defaults to 1e-8.
        :param log_dir: str, Directory for logging the training process. Defaults to "". If no directory is specified,
                        no logging will be performed.
        :param sample_uniform: bool, Whether to sample from uniformly distribution or from the standard normal
                               distribution. Defaults to True.
        :param autoscaling: bool, Whether to automatically scale the sampling range to the training data STD and mean.


        """
        super(DM, self).__init__()
        self.num_steps = num_steps
        self.nn = nn

        self.diff_range = diff_range
        self.lambda_range = lambda_range

        self.param_mean = param_mean
        self.param_std = param_std

        self.sample_uniform = sample_uniform
        self.autoscaling = autoscaling
        self.epsilon = epsilon

        self._logger = None
        self._log_dir = log_dir

    def init_nn(self):
        """ Initialize the neural network model """
        if hasattr(self.nn, '_build_model'):
            self.nn._build_model()

        else:
            raise NotImplementedError("Model does not have a `_build_model` method.")

    @property
    def num_conditions(self):
        return self.nn.num_conditions

    @property
    def logger(self):
        """ Return the logger instance for logging the training process """
        if self._logger is None:
            self._logger = TensorboardLogger(log_dir=self.log_dir, model=self)

        return self._logger

    @property
    def log_dir(self):
        """ Return the log directory for logging the training process """
        return self._log_dir

    @log_dir.setter
    def log_dir(self, log_dir):
        """ Set the log directory for logging the training process """
        if self._logger is not None:
            self._logger = None  # destroy the old logger instance

        self._log_dir = log_dir

    def diffuse(self, x, t):
        """ Abstract diffuse method the input tensor `x` at time `t` """
        raise NotImplementedError("Diffusion method not implemented, should be implemented in subclass.")

    def forward(self, x, t, *conditions):
        """Predict the noise `eps` with given `x` and `t`, `t` [0,1] represents the diffusion time step,
           where `t==0` is the data state (denoised) and `t==1` is the fully noisy state. """
        return self.nn(x, t, *conditions)

    def sample_point(self, xt, *conditions, t_start=0):
        pass

    def draw_random(self, num: int, *shape):
        """ Draw random samples from the standard normal distribution with given shape. """
        if not self.sample_uniform:
            # draw random samples from the standard normal distribution
            x_source = randn(num, *shape)
            if self.autoscaling:
                # scale to the training data STD and mean
                x_source = x_source * self.param_std + self.param_mean

        else:
            # draw random samples from a uniform distribution
            x_source = rand(num, *shape)
            if self.autoscaling:
                # scale to the training data STD and mean
                x_source = (x_source - 0.5) * self.param_std * 3. + self.param_mean

        return x_source

    @no_grad()
    def sample(self, shape: tuple, num: int = None, x_source: Tensor = None, conditions=None, t_start=None):
        r"""Sample `num` points from the diffusion model with given `shape` and `conditions`.
        If `num` is not specified, sample points from `x_source` tensor.

        Args:
            shape (tuple): Shape of the sampled points.
            num (int, optional): Number of points to be sampled. Defaults to None.
            x_source (torch.tensor, optional): Source tensor to sample points. Defaults to None.
            conditions (tuple, optional): Conditions for the diffusion model. Defaults to None.
            t_start (int, optional): Starting time step for the diffusion process. Defaults to None.

        Returns:
            torch.tensor: Sampled points from the diffusion model.
        """
        if (num is None) and (x_source is None):
            raise ValueError("Either `num` or `xt` should be specified")

        if (num is not None) and (x_source is not None):
            raise ValueError("Only one of `num` and `xt` should be specified")

        if conditions is None:
            conditions = tuple()

        if num is not None:
            x_source = self.draw_random(num, *shape)

        self.eval()
        self.nn.eval()
        sample_vectorized = vmap(self.sample_point, randomness='different')
        x_sampled = sample_vectorized(x_source, *conditions, t_start=t_start)

        # check for valid parameter range
        exceeding_x = self.exceeds_diff_range(x_sampled) > 0
        exceeding_count = 0
        while exceeding_x.any():
            # new sample points
            exceeding_x_source = self.draw_random(int(sum(exceeding_x)), *shape)

            if exceeding_count > 2:  # try 10 times to sample valid points
                # clamp to diff_range if too many iterations
                exceeding_x_source = exceeding_x_source.clamp(-self.diff_range, self.diff_range)
                x_sampled[exceeding_x] = exceeding_x_source
                break

            else:
                exceeding_conditions = [condition[exceeding_x] for condition in conditions]
                x_resampled = sample_vectorized(exceeding_x_source, *exceeding_conditions)

                # check for valid parameter range and integrate into samples
                exceeding_resampled = self.exceeds_diff_range(x_resampled) > 0
                valid_resampled = torch.where(exceeding_x)[0][~exceeding_resampled]
                x_sampled[valid_resampled] = x_resampled[~exceeding_resampled]

            exceeding_x = self.exceeds_diff_range(x_sampled) > 0
            exceeding_count += 1

        return x_sampled

    def eval_val_pred(self, x, *conditions):
        """ Evaluate the value (actual noise during diffusion) and predicted noise of the diffusion model.
            `x` is the original data, and `conditions` are the conditions for the diffusion model.
            During evaluation, random noise is added to the data to simulate the diffusion process,
            and the model is used to predict the noise. Returns the actual noise value and predicted noise.

            :param x: torch.tensor, Input data (original training data) for the diffusion model.
            :param conditions: tuple, Conditions for the diffusion model.
            :return: tuple, Actual noise value and predicted noise.
        """
        raise NotImplementedError("eval_loss method not implemented, returns tuple of value and prediction (v, v_pred).")

    @property
    def device(self):
        return next(self.nn.parameters()).device

    def regularize(self, x_batch, w_batch, *c_batch):
        """ Optional regularizer-function for the denoising during training, defaults to 0. """
        return 0.
    
    def exceeds_diff_range(self, x):
        """ If the parameter range is specified, evaluate the exceed of the parameter range (via ReLU),
            otherwise return zeros tensor. """
        if self.diff_range is None:
            return zeros_like(x[:, 0])

        # evaluate the exceed of the parameter range (via ReLU)
        return ReLU()((x ** 2 - self.diff_range ** 2).reshape(x.shape[0], -1)).mean(dim=-1).sqrt()

    @no_grad()
    def get_standard_scaler(self, x, weights=None, conditions=()):
        """ Update the standard scaler for the diffusion model. """

        # weight datapoints
        if weights is None:
            weights = ones(x.shape[0], *(1 for _ in range(len(x.shape[1:]))), device=x.device)

        else:
            # check for NaN values in weights
            nan_weights = torch.isnan(weights)
            if nan_weights.any():
                x = x[~nan_weights]
                conditions = [c[~nan_weights] for c in conditions]
                weights = weights[~nan_weights]

        # check for exceeding parameter range
        exceeding_x = self.exceeds_diff_range(x) > 0
        if exceeding_x.any():
            x = x[~exceeding_x]
            conditions = [c[~exceeding_x] for c in conditions]
            weights = weights[~exceeding_x]

        self.param_mean = x.mean(dim=0, keepdim=True)
        self.param_std = x.std(dim=0, keepdim=True)

        return x, weights, conditions

    def fit(self, x, *conditions, weights=None, optimizer: Union[str, type] = optim.Adam, max_epoch=100, lr=1e-3,
            weight_decay=1e-5, batch_size=32, scheduler="cosine"):
        """ Train the diffusion model to the given data.

        The diffusion model is first set to training mode, then the optimizer is initialized with the given parameters.
        The loss function is set to MSELoss. If weights are not specified, they are set to ones.
        After training, the model is set to evaluation mode.

        :param x: torch.tensor, Input data for the diffusion model.
        :param conditions: tuple, Conditions for the diffusion model (will be concatenated with x in last `dim`).
        :param weights: torch.tensor, Weights for data point in the loss function (to weight high-fitness appropriately).
                        Defaults to None.
        :param optimizer: str or type, Optimizer for the diffusion model. Defaults to optim.Adam.
        :param max_epoch: int, Maximum number of epochs for training. Defaults to 100.
        :param lr: float, Learning rate for the optimizer. Defaults to 1e-3.
        :param weight_decay: float, Weight decay for the optimizer. Defaults to 1e-5.
        :param batch_size: int, Batch size for the training data. Defaults to 32.
        :param scheduler: str or type, Scheduler for the optimizer. Defaults to None.

        :return: list, Loss history of the training process.
        """
        self.train()
        self.nn.train()

        if isinstance(optimizer, str):
            optimizer = getattr(optim, optimizer)

        optimizer = optimizer(self.parameters(), lr=lr, weight_decay=weight_decay)
        if scheduler == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, max_epoch, eta_min=1e-6)

        loss_function = MSELoss(reduction='none')

        # Scale data and weights to standard normal distribution, correct nans
        x, weights, conditions = self.get_standard_scaler(x=x, weights=weights, conditions=conditions)
        dataset = TensorDataset(x, *conditions, weights)
        training_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # log dataset
        # self.logger.log_dataset(x, weights, *conditions)

        loss_history = []
        for epoch in tqdm(range(int(max_epoch))):
            batch_loss = 0
            for x_batch, *c_batch, w_batch in training_dataloader:
                optimizer.zero_grad()
                v, v_pred = self.eval_val_pred(x_batch, *c_batch)
                loss = loss_function(v, v_pred) * w_batch
                reg_loss = self.regularize(x_batch, w_batch, *c_batch)
                loss = (loss + reg_loss).mean()
                loss.backward()
                batch_loss = batch_loss + loss.item()
                optimizer.step()

            batch_loss = batch_loss/batch_size
            loss_history.append(batch_loss)
            if scheduler is not None:
                scheduler.step()

        self.logger.log_scalar("evo/buffer/loss", loss_history[-1], self.logger.generation)
        self.logger.next()
        self.eval()
        self.nn.eval()
        return loss_history
