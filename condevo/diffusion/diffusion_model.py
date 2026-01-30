from torch import cat, optim, ones, zeros_like, Tensor, no_grad, randn, rand
from torch.nn import MSELoss, Module, ReLU
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from typing import Union
import torch
from ..logger import TensorboardLogger
from ..preprocessing import Scaler
from copy import deepcopy


class DM(Module):
    """ Diffusion Model base-class for condevo package. """

    def __init__(self, nn, num_steps=100,
                 epsilon=1e-8, log_dir="",
                 scaler=None, diff_range=None, diff_range_filter=True,
                 lambda_range=0., clip_gradients=None, cfg_scale: float = 0.0,
                 ):
        """ Initialize the Diffusion Model

        :param nn: torch.nn.Module, Neural network to be used for the diffusion model. Needs to have a `_build_model` method.
                   The dimension of the input tensor should be (batch_size, num_params + num_conditions + 1).
        :param num_steps: int, Number of steps for the diffusion model. Defaults to 100.
        :param diff_range: float, Parameter range for generated samples of the diffusion model. Defaults to None.
                           The diffusion model will also scale the input tensor to the standard normal distribution
                           of the given training data upon calling the `fit` method.
        :param diff_range_filter: bool, Whether to filter the training data for exceeding the parameter range
                                  (any dimension larger than diff_range). Defaults to True.
        :param lambda_range: float, Magnitude of loss if denoised parameters exceed parameter range. Defaults to 0.
        :param epsilon: float, Small value to avoid division by zero in the standard scaler and fitness weight. Defaults to 1e-8.
        :param log_dir: str, Directory for logging the training process. Defaults to "". If no directory is specified,
                        no logging will be performed.
        :param scaler: (optional) `condevo.preprocessing.Scaler` instance for scaling input data (x space) into standardized
                       data for the diffusion model (z space). The DM learns to denoise in z space (by transforming x data
                       in `fit` [and in sample]), while generated samples are inversely transformed to x space upon calling `sample`.
                       Note: Scaler behavior is WIP.
        :param clip_gradients: bool or float, Whether to clip the gradients during training to avoid exploding gradients.
                               If float, the value is used as the max norm for clipping. Defaults to None (no clipping).
        :param cfg_scale: float, Classifier-free guidance scale for conditional generation. Defaults to 1.0 (no guidance).

        """
        super(DM, self).__init__()
        self.num_steps = num_steps
        self.nn = nn

        self.diff_range = diff_range
        self.diff_range_filter = diff_range_filter
        self.lambda_range = lambda_range

        self._scaler = None
        self.scaler = scaler

        self.epsilon = epsilon

        self.clip_gradients = clip_gradients

        self.cfg_scale = cfg_scale

        self._logger = None
        self._log_dir = log_dir

    def init_nn(self):
        """ Initialize the neural network model """
        if hasattr(self.nn, '_build_model'):
            backup_device = self.device
            self.nn._build_model()
            self.nn.to(backup_device)  # move model to the device

        else:
            raise NotImplementedError("Model does not have a `_build_model` method.")

    def to(self, device):
        super(DM, self).to(device)
        self.nn.to(device)
        return self

    def train(self, *args, **kwargs):
        self.nn.train(*args, **kwargs)
        return super(DM, self).train(*args, **kwargs)

    def eval(self):
        self.nn.eval()
        return super(DM, self).eval()

    def diffuse(self, x0, t):
        """ Abstract diffuse method the input tensor `x` at time `t` """
        raise NotImplementedError("Diffusion method not implemented, should be implemented in subclass.")

    def get_diffusion_time(self, noise_level, device=None):
        """ Get the diffusion time (ratio of denoising steps towards solution) for a given noise ratio. """
        if isinstance(noise_level, Tensor):
            return noise_level
        else:
            return torch.tensor(noise_level, device=device or self.device, dtype=torch.long)

    @property
    def scaler(self):
        return self._scaler

    @scaler.setter
    def scaler(self, value):
        if value is None:
            value = Scaler()

        if isinstance(value, str):
            # dynamically load from condevo.preprocessing
            import importlib

            # allow either "ClassName" (from condevo.preprocessing) or full module path "pkg.module.ClassName"
            if "." in value:
                mod_name, cls_name = value.rsplit(".", 1)
            else:
                mod_name, cls_name = "condevo.preprocessing", value

            try:
                mod = importlib.import_module(mod_name)
                cls = getattr(mod, cls_name)
            except (ModuleNotFoundError, AttributeError) as e:
                raise ImportError(f"Could not load scaler `{value}`: {e}")

            # instantiate if it's a class, otherwise keep as-is
            value = cls() if isinstance(cls, type) else cls

        if not isinstance(value, Scaler):
            raise TypeError(f"scaler must be a Scaler, got {type(value)}")

        self._scaler = value
        self._modules["scaler"] = value  # register submodule

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

    def get_null_conditions(self, *conditions):
        """ return null conditions for classifier-free guidance """
        if not len(conditions):
            return ()

        return tuple(torch.zeros_like(c) for c in conditions)

    def forward(self, x, t, *conditions):
        """Predict the noise `eps` with given `x` and `t`, `t` [0,1] represents the diffusion time step,
           where `t==0` is the data state (denoised) and `t==1` is the fully noisy state. """
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        t = t.to(x.dtype)

        if self.training or self.cfg_scale == 0.0 or len(conditions) == 0:
            return self.nn(x, t, *conditions)

        eps_c = self.nn(x, t, *conditions)

        null_conds = self.get_null_conditions(*conditions)
        eps_u = self.nn(x, t, *null_conds)

        w = self.cfg_scale
        return eps_u + w * (eps_c - eps_u)

    def sample_batch(self, xt, *conditions, t_start=0) -> Tensor:
        """" sample batch of data points, starting from xt ( xt.shape = (num_samples, dim) ) at time step t_start """
        raise NotImplementedError()

    def draw_random(self, num: int, *shape, dtype=None, device=None):
        """ Draw random samples from the standard normal distribution with given shape. """
        if dtype is None:
            dtype = next(self.nn.parameters()).dtype

        return randn(num, *shape, device=device or self.device, dtype=dtype)

    @no_grad()
    def sample(self, shape: tuple, num: int = None, x_source: Tensor = None, conditions=None, t_start=None, max_tries=5):
        r"""Sample `num` points from the diffusion model with given `shape` and `conditions`.
        If `num` is not specified, sample points from `x_source` tensor.

        Args:
            shape (tuple): Shape of the sampled points.
            num (int, optional): Number of points to be sampled. Defaults to None.
            x_source (torch.tensor, optional): Source tensor to sample points. Defaults to None.
            conditions (tuple, optional): Conditions for the diffusion model. Defaults to None.
            t_start (int, optional): Starting time step for the diffusion process. Defaults to None.
            max_tries (int, optional): Maximum number of resampling if the diffusio model generates bad (out of bounds)
                                       solutions. Only applicable if diff_range is set. Only resamples bad solutions.
                                       Defaults to 5.

        Returns:
            torch.tensor: Sampled points from the diffusion model.
        """
        if (num is None) and (x_source is None):
            raise ValueError("Either `num` or `x_source` should be specified")

        if (num is not None) and (x_source is not None):
            raise ValueError("Only one of `num` and `x_source` should be specified")

        if conditions is None:
            conditions = tuple()

        if num is not None:
            x_source = self.draw_random(num, *shape)

        elif x_source is not None:
            x_source = self.scaler.transform(x_source)
            # working in diffusion model z-space below

        self.eval()
        x_sampled = self.sample_batch(x_source, *conditions, t_start=t_start)

        # check for valid parameter range
        if self.diff_range not in [None, 0, 0.0]:
            bad = self.exceeds_diff_range(self.scaler.inverse_transform(x_sampled)) > 0      # check in x-space
            tries = 0

            while bad.any():
                bad_idx = bad.nonzero(as_tuple=False).squeeze(1)  # indices into the full batch
                B_bad = bad_idx.numel()

                # fallback: uniform genomes in x-space
                if tries >= max_tries:
                    # uniform in x-space in [-diff_range, diff_range]
                    x_fallback = (torch.rand(B_bad, *shape, device=x_sampled.device, dtype=x_sampled.dtype) * 2 - 1) * self.diff_range
                    x_sampled[bad_idx] = self.scaler.transform(x_fallback)
                    break

                # resample only bad indices
                resample_x_source = self.draw_random(B_bad, *shape, device=x_sampled.device, dtype=x_sampled.dtype)
                resample_conditions = [condition[bad_idx] for condition in conditions]
                x_resampled = self.sample_batch(resample_x_source, *resample_conditions)     # sample fresh

                # check resampled validity in x-space
                bad_resampled = self.exceeds_diff_range(self.scaler.inverse_transform(x_resampled)) > 0
                valid_resampled_idx = bad_idx[~bad_resampled]
                x_sampled[valid_resampled_idx] = x_resampled[~bad_resampled]

                bad = self.exceeds_diff_range(self.scaler.inverse_transform(x_sampled)) > 0  # check in x-space
                tries += 1

        x_sampled = self.scaler.inverse_transform(x_sampled)
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
        if self.diff_range in [None, 0, 0.0]:
            return zeros_like(x[:, 0], device=self.device)

        # evaluate the remainder of x that exceeds the parameter range (via ReLU)
        return ReLU()((x ** 2 - self.diff_range ** 2).reshape(x.shape[0], -1)).mean(dim=-1).sqrt()

    def diff_clamp(self, x, from_z_space=False):
        """ Clamp the input tensor `x` to the diffusion range if specified.

        If the `from_z_space` flag is set, x will be assumed to be scaled to the models z space.
        Thus, x will be transformed into x space via scaler.inverse_transform, and the clamped result will be
        transformed back into z-space using scaler.transform.
        """
        if self.diff_range in [None, 0, 0.0]:
            return x

        if from_z_space:
            x = self.scaler.inverse_transform(x)

        # clamp the input tensor to the diffusion range
        x_clamped = x.clamp(-self.diff_range, self.diff_range)

        if from_z_space:
            x_clamped = self.scaler.transform(x_clamped)

        return x_clamped

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
        :param scheduler: str or type, Scheduler for the optimizer. Defaults to None;
                          other choices are "cosine", "linear", "reduce_on_plateau", or a torch scheduler instance.

        :return: list, Loss history of the training process.
        """
        self.train()
        device = self.device

        # --- initialize optimizer ---
        if isinstance(optimizer, str):
            optimizer = getattr(optim, optimizer)

        optimizer = optimizer(self.parameters(), lr=lr, weight_decay=weight_decay)

        # --- learning rate scheduler ---
        if scheduler == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, max_epoch, eta_min=1e-6
            )
        elif scheduler == "linear":
            scheduler = optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1.0, end_factor=0.0, total_iters=max_epoch
            )
        elif scheduler == "reduce_on_plateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.9, patience=10, threshold=1e-4, min_lr=1e-6
            )

        # --- loss function ---
        loss_function = MSELoss(reduction='none')
        grad_clip_value = self.clip_gradients if not isinstance(self.clip_gradients, bool) else 1.

        # -- Scale data and clean according to defined condevo.preprocessign.Scaler ---
        x, weights, conditions = self.scaler.clean(x=x, weights=weights, conditions=conditions)

        if self.diff_range_filter:
            # filter out potential exceeding data
            exceeding = self.exceeds_diff_range(x) > 0
            if exceeding.any():
                x = x[~exceeding]
                weights = weights[~exceeding]
                conditions = tuple(c[~exceeding] for c in conditions)

        if x.shape[0] == 0:
            raise ValueError("All samples removed after NaN cleaning and/or diff_range filtering.")

        # transform data into z-space
        x, weights, conditions = self.scaler.fit_transform(x=x, weights=weights, conditions=conditions)

        dataset = TensorDataset(x, *conditions, weights)
        training_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # --- log dataset ---
        self.logger.next()
        # self.logger.log_dataset(x, weights, *conditions)

        # --- training loop ---
        loss_history, best_model, best_loss = [], None, torch.inf
        bar = tqdm(range(max_epoch), desc="Training Diffusion Model", unit="epoch")
        bar.set_postfix(loss=0.0)
        for epoch in bar:
            epoch_loss = 0
            num_updates = 0
            for x_batch, *c_batch, w_batch in training_dataloader:
                x_batch = x_batch.to(device)
                c_batch = [c.to(device) for c in c_batch]
                w_batch = w_batch.to(device)
                optimizer.zero_grad()

                v, v_pred = self.eval_val_pred(x_batch, *c_batch)
                loss = loss_function(v, v_pred) * w_batch
                reg_loss = self.regularize(x_batch, w_batch, *c_batch)
                loss = (loss + reg_loss).mean()

                if self.clip_gradients:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip_value)

                epoch_loss = epoch_loss + loss.item()

                loss.backward()
                optimizer.step()
                num_updates += 1

            self.logger.log_scalar(f"Loss/Train", epoch_loss, epoch)
            epoch_loss = epoch_loss/(num_updates or 1)  # avoid division by zero
            loss_history.append(epoch_loss)
            bar.set_postfix(loss=epoch_loss)
            if scheduler is not None:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(epoch_loss)
                else:
                    scheduler.step()
                current_lr = optimizer.param_groups[0]['lr']
                self.logger.log_scalar(f"LR", current_lr, epoch)

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model = deepcopy(self.nn.state_dict())

        # load best model
        if best_model is not None:
            self.nn.load_state_dict(best_model)

        self.logger.log_scalar(f"Loss/Generation", best_loss, self.logger.generation)
        self.eval()
        self.nn.eval()
        return loss_history
