from torch import cat, optim, ones, zeros_like, Tensor, no_grad, vmap, randn
from torch.nn import MSELoss, Module, ReLU
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from typing import Union


class DM(Module):
    """ Diffusion Model base-class for condevo package. """

    def __init__(self, nn, num_steps=100, param_range=None, lambda_range=0.):
        """ Initialize the Diffusion Model """
        super(DM, self).__init__()
        self.num_steps = num_steps
        self.nn = nn
        self.param_range = param_range
        self.lambda_range = lambda_range

    def init_nn(self):
        """ Initialize the neural network model """
        if hasattr(self.nn, '_build_model'):
            self.nn._build_model()

        else:
            raise NotImplementedError("Model does not have a `_build_model` method.")

    def forward(self, x, t, *conditions):
        """Predict the noise `eps` with given `x` and `t`, `t` [0,1] represents the diffusion time step,
           where `t==0` is the data state (denoised) and `t==1` is the fully noisy state. """
        return self.nn(cat([x, t, *conditions], dim=-1))

    def sample_point(self, xt, *conditions, t_start=0):
        pass

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

        if (x_source is None) and (num is not None):
            x_source = randn(num, *shape)
        
        if conditions is None:
            conditions = tuple()

        sample_vectorized = vmap(self.sample_point, randomness='different')
        x_sampled = sample_vectorized(x_source, *conditions, t_start=t_start)
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
    
    def exceeds_param_range(self, x):
        """ If the parameter range is specified, evaluate the exceed of the parameter range (via ReLU),
            otherwise return zeros tensor. """
        if self.param_range is None:
            return zeros_like(x[:, 0])
        
        # evaluate the exceed of the parameter range (via ReLU)
        return ReLU()((x ** 2).sum(dim=-1) - self.param_range ** 2)

    def fit(self, x, *conditions, weights=None, optimizer: Union[str, type] = optim.Adam, max_epoch=100, lr=1e-3,
            weight_decay=1e-5, batch_size=32, scheduler=None):
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

        if isinstance(optimizer, str):
            optimizer = getattr(optim, optimizer)

        optimizer = optimizer(self.parameters(), lr=lr, weight_decay=weight_decay)
        if scheduler == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, max_epoch, eta_min=1e-6)

        loss_function = MSELoss(reduction='none')

        if weights is None:
            weights = ones(x.shape[0], *(1 for _ in range(len(x.shape[1:]))), device=x.device)

        exceeding_x = self.exceeds_param_range(x) > 0
        if exceeding_x.any():
            x = x[~exceeding_x]
            conditions = [c[~exceeding_x] for c in conditions]
            weights = weights[~exceeding_x]
        
        dataset = TensorDataset(x, *conditions, weights)

        training_dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
        )

        loss_history = []
        for _ in tqdm(range(int(max_epoch))):
            batch_loss = 0
            for x_batch, *c_batch, w_batch in training_dataloader:
                w_batch = w_batch.to(x_batch.device)
                optimizer.zero_grad()
                v, v_pred = self.eval_val_pred(x_batch, *c_batch)
                loss = loss_function(v, v_pred) * w_batch
                reg_loss = self.regularize(x_batch, w_batch, *c_batch)
                loss = (loss + reg_loss).mean()
                loss.backward()
                batch_loss = batch_loss + loss.item()
                optimizer.step()
            loss_history.append(batch_loss)
            if scheduler is not None:
                scheduler.step()

        self.eval()
        return loss_history
