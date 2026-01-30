from torch import rand, no_grad, full, randn_like
from ..diffusion import DDIM


class XPred(DDIM):
    """ X Prediction denoiser """

    def __init__(self, nn, num_steps=1000, noise_level=1.0,
                 diff_range=None, lambda_range=0., scaler=None,
                 alpha_schedule="linear", matthew_factor=1.0,
                 log_dir="", diff_range_filter=True,
                 clip_gradients=None, **kwargs):
        """ Initialize the X-Prediction model. [^1]

        [^1]: Tianhong Li, Kaiming He, "Back to Basics: Let Denoising Generative Models Denoise", https://arxiv.org/abs/2511.13720

        :param nn: torch.nn.Module, Neural network to be used for the diffusion model.
        :param num_steps: int, Number of steps for the diffusion model. Defaults to 100.
        :param noise_level: float, Noise level for the diffusion model. Defaults to 1.0.
        :param diff_range: float, Parameter range for generated samples of the diffusion model. Defaults to None.
        :param diff_range_filter: bool, Whether to filter the training data for exceeding the parameter range
                                  (any dimension larger than diff_range). Defaults to True.
        :param lambda_range: float, Magnitude of loss if denoised parameters exceed parameter range. Defaults to 0.
        :param alpha_schedule: str, Schedule for the alpha parameter. Defaults to "linear".
        :param matthew_factor: float, Matthew factor for scaling the estimated error during sampling.
                               Defaults to 1.0, set to 0.8 for exploration.
        :param scaler: (optional) `condevo.preprocessing.Scaler` instance for scaling input data (x space) into standardized
                       data for the diffusion model (z space). The DM learns to denoise in z space (by transforming x data
                       in `fit` [and in sample]), while generated samples are inversely transformed to x space upon calling `sample`.
                       Note: Scaler behavior is WIP.
        :param log_dir: str, Directory for tensorboard logging. Defaults to "". If no directory is specified,
                        no logging will be performed. (WIP)
        :param clip_gradients: float|None, If not None, clip the gradients to this value. Defaults to None.
        :param kwargs: Additional arguments for the base class.
        """
        # call the base class constructor, sets nn and num_steps attributes
        kwargs.setdefault("skip_connection", False)
        super().__init__(
            nn=nn,
            num_steps=num_steps,
            skip_connection=kwargs.pop("skip_connection", False),
            noise_level=noise_level,
            diff_range=diff_range,
            lambda_range=lambda_range,
            scaler=scaler,
            alpha_schedule=alpha_schedule,
            matthew_factor=matthew_factor,
            log_dir=log_dir,
            diff_range_filter=diff_range_filter,
            clip_gradients=clip_gradients,
            **kwargs
        )

    def forward(self, xt, t, *conditions):
        """Network outputs x0_pred directly (bypass DDIM.forward residual)."""
        # skip DDIM forward to avoid `skip_connection`, and call super of DDIM, i.e., DM model
        return super(DDIM, self).forward(xt.clone(), t, *conditions)

    def get_clamped_eps_x0(self, xt, model_prediction, T):
        """ clamp x0 and eps to `diff_range`, if `diff_range_filter` flag is set"""
        x0_pred = model_prediction
        if self.diff_range_filter:
            x0_pred = self.diff_clamp(x0_pred, from_z_space=True)

        eps = self.get_eps(xt, x0_pred, T)
        return eps, x0_pred

    def eval_val_pred(self, x, *conditions):
        t = rand(x.shape[0], device=x.device).reshape(-1, 1)
        xt, _eps = self.diffuse(x, t)
        x0_pred = self(xt, t, *conditions)
        return x, x0_pred

    def regularize(self, x_batch, w_batch, *c_batch):
        # regularize the denoising steps
        if self.diff_range is not None and self.lambda_range:
            # random time steps for the diffusion process
            t = rand(x_batch.shape[0], device=x_batch.device, dtype=x_batch.dtype).reshape(-1, 1)
            T = self._step_continuous_to_discrete(t)

            # apply diffusion and predict the noise
            xt, _ = self.diffuse(x_batch, t)
            x0_direct = self(xt, t, *c_batch)

            # return the regularization loss
            x0_direct = self.scaler.inverse_transform(x0_direct)  # transform from z to x space
            return self.lambda_range * self.exceeds_diff_range(x0_direct)[:, None]

        # default regularization
        return super(DDIM, self).regularize(x_batch, w_batch, *c_batch)

    @no_grad()
    def sample_batch(self, xt, *conditions, t_start=None):
        """" sample batch of data points, starting from xt ( xt.shape = (num_samples, dim) ) at time step t_start """
        if t_start is None:
            t_start = self.num_steps - 1

        B = xt.shape[0]
        device = xt.device
        dtype = xt.dtype

        for T in range(t_start, 0, -1):
            t = full((B, 1), self._step_discrete_to_continuous(T), device=device, dtype=dtype)

            a_prev = self.alpha[T-1]
            s = self.sigma[T] * self.noise_level
            z = randn_like(xt)

            model_prediction = self(xt, t, *conditions)
            eps, x0_pred = self.get_clamped_eps_x0(xt, model_prediction, T)
            eps = eps  * self.matthew_factor  # apply mathew factor only to eps (WIP, might be removed and unified with DDIM)

            eps_sqrt_term = (1 - a_prev - s ** 2).clamp_min(0).sqrt()
            xt = a_prev.sqrt() * x0_pred + eps_sqrt_term * eps + s * z

        return xt
