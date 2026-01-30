from torch import tensor, ones, rand, randn, randn_like, cat, linspace, sqrt, cos, pi, no_grad, full
from ..diffusion import DM


class DDIM(DM):
    """ DDIM: Denoising Diffusion Implicit Model for the `condevo` package. """

    def __init__(self, nn, num_steps=1000, skip_connection=True, noise_level=1.0,
                 diff_range=None, lambda_range=0., scaler=None,
                 alpha_schedule="linear", matthew_factor=1.0,
                 log_dir="", diff_range_filter=True,
                 clip_gradients=None, **kwargs):
        """ Initialize the DDIM model

        :param nn: torch.nn.Module, Neural network to be used for the diffusion model.
        :param num_steps: int, Number of steps for the diffusion model. Defaults to 100.
        :param skip_connection: bool, Using skip connections for the diffusion model error estimate. Defaults to True.
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
        super(DDIM, self).__init__(nn=nn, num_steps=num_steps, diff_range=diff_range, lambda_range=lambda_range,
                                   scaler=scaler, log_dir=log_dir, diff_range_filter=diff_range_filter,
                                   clip_gradients=clip_gradients, **kwargs)

        self.skip_connection = skip_connection

        # DDIM parameters
        self.noise_level = noise_level
        self._alpha_schedule = None
        self.alpha_schedule = alpha_schedule
        self.matthew_factor = matthew_factor

    @property
    def alpha_schedule(self):
        return self._alpha_schedule

    @alpha_schedule.setter
    def alpha_schedule(self, value):
        if value == "linear":
            alpha = linspace(1 - 1 / self.num_steps, 1e-8, self.num_steps)
            alpha = alpha.clamp_min(1e-8)

        elif value == "cosine":
            delta = 1e-3
            x = linspace(0, pi, self.num_steps)
            alpha = ((cos(x) * (1 - 2 * delta) + 1) / 2)
            alpha = alpha.clamp(1e-6, 1.0 - 1e-6)

        elif value == "cosine_nichol":
            # Nichol & Dhariwal 2021 cosine schedule for alpha_bar
            s = 0.008
            i = linspace(0, self.num_steps, self.num_steps)  # 0..T
            f = ((i / self.num_steps + s) / (1 + s)) * (pi / 2)
            alpha_bar = (cos(f) ** 2) / (cos(tensor(s / (1 + s) * pi / 2)) ** 2)
            alpha = alpha_bar.clamp(1e-6, 1.0 - 1e-6)

        else:
            raise NotImplementedError(f"Alpha schedule `{value}` not implemented.")

        self._alpha_schedule = value

        one = tensor([1], device=alpha.device, dtype=alpha.dtype)
        a = cat([one, alpha])
        sigma = sqrt((1 - a[:-1]) / (1 - a[1:]) * (1 - a[1:] / a[:-1]).clamp_min(0.0))

        # register buffers
        self.register_buffer("alpha", alpha, persistent=False)
        self.register_buffer("sigma", sigma, persistent=False)

    def _step_continuous_to_discrete(self, t):
        # convert continuous time t in [0, 1] to discrete time T in [0, num_steps-1]
        T = (t * (self.num_steps - 1)).long()
        return T

    def _step_discrete_to_continuous(self, T):
        # convert discrete time T in [0, num_steps-1] to continuous time t in [0, 1]
        t = T / (self.num_steps - 1)
        return t

    def forward(self, xt, t, *conditions):
        r"""Predicting the noise `eps` with given `xt` and `t`, where
        `xt` is x0 after diffusion and `t` is the time step. `t` is the
        time, which is in the range of [0, 1].
        """
        y = super().forward(xt.clone(), t, *conditions)
        if self.skip_connection:
            return y + xt
        return y

    def diffuse(self, x, t):
        """Diffuse the input tensor `x` with noise at time `t`

        Args:
            x (torch.tensor): Input tensor to be diffused.
            t (torch.tensor): Time step for the diffusion process, ranging from 0 to 1.

        Returns:
            tuple: Diffused tensor `xt` and the total noise `eps`.
        """
        eps = self.draw_random(*x.shape, dtype=x.dtype, device=x.device)
        if isinstance(t, float):
            t = tensor(t, device=x.device, dtype=x.dtype)
        T = self._step_continuous_to_discrete(t)
        eps_t = (1 - self.alpha[T]).sqrt() * eps
        xt = self.alpha[T].sqrt() * x + eps_t
        return xt, eps  # return xt and total noise eps for x0 prediction

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

            model_prediction = self(xt, t, *conditions) * self.matthew_factor
            eps, x0_pred = self.get_clamped_eps_x0(xt, model_prediction, T)

            eps_sqrt_term = (1 - a_prev - s ** 2).clamp_min(0).sqrt()
            xt = a_prev.sqrt() * x0_pred + eps_sqrt_term * eps + s * z

        return xt

    def get_clamped_eps_x0(self, xt, model_prediction, T):
        eps = model_prediction
        x0 = self.get_x0(xt, eps, T)
        if not self.diff_range_filter:
            return eps, x0

        x0_clipped = self.diff_clamp(x0, from_z_space=True)
        eps_clipped = self.get_eps(xt, x0_clipped, T)
        return eps_clipped, x0_clipped

    def get_x0(self, xt, eps, T):
        return (xt - (1 - self.alpha[T]).sqrt() * eps) / (self.alpha[T].sqrt())

    def get_eps(self, xt, x0, T):
        return (xt - self.alpha[T].sqrt() * x0) / ((1 - self.alpha[T]).sqrt())

    def eval_val_pred(self, x, *conditions):
        t = rand(x.shape[0], device=x.device).reshape(-1, 1)
        xt, eps = self.diffuse(x, t)
        eps_pred = self(xt, t, *conditions)
        return eps, eps_pred

    def regularize(self, x_batch, w_batch, *c_batch):
        # regularize the denoising steps
        if self.diff_range is not None and self.lambda_range:
            # random time steps for the diffusion process
            t = rand(x_batch.shape[0], device=x_batch.device, dtype=x_batch.dtype).reshape(-1, 1)
            T = self._step_continuous_to_discrete(t)

            # apply diffusion and predict the noise
            xt, _ = self.diffuse(x_batch, t)
            eps_pred = self(xt, t, *c_batch)

            # recover the denoised parameters
            x0_direct = self.get_x0(xt, eps_pred, T)

            # return the regularization loss
            x0_direct = self.scaler.inverse_transform(x0_direct)  # transform from z to x space
            return self.lambda_range * self.exceeds_diff_range(x0_direct)[:, None]

        # default regularization
        return super(DDIM, self).regularize(x_batch, w_batch, *c_batch)
