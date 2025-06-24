from torch import tensor, ones, rand, randn_like, cat, linspace, sqrt, cos, pi
from ..diffusion import DM


class DDIM(DM):
    """ DDIM: Denoising Diffusion Implicit Model for the `condevo` package. """

    ALPHA_SCHEDULES = ["linear", "cosine", ]

    def __init__(self, nn, num_steps=1000, skip_connection=True, noise_level=1.0,
                 diff_range=None, lambda_range=0., predict_eps_t=False, param_mean=0.0, param_std=1.0,
                 alpha_schedule="linear", matthew_factor=1.0, sample_uniform=False, autoscaling=False,
                 log_dir="", normalize_steps=False, diff_range_filter=True,
                 clip_gradients=None):
        """ Initialize the DDIM model

        :param nn: torch.nn.Module, Neural network to be used for the diffusion model.
        :param num_steps: int, Number of steps for the diffusion model. Defaults to 100.
        :param skip_connection: bool, Using skip connections for the diffusion model error estimate. Defaults to True.
        :param noise_level: float, Noise level for the diffusion model. Defaults to 1.0.
        :param diff_range: float, Parameter range for generated samples of the diffusion model. Defaults to None.
        :param diff_range_filter: bool, Whether to filter the training data for exceeding the parameter range
                                  (any dimension larger than diff_range). Defaults to True.
        :param lambda_range: float, Magnitude of loss if denoised parameters exceed parameter range. Defaults to 0.
        :param predict_eps_t: bool, Whether to predicting the noise `eps_t` for a given `xt` and `t`, or
                              or the total noise `eps` as if `t==T` for a given `xt`. Defaults to False (total noise).
        :param alpha_schedule: str, Schedule for the alpha parameter. Defaults to "linear".
        :param matthew_factor: float, Matthew factor for scaling the estimated error during sampling.
                               Defaults to 1.0, set to 0.8 for exploration.
        """
        # call the base class constructor, sets nn and num_steps attributes
        super(DDIM, self).__init__(nn=nn, num_steps=num_steps, diff_range=diff_range, lambda_range=lambda_range,
                                   param_mean=param_mean, param_std=param_std, sample_uniform=sample_uniform,
                                   autoscaling=autoscaling, log_dir=log_dir, diff_range_filter=diff_range_filter,
                                   clip_gradients=clip_gradients)
        self.skip_connection = skip_connection
        self.normalize_steps = normalize_steps

        # DDIM parameters
        self.alpha = None
        self.noise_level = noise_level
        self.predict_eps_t = predict_eps_t
        self._alpha_schedule = None
        self.alpha_schedule = alpha_schedule
        self.matthew_factor = matthew_factor

    @property
    def alpha_schedule(self):
        return self._alpha_schedule

    @alpha_schedule.setter
    def alpha_schedule(self, value):
        if value not in self.ALPHA_SCHEDULES:
            raise ValueError(f"Invalid alpha schedule: {value}. Must be one of {self.ALPHA_SCHEDULES}.")
        self._alpha_schedule = value

        if value == "linear":
            self.alpha = linspace(1 - 1 / self.num_steps, 1e-8, self.num_steps).to(self.device)

        elif value == "cosine":
            delta = 1e-3
            x = linspace(0, pi, self.num_steps)
            self.alpha = ((cos(x) * (1 - 2 * delta) + 1) / 2).to(self.device)

        else:
            raise NotImplementedError(f"Alpha schedule `{value}` not implemented.")

        a = cat([tensor([1], device=self.device), self.alpha])
        self.sigma = (1 - a[:-1]) / (1 - a[1:]) * (1 - a[1:] / a[:-1])
        self.sigma = sqrt(self.sigma)
    
    def forward(self, xt, t, *conditions):
        r"""Predicting the noise `eps` with given `xt` and `t`, where
        `xt` is x0 after diffusion and `t` is the time step. `t` is the
        time, which is in the range of [0, 1].
        """

        if self.normalize_steps:
            t = t / self.num_steps

        y = super().forward(xt.clone(), t, *conditions)
        if self.skip_connection:
            return y + xt
        else:
            return y

    def diffuse(self, x, t):
        """Diffuse the input tensor `x` with noise at time `t`
        
        Args:
            x (torch.tensor): Input tensor to be diffused.
            t (torch.tensor): Time step for the diffusion process, ranging from 0 to 1.

        Returns:
            tuple: Diffused tensor `xt` and the totoal noise `eps`.
                   In case of `self.predict_eps_t`, the returned noise is the actual noise `eps_t` at time `t`.
        """
        eps = randn_like(x)
        if self.autoscaling:
            # explore larger parameter space if necessary
            eps = eps * self.param_std

        if isinstance(t, float):
            t = tensor(t)
        T = (t * (self.num_steps - 1)).long()
        eps_t = (1 - self.alpha[T]).sqrt() * eps 
        xt = self.alpha[T].sqrt() * x + eps_t
        if self.predict_eps_t:
            eps_t = xt - x
            return xt, eps_t
        return xt, eps

    def sample_point(self, xt, *conditions, t_start=None):
        # sample one point from xt ( xt.shape = (dim) )
        if t_start is None:
            t_start = self.num_steps - 1  # avoid the first step, special alpha[T] value

        xt = xt.unsqueeze(0)
        conditions = tuple(c.unsqueeze(0) for c in conditions)
        one = ones(1, 1, device=self.device)

        for T in range(t_start - 1, 0, -1):
            t = one * T / self.num_steps
            s = self.sigma[T-1] * self.noise_level
            z = randn_like(xt)

            eps = self(xt, t, *conditions) * self.matthew_factor
            if self.predict_eps_t:
                x0_pred = xt - eps
            else:
                x0_pred = (xt - (1-self.alpha[T]).sqrt() * eps) / (self.alpha[T].sqrt())

            xt = self.alpha[T-1].sqrt() * x0_pred + (1 - self.alpha[T-1] - s ** 2).sqrt() * eps + s * z

        return xt.squeeze(0)

    def eval_val_pred(self, x, *conditions):
        t = rand(x.shape[0], device=self.device).reshape(-1, 1)
        xt, eps = self.diffuse(x, t)
        eps_pred = self.forward(xt, t, *conditions)
        return eps, eps_pred        

    def regularize(self, x_batch, w_batch, *c_batch):
        # regularize the denoising steps
        if self.diff_range is not None and self.lambda_range:
            # random time steps for the diffusion process
            t = rand(x_batch.shape[0]).reshape(-1, 1)
            T = (t * (self.num_steps - 1)).long()

            # apply diffusion and predict the noise
            xt, _ = self.diffuse(x_batch, t)
            eps_pred = self(xt, t, *c_batch)

            # recover the denoised parameters
            score = eps_pred / (self.alpha[T].sqrt())
            x0_direct = xt - score

            # return the regularization loss
            return self.lambda_range * self.exceeds_diff_range(x0_direct)[:, None]

        # default regularization
        return super(DDIM, self).regularize(x_batch, w_batch, *c_batch)
