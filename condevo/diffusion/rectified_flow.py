from torch import rand, randn_like, arange, linspace, tensor, Tensor, no_grad
from ..diffusion import DM


class RectFlow(DM):
    """ Rectified Flow model for `condevo` package. """

    def __init__(self, nn, num_steps=100, diff_range=None, lambda_range=0., matthew_factor=1.0,
                 scaler=None, log_dir="", noise_level=0.0, diff_range_filter=True, clip_gradients=False,
                 integration="Euler", **kwargs):
        """ Initialize the RectFlow model

        :param nn: torch.nn.Module, Neural network to be used for the diffusion model.
        :param num_steps: int, Number of steps for the diffusion model. Defaults to 100.
        :param diff_range: float, Parameter range for generated samples of the diffusion model. Defaults to None.
        :param diff_range_filter: bool, Whether to filter the training data for exceeding the parameter range
                                  (any dimension larger than diff_range). Defaults to True.
        :param lambda_range: float, Magnitude of loss if denoised parameters exceed parameter range. Defaults to 0.
        :param matthew_factor: float, Matthew factor for scaling the estimated error during sampling.
                               Defaults to 1.0, set to 0.8 for exploration.
        :param scaler: (optional) `condevo.preprocessing.Scaler` instance for scaling input data (x space) into standardized
                       data for the diffusion model (z space). The DM learns to denoise in z space (by transforming x data
                       in `fit` [and in sample]), while generated samples are inversely transformed to x space upon calling `sample`.
                       Note: Scaler behavior is WIP.
        :param log_dir: str, Directory for tensorboard logging. Defaults to "". If no directory is specified,
                        no logging will be performed. (WIP)
        :param noise_level: float, Noise level for the diffusion model. Defaults to 0.0 (no noise).
        :param clip_gradients: float|None, If not None, clip the gradients to this value. Defaults to None.
        :param kwargs: Additional arguments for the base class.
        """
        super(RectFlow, self).__init__(nn=nn, num_steps=num_steps,
                                       diff_range=diff_range, lambda_range=lambda_range,
                                       scaler=scaler, log_dir=log_dir,
                                       diff_range_filter=diff_range_filter,
                                       clip_gradients=clip_gradients, **kwargs)

        self.matthew_factor = matthew_factor
        self.noise_level = noise_level
        self.integration = integration

    def interpolate(self, x1, t):
        # Note: different from DDPM or DDIM, x1~data, and x0~noise
        x0 = self.draw_random(*x1.shape, device=x1.device, dtype=x1.dtype)
        xt = t * x1 + (1 - t) * x0
        return xt, x0

    def diffuse(self, x, t):
        """ Diffuse the input tensor `x` at time `t`, functionally equivalent to `self.interpolate(x, t)`
        but same naming as in DDIM and DDPM. """
        return self.interpolate(x, t)

    def get_diffusion_time(self, noise_level, device=None):
        """ Get the diffusion steps for a given noise ratio."""
        if isinstance(noise_level, Tensor):
            return 1 - noise_level
        else:
            return tensor(1 - noise_level, device=device or self.device)

    @no_grad()
    def sample_batch(self, xt, *conditions, t_start=None):
        """" sample batch of data points, starting from xt ( xt.shape = (num_samples, dim) ) at time step t_start

        This is done by solving the ODE dx/dt = v(x, t) with Euler method (or RK2, depending on `self.integration`)
        """
        if t_start is None:
            t_start = 0

        B = xt.shape[0]
        device = xt.device
        dtype = xt.dtype

        # Precompute contiguous time grid
        tt = linspace(0, 1, self.num_steps, device=device, dtype=dtype).view(-1, 1)  # (T, 1)
        dt = (1.0 / self.num_steps) * self.matthew_factor

        for T in range(t_start, self.num_steps):
            t = tt[T].expand(B, 1)  # (B, 1)

            v = self(xt, t, *conditions)

            if self.integration == "Euler":
                xt = xt + v * dt

            elif self.integration == "RK2":
                v1 = v
                x_mid = xt + v1 * dt
                t_mid = (t + dt).clamp(max=1.0)  # rough next-t
                v2 = self(x_mid, t_mid, *conditions)
                xt = xt + 0.5 * (v1 + v2) * dt

            else:
                raise ValueError(f"Unknown integration method: {self.integration}")

            # Optional stochasticity (SDE-like correction)
            if self.noise_level:
                xt += (dt ** 0.5) * randn_like(xt) * self.noise_level * (1 - t)

            # Optional range enforcement
            if self.diff_range_filter:
                xt = self.diff_clamp(xt, from_z_space=True)

        return xt

    def eval_val_pred(self, x, *conditions):
        """ Evaluate the actual error value and error prediction of the model """
        t = rand(x.shape[0], 1, device=x.device, dtype=x.dtype)
        xt, x0 = self.diffuse(x, t)
        v_pred = self(xt, t, *conditions)
        v = x - x0
        return v, v_pred

    def regularize(self, x_batch, w_batch, *c_batch):
        # regularize the denoising steps
        if self.diff_range is not None and self.lambda_range:
            # random time steps for the diffusion process
            t = rand(x_batch.shape[0], device=x_batch.device, dtype=x_batch.dtype).reshape(-1, 1)

            # apply diffusion and predict the noise
            xt, _ = self.diffuse(x_batch, t)
            v_pred = self(xt, t, *c_batch)

            # constrain predicted clean sample x1
            x1_direct = xt + (1 - t) * v_pred

            # return the regularization loss
            x1_direct = self.scaler.inverse_transform(x1_direct)
            return self.lambda_range * self.exceeds_diff_range(x1_direct)[:, None]

        # default regularization
        return super(RectFlow, self).regularize(x_batch, w_batch, *c_batch)
