from torch import ones, rand, randn_like, sqrt, arange
from ..diffusion import DM


class RectFlow(DM):
    """ Rectified Flow model for `condevo` package. """

    def __init__(self, nn, num_steps=100, diff_range=None, lambda_range=0., matthew_factor=0.8,
                 param_mean=0.0, param_std=1.0, autoscaling=True, sample_uniform=True, log_dir="", noise_level=0.0):
        """ Initialize the RectFlow model

        :param nn: torch.nn.Module, Neural network to be used for the diffusion model.
        :param num_steps: int, Number of steps for the diffusion model. Defaults to 100.
        :param diff_range: float, Parameter range for generated samples of the diffusion model. Defaults to None.
        :param lambda_range: float, Magnitude of loss if denoised parameters exceed parameter range. Defaults to 0.
        :param matthew_factor: float, Matthew factor for scaling the estimated error during sampling. Defaults to 0.8.
        :param param_mean: float, Initial mean of the parameters. Defaults to 0.0.
        :param param_std: float, Initial standard deviation of the parameters. Defaults to 1.0.
        :param autoscaling: bool, Whether to use autoscaling for the sampled parameters before denoising. Defaults to True.
        :param sample_uniform: bool, Whether to sample uniformly or not. Defaults to True.
        :param log_dir: str, Directory for tensorboard logging. Defaults to "". If no directory is specified,
                        no logging will be performed. (WIP)
        """
        super(RectFlow, self).__init__(nn=nn, num_steps=num_steps,
                                       diff_range=diff_range, lambda_range=lambda_range,
                                       param_mean=param_mean, param_std=param_std, autoscaling=autoscaling,
                                       sample_uniform=sample_uniform, log_dir=log_dir)
        self.matthew_factor = matthew_factor
        self.noise_level = noise_level

    def interpolate(self, x1, t):
        # Note: different from DDPM or DDIM, x1~data, and x0~noise
        x0 = self.draw_random(*x1.shape)
        xt = t * x1 + (1 - t) * x0
        return xt, x0

    def diffuse(self, x, t):
        """ Diffuse the input tensor `x` at time `t`, functionally equivalent to `self.interpolate(x, t)`
        but same naming as in DDIM and DDPM. """
        return self.interpolate(x, t)

    def sample_point(self, xt, *conditions, t_start=None):
        # Solving the ODE dx/dt = v(x, t) with Euler method
        if t_start is None:
            t_start = 0

        xt = xt.unsqueeze(0)
        conditions = tuple(c.unsqueeze(0) for c in conditions)
        tt = arange(0, self.num_steps, 1).reshape(-1, 1, 1) / self.num_steps  # add (batch, 1) to t
        dt = 1. / self.num_steps * self.matthew_factor
        noiset = np.sqrt(dt) * randn_like(xt) * (self.noise_level * torch.linspace(1, 0, self.num_steps).reshape(-1, 1, 1))

        for T in range(t_start, self.num_steps):
            t = tt[T]
            v = self(xt, t, *conditions)
            xt = xt + v * dt + noiset[T]

        return xt.squeeze(0)

    def eval_val_pred(self, x, *conditions):
        """ Evaluate the actual error value and error prediction of the model """
        t = rand(x.shape[0], 1)
        xt, x0 = self.diffuse(x, t)
        v_pred = self(xt, t, *conditions)
        v = x - x0
        return v, v_pred

    def regularize(self, x_batch, w_batch, *c_batch):
        # regularize the denoising steps
        if self.diff_range is not None and self.lambda_range:
            # random time steps for the diffusion process
            t = rand(x_batch.shape[0]).reshape(-1, 1)

            # apply diffusion and predict the noise
            xt, _ = self.diffuse(x_batch, t)
            v = self(xt, t, *c_batch)

            # recover the denoised parameters
            x0_direct = xt + v

            # return the regularization loss
            return self.lambda_range * self.exceeds_diff_range(x0_direct)[:, None]

        # default regularization
        return super(RectFlow, self).regularize(x_batch, w_batch, *c_batch)
