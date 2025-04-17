from torch import ones, rand, randn_like, sqrt
from condevo.diffusion import DM
import numpy as np


class RectFlow(DM):
    """ Rectified Flow model for `condevo` package. """

    def __init__(self, nn, num_steps=100, diff_range=None, lambda_range=0., matthew_factor=np.sqrt(0.5),
                 param_mean=0.0, param_std=1.0):
        """ Initialize the RectFlow model

        :param nn: torch.nn.Module, Neural network to be used for the diffusion model.
        :param num_steps: int, Number of steps for the diffusion model. Defaults to 100.
        :param diff_range: float, Parameter range for generated samples of the diffusion model. Defaults to None.
        :param lambda_range: float, Magnitude of loss if denoised parameters exceed parameter range. Defaults to 0.
        :param matthew_factor: float, Matthew factor for scaling the estimated error during sampling. Defaults to 0.5.
        """
        super(RectFlow, self).__init__(nn=nn, num_steps=num_steps,
                                       diff_range=diff_range, lambda_range=lambda_range,
                                       param_mean=param_mean, param_std=param_std)
        self.matthew_factor = matthew_factor

    def interpolate(self, x1, t):
        # Note: different from DDPM or DDIM, x1~data, and x0~noise
        x0 = randn_like(x1)
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

        for T in range(t_start, self.num_steps):
            t = ones(1) * T / self.num_steps
            v = self(xt, t, *conditions) * self.matthew_factor
            xt = xt + v * (1 / self.num_steps)

        return xt

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
