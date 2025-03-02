from torch import ones, rand, randn_like
from condevo.diffusion import DM


class RectFlow(DM):
    """ Rectified Flow model for condevo package. """

    def __init__(self, nn, num_steps=100, param_range=None, lambda_range=0.):
        """ Initialize the RectFlow model

        :param nn: torch.nn.Module, Neural network to be used for the diffusion model.
        :param num_steps: int, Number of steps for the diffusion model. Defaults to 100.
        :param param_range: float, Parameter range for generated samples of the diffusion model. Defaults to None.
        :param lambda_range: float, Magnitude of loss if denoised parameters exceed parameter range. Defaults to 0.
        """
        super(RectFlow, self).__init__(nn=nn, num_steps=num_steps, param_range=param_range, lambda_range=lambda_range)

    def interpolate(self, x1, t):
        # Note: different from DDPM or DDIM, x1~data, and x0~noise
        x0 = randn_like(x1)
        xt = t * x1 + (1 - t) * x0
        return x0, xt

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
            v = self(xt, t, *conditions)
            xt = xt + v * (1 / self.num_steps)

        return xt

    def eval_val_pred(self, x, *conditions):
        """ Evaluate the actual error value and error prediction of the model """
        t = rand(x.shape[0], 1)
        x0, xt = self.interpolate(x, t)
        v_pred = self(xt, t, *conditions)
        v = x - x0
        return v, v_pred
