from torch import tensor, ones, rand, randn_like, cat, linspace, sqrt
from ..diffusion import DM


class DDIM(DM):
    """ DDIM: Denoising Diffusion Implicit Model """
    def __init__(self, nn, num_steps=1000, skip_connection=True, noise_level=1.0,
                 param_range=None, lambda_range=0., predict_eps_t=False, device='cpu'):
        """ Initialize the DDIM model

        :param nn: torch.nn.Module, Neural network to be used for the diffusion model.
        :param num_steps: int, Number of steps for the diffusion model. Defaults to 100.
        :param skip_connection: bool, Using skip connections for the diffusion model. Defaults to True.
        :param noise_level: float, Noise level for the diffusion model. Defaults to 1.0.
        :param param_range: float, Parameter range for generated samples of the diffusion model. Defaults to None.
        :param lambda_range: float, Magnitude of loss if denoised parameters exceed parameter range. Defaults to 0.
        :param predict_eps_t: bool, Whether to predicting the noise `eps_t` for a given `xt` and `t`, or
                              or the total noise `eps` as if `t==T` for a given `xt`. Defaults to False (total noise).
        """
        # call the base class constructor, sets nn and num_steps attributes
        super(DDIM, self).__init__(nn=nn, num_steps=num_steps, param_range=param_range, lambda_range=lambda_range)
        self.skip_connection = skip_connection

        # DDIM parameters
        self.alpha = None
        self.noise_level = noise_level
        self.predict_eps_t = predict_eps_t
        self._init_DDIM()

    def _init_DDIM(self):
        """ Initialize the DDIM parameters """
        # alpha[0] = 1, fully denoised; alpha[-1] = 0, fully noised
        self.alpha = linspace(1 - 1 / self.num_steps, 1e-8, self.num_steps, device=self.device)
        a = cat([tensor([1], device=self.device), self.alpha])
        self.sigma = (1 - a[:-1]) / (1 - a[1:]) * (1 - a[1:] / a[:-1])
        self.sigma = sqrt(self.sigma)
    
    def forward(self, xt, t, *conditions):
        r"""Predicting the noise `eps` with given `xt` and `t`, where
        `xt` is x0 after diffusion and `t` is the time step. `t` is the
        time, which is in the range of [0, 1].
        """
        y = super().forward(xt.clone(), t, *conditions)
        if self.skip_connection:
            return y + xt
        else:
            return y

    def diffuse(self, x0, t):
        """Diffuse the input tensor `x0` with noise at time `t`
        
        Args:
            x0 (torch.tensor): Input tensor to be diffused.
            t (torch.tensor): Time step for the diffusion process, ranging from 0 to 1.

        Returns:
            tuple: Diffused tensor `xt` and the totoal noise `eps`.
                   In case of `self.predict_eps_t`, the returned noise is the actual noise `eps_t` at time `t`.
        """
        eps = randn_like(x0, device=x0.device)
        if isinstance(t, float):
            t = tensor(t)
        T = (t * (self.num_steps - 1)).long().cpu()
        eps_t = (1 - self.alpha[T].to(x0.device)).sqrt() * eps
        xt = self.alpha[T].to(x0.device).sqrt() * x0 + eps_t
        if self.predict_eps_t:
            eps_t = xt - x0
            return xt, eps_t
        return xt, eps

    def sample_point(self, xt, *conditions, t_start=None):
        # sample one point from xt ( xt.shape = (dim) )
        if t_start is None:
            t_start = self.num_steps - 1  # avoid the first step, special alpha[T] value

        xt = xt.unsqueeze(0).to(self.alpha.device)
        conditions = tuple(c.unsqueeze(0) for c in conditions)
        one = ones(1, 1, device=xt.device)

        for T in range(t_start - 1, 0, -1):
            t = one * T / self.num_steps
            s = self.sigma[T] * self.noise_level
            z = randn_like(xt, device=xt.device)

            eps = self(xt, t, *conditions)
            x0_pred = (xt - (1-self.alpha[T]).sqrt() * eps) / self.alpha[T].sqrt()

            xt = self.alpha[T-1].sqrt() * x0_pred + (1 - self.alpha[T-1] - s ** 2).sqrt() * eps + s * z

        return xt.squeeze(0)

    def eval_val_pred(self, x, *conditions):
        t = rand(x.shape[0], device=x.device).reshape(-1, 1)
        xt, eps = self.diffuse(x, t)
        eps_pred = self.forward(xt, t, *conditions)
        return eps, eps_pred        

    def regularize(self, x_batch, w_batch, *c_batch):
        # regularize the denoising steps
        if self.param_range is not None:
            # random time steps for the diffusion process
            t = rand(x_batch.shape[0], device=x_batch.device).reshape(-1, 1)
            T = (t * (self.num_steps - 1)).long()

            # apply diffusion and predict the noise
            xt, _ = self.diffuse(x_batch, t)
            eps_pred = self(xt, t, *c_batch)

            # recover the denoised parameters
            x0_pred = (xt - (1 - self.alpha[T]).sqrt() * eps_pred) / self.alpha[T].sqrt()

            # return the regularization loss
            return self.lambda_range * self.exceeds_param_range(x0_pred)[:, None]

        # default regularization
        return super(DDIM, self).regularize(x_batch, w_batch, *c_batch)
