import torch
from torch import sqrt
from .ddim import DDIM

class VPred(DDIM):
    """DDIM with v-prediction target (Nichol & Dhariwal).

        This variant implements the *v-parameterization* introduced in [^1]:

        [1]: Nichol, A. Q., & Dhariwal, P. (2021). *Improved Denoising Diffusion Probabilistic Models*. arXiv:2102.09672

        Instead of predicting the noise ε directly, the network is trained to predict the auxiliary variable

            v_t = sqrt(alpha_bar_t) * ε - sqrt(1 - alpha_bar_t) * x_0 ,

        where alpha_bar_t (here alpha) is the cumulative noise schedule, x_0 is the clean data,
        and ε is standard Gaussian noise.

        This parameterization has improved numerical stability and balances
        gradient magnitudes across noise levels, especially near the low-noise
        (late denoising) regime.

        The implementation minimally overrides the base DDIM behavior by:
          - constructing v-targets during training,
          - reconstructing x_0 from (x_t, v),
          - recovering ε when needed for DDIM sampling updates.

        All scheduling, sampling logic, and classifier-free guidance behavior
        are inherited unchanged from the parent DDIM class.
        """

    def get_x0(self, xt, v_pred, T):
        """(x_t, v) -> x0

        With alpha_bar = self.alpha[T]:
          x0 = sqrt(alpha_bar) * x_t - sqrt(1 - alpha_bar) * v
        """
        a = self.alpha[T]
        return sqrt(a) * xt - sqrt(1.0 - a) * v_pred

    def get_eps(self, xt, x0, T):
        """(x_t, x0) -> eps

        eps = (x_t - sqrt(alpha_bar) * x0) / sqrt(1 - alpha_bar)
        """
        a = self.alpha[T]
        return (xt - sqrt(a) * x0) / sqrt(1.0 - a)

    def eval_val_pred(self, x, *conditions):
        """Return (v_target, v_pred) for loss computation.

        Using:
          v = sqrt(alpha_bar) * eps - sqrt(1 - alpha_bar) * x0
        where x0 == x (clean data).
        """
        t = torch.rand(x.shape[0], device=self.device, dtype=x.dtype).reshape(-1, 1)
        xt, eps = self.diffuse(x, t)
        T = (t * (self.num_steps - 1)).long()

        a = self.alpha[T]
        v_target = sqrt(a) * eps - sqrt(1.0 - a) * x

        v_pred = self(xt, t, *conditions)  # IMPORTANT: keep DM.forward behavior (CFG etc.)
        return v_target, v_pred

    def regularize(self, x_batch, w_batch, *c_batch):
        """Range-regularization using x0 reconstructed from v_pred."""
        if self.diff_range is not None and self.lambda_range:
            t = torch.rand(x_batch.shape[0], device=self.device, dtype=x_batch.dtype).reshape(-1, 1)
            T = (t * (self.num_steps - 1)).long()

            xt, _ = self.diffuse(x_batch, t)
            v_pred = self(xt, t, *c_batch)

            x0_direct = self.get_predicted_x0(xt, v_pred, T)
            x0_direct = self.scaler.inverse_transform(x0_direct)
            return self.lambda_range * self.exceeds_diff_range(x0_direct)[:, None]

        return super().regularize(x_batch, w_batch, *c_batch)
