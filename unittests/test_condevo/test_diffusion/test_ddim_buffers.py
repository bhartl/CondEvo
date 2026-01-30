# Unit tests (unittest framework) for DDIM schedule buffers: alpha (used as alpha_bar) and sigma.
# Validates: no NaNs/Infs, values in valid ranges, monotonicity, and sigma formula consistency
# across schedules: linear, cosine, cosine_nichol.

import unittest
import torch
from torch import nn

from condevo.diffusion.ddim import DDIM


class _DummyNN(nn.Module):
    """Minimal network stub that satisfies DM/DDIM expectations."""
    def __init__(self):
        super().__init__()
        self.num_conditions = 0
        # Need at least one parameter so DM.device works
        self._p = nn.Parameter(torch.zeros(()))

    def forward(self, x, t, *conditions):
        return torch.zeros_like(x)


def _recompute_sigma_from_alpha_bar(alpha_bar: torch.Tensor) -> torch.Tensor:
    """
    Recompute sigma exactly as in DDIM.alpha_schedule setter:
      one = 1
      a = cat([one, alpha_bar])
      sigma = sqrt((1-a[:-1])/(1-a[1:]) * (1 - a[1:]/a[:-1]))
    """
    one = torch.tensor([1.0], device=alpha_bar.device, dtype=alpha_bar.dtype)
    a = torch.cat([one, alpha_bar])
    rad = (1 - a[:-1]) / (1 - a[1:]) * (1 - a[1:] / a[:-1])
    rad = rad.clamp_min(0.0)  # tolerate tiny negatives from float error
    return torch.sqrt(rad)


def _derive_alpha_step_from_alpha_bar(alpha_bar: torch.Tensor) -> torch.Tensor:
    """
    If alpha_bar is cumulative product, then per-step alpha_step[t] = alpha_bar[t] / alpha_bar[t-1],
    with alpha_bar[-1] defined as 1 for t=0.
    """
    one = torch.tensor([1.0], device=alpha_bar.device, dtype=alpha_bar.dtype)
    a_prev = torch.cat([one, alpha_bar[:-1]])
    return alpha_bar / a_prev


class TestDDIMSchedules(unittest.TestCase):
    SCHEDULES = ("linear", "cosine", "cosine_nichol")
    STEP_COUNTS = (32, 100, 1000)

    def setUp(self):
        torch.manual_seed(0)
        self.device = torch.device("cpu")

    def _make_model(self, schedule: str, num_steps: int) -> DDIM:
        model = DDIM(
            nn=_DummyNN(),
            num_steps=num_steps,
            alpha_schedule=schedule,
            diff_range=None,
            diff_range_filter=False,
            autoscaling=False,
            sample_uniform=False,
            clip_gradients=None,
        ).to(self.device)
        return model

    def _assert_all_finite(self, name: str, x: torch.Tensor):
        self.assertTrue(torch.isfinite(x).all().item(), f"{name} contains NaN/Inf")

    def _assert_in_open_interval(self, name: str, x: torch.Tensor, lo: float, hi: float):
        self.assertTrue((x > lo).all().item(), f"{name} has values <= {lo}")
        self.assertTrue((x < hi).all().item(), f"{name} has values >= {hi}")

    def _assert_monotone_nonincreasing(self, name: str, x: torch.Tensor, tol: float = 1e-12):
        dx = x[1:] - x[:-1]
        self.assertTrue((dx <= tol).all().item(), f"{name} is not monotone non-increasing")

    def test_schedule_buffers_finite_and_well_behaved(self):
        for schedule in self.SCHEDULES:
            for num_steps in self.STEP_COUNTS:
                with self.subTest(schedule=schedule, num_steps=num_steps):
                    model = self._make_model(schedule, num_steps)

                    self.assertTrue(hasattr(model, "alpha"), "DDIM must register buffer 'alpha'")
                    self.assertTrue(hasattr(model, "sigma"), "DDIM must register buffer 'sigma'")

                    alpha_bar = model.alpha
                    sigma = model.sigma

                    self.assertEqual(tuple(alpha_bar.shape), (num_steps,))
                    self.assertEqual(tuple(sigma.shape), (num_steps,))

                    # Finite checks
                    self._assert_all_finite("alpha_bar", alpha_bar)
                    self._assert_all_finite("sigma", sigma)

                    # Range checks
                    self._assert_in_open_interval("alpha_bar", alpha_bar, 0.0, 1.0)
                    self.assertTrue((sigma >= 0).all().item(), "sigma has negative values")

                    # Monotonicity of alpha_bar (intended by your schedules)
                    self._assert_monotone_nonincreasing("alpha_bar", alpha_bar)

                    # Derived per-step alpha_step sanity
                    alpha_step = _derive_alpha_step_from_alpha_bar(alpha_bar)
                    self._assert_all_finite("alpha_step", alpha_step)
                    self.assertTrue((alpha_step > 0).all().item(), "alpha_step has non-positive values")
                    self.assertTrue((alpha_step <= 1.0 + 1e-12).all().item(), "alpha_step has values > 1 (unexpected)")

                    # Sigma consistency with the formula used in the setter
                    sigma_ref = _recompute_sigma_from_alpha_bar(alpha_bar)
                    self._assert_all_finite("sigma_ref", sigma_ref)

                    # Assert close with conservative tolerances
                    max_diff = (sigma - sigma_ref).abs().max().item()
                    self.assertLessEqual(
                        max_diff, 1e-4,
                        f"sigma does not match recomputed sigma_ref (max_abs_diff={max_diff})"
                    )

                    # Guard against "diverging coefficients" (very loose bound; adjust if needed)
                    self.assertLess(sigma.max().item(), 10.0, f"sigma seems to diverge (max={sigma.max().item()})")

                    one_minus = 1.0 - alpha_bar
                    self._assert_all_finite("1-alpha_bar", one_minus)
                    self.assertTrue((one_minus > 0).all().item(), "1 - alpha_bar has non-positive values (unexpected)")

    def test_sigma_radicand_nonnegative(self):
        """Directly validate the sigma radicand is >= 0 up to numerical tolerance."""
        num_steps = 512
        for schedule in self.SCHEDULES:
            with self.subTest(schedule=schedule):
                model = self._make_model(schedule, num_steps)
                alpha_bar = model.alpha

                one = torch.tensor([1.0], device=alpha_bar.device, dtype=alpha_bar.dtype)
                a = torch.cat([one, alpha_bar])

                rad = (1 - a[:-1]) / (1 - a[1:]) * (1 - a[1:] / a[:-1])
                self._assert_all_finite("sigma_radicand", rad)

                min_rad = rad.min().item()
                self.assertGreaterEqual(min_rad, -1e-6, f"sigma radicand significantly negative: min={min_rad}")

    def test_schedule_endpoints_reasonable(self):
        """Loose endpoint sanity checks: alpha_bar starts high and ends small."""
        num_steps = 1000
        for schedule in self.SCHEDULES:
            with self.subTest(schedule=schedule):
                model = self._make_model(schedule, num_steps)
                alpha_bar = model.alpha
                self.assertGreater(alpha_bar[0].item(), 0.5, f"{schedule}: alpha_bar[0] unexpectedly low")
                self.assertLess(alpha_bar[-1].item(), 0.1, f"{schedule}: alpha_bar[-1] unexpectedly high")

    def test_cosine_schedule_is_clamped(self):
        """
        If you implemented clamping for 'cosine', ensure it's actually clamped away from 0/1.
        This will FAIL until you clamp cosine like you did for linear/cosine_nichol.
        """
        num_steps = 1000
        model = self._make_model("cosine", num_steps)
        alpha_bar = model.alpha
        self.assertGreater(alpha_bar.min().item(), 0.0, "cosine: alpha_bar hits 0 (should be clamped)")
        self.assertLess(alpha_bar.max().item(), 1.0, "cosine: alpha_bar hits 1 (should be clamped)")

    def test_cosine_nichol_schedule_is_clamped(self):
        """
        If you implemented clamping for 'cosine', ensure it's actually clamped away from 0/1.
        This will FAIL until you clamp cosine like you did for linear/cosine_nichol.
        """
        num_steps = 1000
        model = self._make_model("cosine_nichol", num_steps)
        alpha_bar = model.alpha
        self.assertGreater(alpha_bar.min().item(), 0.0, "cosine: alpha_bar hits 0 (should be clamped)")
        self.assertLess(alpha_bar.max().item(), 1.0, "cosine: alpha_bar hits 1 (should be clamped)")

    def test_linear_schedule_is_clamped(self):
        """
        If you implemented clamping for 'cosine', ensure it's actually clamped away from 0/1.
        This will FAIL until you clamp cosine like you did for linear/cosine_nichol.
        """
        num_steps = 1000
        model = self._make_model("linear", num_steps)
        alpha_bar = model.alpha
        self.assertGreater(alpha_bar.min().item(), 0.0, "linear: alpha_bar hits 0 (should be clamped)")
        self.assertLess(alpha_bar.max().item(), 1.0, "linear: alpha_bar hits 1 (should be clamped)")


if __name__ == "__main__":
    unittest.main()
