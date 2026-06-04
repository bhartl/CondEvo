from torch import Tensor, empty, no_grad
from typing import Optional, Tuple
from .scaler import Scaler


class MinMaxScaler(Scaler):
    """
    Per-feature min-max scaling:

        z = (x - x_min) / (x_max - x_min + eps)            -> [-1, 1]
        z = z * (hi - lo) + lo                             -> [lo, hi] if feature_range != (-1, 1)

    - features are along the LAST dimension
    - min/max stored as (1, D) buffers
    - weights and conditions are not transformed (only filtered during fit cleaning)
    - weights are ONLY used for alignment/masking (no weighted moments)
    """

    def __init__(self, eps: float = 1e-8, feature_range: Tuple[float, float] = (-1.0, 1.0)) -> None:
        super().__init__()
        self.eps = float(eps)

        lo, hi = feature_range
        if not (hi > lo):
            raise ValueError(f"feature_range must satisfy hi > lo; got {feature_range}.")
        self.lo = float(lo)
        self.hi = float(hi)

        self.register_buffer("min", empty(0), persistent=True)
        self.register_buffer("max", empty(0), persistent=True)

    @property
    def is_fitted(self) -> bool:
        return self.min.numel() != 0 and self.max.numel() != 0

    @no_grad()
    def fit(
        self,
        x: Tensor,
        weights: Optional[Tensor] = None,
        conditions: Tuple[Tensor, ...] = (),
    ) -> Tuple[Tensor, Optional[Tensor], Tuple[Tensor, ...]]:
        # Prepare + clean using base behavior (weights only for alignment/masking)
        x, weights, conditions = Scaler.fit(self, x, weights, conditions)

        if x.shape[-1] < 1:
            raise ValueError("Expected features in last dimension; got x.shape[-1] < 1.")

        D = x.shape[-1]
        x2 = x.reshape(-1, D)  # (N, D)

        x_min = x2.amin(dim=0)  # (D,)
        x_max = x2.amax(dim=0)  # (D,)

        self.min = x_min.reshape(1, D).to(device=x.device, dtype=x.dtype)
        self.max = x_max.reshape(1, D).to(device=x.device, dtype=x.dtype)

        return x, weights, conditions

    def transform(self, x: Tensor) -> Tensor:
        if not self.is_fitted:
            return x

        denom = (self.max - self.min).clamp_min(self.eps)
        z01 = (x - self.min) / denom  # [0,1] (approximately, may exceed if x outside fit range)

        return z01 * (self.hi - self.lo) + self.lo

    def inverse_transform(self, z: Tensor) -> Tensor:
        if not self.is_fitted:
            return z

        z01 = (z - self.lo) / (self.hi - self.lo)
        denom = (self.max - self.min).clamp_min(self.eps)
        return z01 * denom + self.min

    def get_spread(self):
        """
        For min-max scaling, a sensible 'spread' is the per-feature range.
        """
        if not self.is_fitted:
            return None
        return (self.max - self.min)
