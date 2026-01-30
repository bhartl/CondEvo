from torch import Tensor, empty, no_grad
from typing import Optional, Tuple, Union
from .scaler import Scaler


class StandardScaler(Scaler):
    """
    Per-feature standardization: z = (x - mean) / std

    - features are along the LAST dimension
    - mean/std stored as (1, D) buffers
    - weights and conditions are not transformed (only filtered during fit cleaning)
    - weights are ONLY used for alignment/masking (no weighted moments)
    """

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = float(eps)

        self.register_buffer("mean", empty(0), persistent=True)
        self.register_buffer("std", empty(0), persistent=True)

    @property
    def is_fitted(self) -> bool:
        return self.mean.numel() != 0

    @no_grad()
    def fit(self, x: Tensor, weights: Optional[Tensor] = None, conditions: Tuple[Tensor, ...] = ()) -> Tuple[Tensor, Tensor, Tuple[Tensor, ...]]:
        # Prepare + clean using base behavior (weights only for alignment/masking)
        x, weights, conditions = Scaler.fit(self, x, weights, conditions)

        if x.shape[-1] < 1:
            raise ValueError("Expected features in last dimension; got x.shape[-1] < 1.")

        # Per-feature moments over all non-feature dims, flatten everything except feature dim
        D = x.shape[-1]
        x2 = x.reshape(-1, D)  # (N, D)

        mean = x2.mean(dim=0)  # (D,)
        var = x2.var(dim=0, unbiased=False)  # (D,)
        std = (var + self.eps).sqrt()

        self.mean = mean.reshape(1, D).to(device=x.device, dtype=x.dtype)
        self.std = std.reshape(1, D).to(device=x.device, dtype=x.dtype)

        return x, weights, conditions

    def transform(self, x: Tensor) -> Tensor:
        if not self.is_fitted:
            return x
        return (x - self.mean) / self.std

    def inverse_transform(self, z: Tensor) -> Tensor:
        if not self.is_fitted:
            return z
        return z * self.std + self.mean

    def get_spread(self):
        if not self.is_fitted:
            return None

        return self.std
