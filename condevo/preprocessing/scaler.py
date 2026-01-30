from torch import Tensor, ones, no_grad, is_floating_point, isnan
from typing import Optional, Tuple, Union
from torch.nn import Module


class Scaler(Module):
    """
    Base scaler interface (identity by default).

    - fit() cleans data by dropping samples with NaNs in x or weights, filtering conditions accordingly
    - Does NOT change conditions (only filters rows)
    """

    def __init__(self) -> None:
        super().__init__()

    @no_grad()
    def clean(self, x: Tensor, weights: Optional[Tensor] = None, conditions: Tuple[Tensor, ...] = ()) -> Tuple[Tensor, Tensor, Tuple[Tensor, ...]]:
        x, weights, conditions = self._prepare_fit_inputs(x, weights, conditions)
        x, weights, conditions = self._drop_nan_x(x, weights, conditions)
        x, weights, conditions = self._drop_nan_weights(x, weights, conditions)
        return x, weights, conditions

    @no_grad()
    def fit(self, x: Tensor, weights: Optional[Tensor] = None, conditions: Tuple[Tensor, ...] = ()) -> Tuple[Tensor, Tensor, Tuple[Tensor, ...]]:
        """
        Identity fit, but performs cleaning + alignment:
        1) ensure weights exist, defaults to ones
        2) drop samples with NaNs in x
        3) drop samples with NaNs in weights
        Returns cleaned (x, weights, conditions)
        """
        return self.clean(x, weights=weights, conditions=conditions)

    def transform(self, x: Tensor) -> Tensor:
        return x

    def inverse_transform(self, z: Tensor) -> Tensor:
        return z

    def fit_transform(self, x: Tensor, weights: Optional[Tensor] = None, conditions: Tuple[Tensor, ...] = ()) -> Tuple[Tensor, Tensor, Tuple[Tensor, ...]]:
        x, weights, conditions = self.fit(x, weights=weights, conditions=conditions)
        return self.transform(x), weights, conditions

    def _prepare_fit_inputs(self, x: Tensor, weights: Optional[Tensor], conditions: Tuple[Tensor, ...]) -> Tuple[Tensor, Tensor, Tuple[Tensor, ...]]:
        if x.shape[0] < 1:
            raise ValueError("Expected x.shape[0] >= 1.")

        conditions = tuple(conditions) if conditions is not None else tuple()

        # Default weights: ones shaped like (B, 1, 1, ..., 1) matching x.ndim-1 trailing dims
        if weights is None:
            weights = ones(x.shape[0], *(1 for _ in range(x.dim() - 1)), device=x.device, dtype=x.dtype)

        else:
            weights = weights
            if not is_floating_point(weights):
                weights = weights.to(dtype=x.dtype)

            if weights.device != x.device:
                weights = weights.to(device=x.device)

        # Basic alignment checks (broadcast allowed, but batch dim must match)
        if weights.shape[0] != x.shape[0]:
            raise ValueError(f"weights.shape[0] must match x.shape[0]; got {weights.shape[0]} vs {x.shape[0]}")

        for c in conditions:
            if c.shape[0] != x.shape[0]:
                raise ValueError(f"condition batch dim must match x.shape[0]; got {c.shape[0]} vs {x.shape[0]}")

        return x, weights, conditions

    def _reduce_to_batch_mask(self, mask: Tensor) -> Tensor:
        """
        Reduce an elementwise mask over non-batch dims to a (B,) mask.
        """
        if mask.dim() == 1:
            return mask

        # reduce all dims except batch (dim 0)
        for _ in range(mask.dim() - 1):
            mask = mask.any(dim=-1)

        return mask  # (B,)

    def _drop_nan_x(self, x: Tensor, weights: Tensor, conditions: Tuple[Tensor, ...]) -> Tuple[Tensor, Tensor, Tuple[Tensor, ...]]:
        """
        Drop samples where x contains any NaNs (in any feature / trailing dims).
        """
        nan_mask = isnan(x)
        if nan_mask.any():
            nan_b = self._reduce_to_batch_mask(nan_mask)  # (B,)
            keep = ~nan_b
            x = x[keep]
            weights = weights[keep]
            conditions = tuple(c[keep] for c in conditions)

        return x, weights, conditions

    def _drop_nan_weights(self, x: Tensor, weights: Tensor, conditions: Tuple[Tensor, ...]) -> Tuple[Tensor, Tensor, Tuple[Tensor, ...]]:
        """
        Drop samples where weights contain NaNs (anywhere in trailing dims).
        """
        nan_mask = isnan(weights)
        if nan_mask.any():
            nan_b = self._reduce_to_batch_mask(nan_mask)  # (B,)
            keep = ~nan_b
            x = x[keep]
            weights = weights[keep]
            conditions = tuple(c[keep] for c in conditions)

        return x, weights, conditions

    def get_spread(self):
        return None
