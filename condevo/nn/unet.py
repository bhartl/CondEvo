from typing import List, Sequence, Type, Union
from torch import Tensor, cat
from torch.nn import Module, Identity, Linear
import torch.nn as tnn


def _resolve_activation(foo: Union[Type, str]):
    if isinstance(foo, str):
        return getattr(tnn, foo)
    return foo


def _linear_block(
    in_features: int,
    out_features: int,
    *,
    activation: Union[Type, str] = tnn.ReLU,
    batch_norm: bool = False,
    layer_norm: bool = False,
    dropout: float = 0.0,
):
    if batch_norm and layer_norm:
        raise ValueError("Choose at most one normalization: batch_norm or layer_norm.")

    Act = _resolve_activation(activation)
    layers: List[tnn.Module] = [tnn.Linear(in_features, out_features)]

    if batch_norm:
        layers.append(tnn.BatchNorm1d(out_features))
    if layer_norm:
        layers.append(tnn.LayerNorm(out_features))

    layers.append(Act())

    if dropout > 0:
        layers.append(tnn.Dropout(dropout))

    return tnn.Sequential(*layers)


class UNet(Module):
    """
    A fully-connected (dense) U-Net that mirrors the MLP interface.

    You specify the encoder widths via `channels_down` (e.g., [128, 64, 32]).
    The decoder is built in reverse order with skip connections by concatenation.

    Inputs/outputs match the MLP: given x (B, num_params), time t (B, 1), and optional
    condition tensors, we concatenate them along the last dimension and predict (B, num_params).

    Args:
        num_params (int): Number of parameters for x and the output.
        num_hidden (Sequence[int]): Hidden sizes for the encoder path. Depth = len(list).
        activation (Type|str): Activation class or its name (e.g., ReLU, "GELU").
        last_activation (Type|str): Activation for the final layer (default Identity).
        num_conditions (int): Number of conditioning features (besides time).
        batch_norm (bool): If True, use BatchNorm1d in blocks.
        layer_norm (bool): If True, use LayerNorm in blocks (mutually exclusive with batch_norm).
        dropout (float): Dropout probability inside blocks.
        time_embedding (int): Size of time embedding. Defaults to 0 (no embedding).
    """

    def __init__(
        self,
        num_params: int = 2,
        num_hidden: Sequence[int] = (128, 64, 32),
        activation: Union[Type, str] = tnn.ReLU,
        last_activation: Union[Type, str] = Identity,
        num_conditions: int = 0,
        batch_norm: bool = False,
        layer_norm: bool = False,
        dropout: float = 0.0,
        time_embedding: int = 0,
    ) -> None:
        super().__init__()
        if len(num_hidden) < 1:
            raise ValueError("channels_down must contain at least one layer width.")
        if batch_norm and layer_norm:
            raise ValueError("batch_norm and layer_norm are mutually exclusive.")

        self.num_params = num_params
        self.num_hidden = list(num_hidden)
        self.activation = activation
        self.last_activation = last_activation
        self.num_conditions = num_conditions
        self.batch_norm = batch_norm
        self.layer_norm = layer_norm
        self.dropout = dropout
        self.time_embedding = time_embedding

        self.encoder: tnn.ModuleList
        self.decoder: tnn.ModuleList
        self.out: tnn.Sequential
        self.time_embed: Union[None, Module]
        self._build_model()

    def _build_model(self):
        h, foo, foo_l = self.num_hidden, self.activation, self.last_activation
        if isinstance(foo, str):
            foo = getattr(tnn, foo)
        if isinstance(foo_l, str):
            foo_l = getattr(tnn, foo_l)

        time_dim = self.time_embedding if self.time_embedding > 0 else 1
        input_len = self.num_params + time_dim + self.num_conditions  # x, t, conditions
        output_len = self.num_params

        # --- Encoder ---
        enc_layers: List[Module] = []
        prev = input_len
        for width in self.num_hidden:
            enc_layers.append(
                _linear_block(
                    prev,
                    width,
                    activation=foo,
                    batch_norm=self.batch_norm,
                    layer_norm=self.layer_norm,
                    dropout=self.dropout,
                )
            )
            prev = width
        self.encoder = tnn.ModuleList(enc_layers)

        # --- Decoder (reverse order) ---
        # Current feature size after encoder
        current = self.num_hidden[-1]
        dec_layers: List[Module] = []
        # skip widths are encoder outputs in forward order
        skip_widths = self.num_hidden[:-1]
        for skip_w in reversed(skip_widths):
            # concatenate current with corresponding skip features
            dec_layers.append(
                _linear_block(
                    current + skip_w,
                    skip_w,  # reduce back to skip width
                    activation=foo,
                    batch_norm=self.batch_norm,
                    layer_norm=self.layer_norm,
                    dropout=self.dropout,
                )
            )
            current = skip_w
        self.decoder = tnn.ModuleList(dec_layers)

        # --- Output head ---
        self.out = tnn.Sequential(
            tnn.Linear(current, output_len),
            foo_l(),
        )

        if self.time_embedding > 0:
            from .embeddings import SinusoidalTimeEmbedding
            self.time_embed = SinusoidalTimeEmbedding(self.time_embedding)

        return self.encoder, self.decoder, self.out

    def forward(self, x: Tensor, t: Tensor, *conditions: Tensor) -> Tensor:
        """Forward pass.

        Expects t with shape (B, 1), mirroring the original MLP behavior.
        All `conditions` are concatenated as provided along the last dimension.
        """
        if self.time_embedding > 0:
            t = self.time_embed(t.flatten())

        x = cat([x, t, *conditions], dim=-1)

        # Encoder, keep skip features
        skips: List[Tensor] = []
        h = x
        for layer in self.encoder:
            h = layer(h)
            skips.append(h)

        # Decoder: mirror order and concat with corresponding skips
        for layer, skip in zip(self.decoder, reversed(skips[:-1])):
            h = cat([h, skip], dim=-1)
            h = layer(h)

        return self.out(h)

    def __repr__(self):
        return (
            f"UNet(num_params={self.num_params}, channels_down={self.num_hidden}, "
            f"activation={self.activation}, last_activation={self.last_activation}, "
            f"num_conditions={self.num_conditions}, batch_norm={self.batch_norm}, "
            f"layer_norm={self.layer_norm}, dropout={self.dropout})"
        )


def _init_ddim_dense_unet(m: Module):
    # 1) Default for all Linear layers
    if isinstance(m, Linear):
        tnn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            tnn.init.zeros_(m.bias)

    # 2) Norm layers
    if isinstance(m, (tnn.LayerNorm, tnn.BatchNorm1d)):
        if getattr(m, 'weight', None) is not None:
            tnn.init.ones_(m.weight)
        if getattr(m, 'bias', None) is not None:
            tnn.init.zeros_(m.bias)

    return m
