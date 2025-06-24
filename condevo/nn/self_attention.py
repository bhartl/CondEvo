from torch import Tensor, cat, nn as tnn
from torch.nn import Module, Tanh, Identity, Linear, Sequential, BatchNorm1d, Dropout, ReLU, TransformerEncoderLayer
from typing import Type, Union


class SelfAttentionMLP(Module):
    def __init__(self, num_params: int = 2, num_hidden: int = 32, num_layers: int = 2,
                 activation: Union[Type, str] = ReLU, last_activation: Union[Type, str] = Identity,
                 num_conditions: int = 0, batch_norm: bool = False, dropout: float = 0.0,
                 num_heads: int = 4):
        """ MLP-like model with self-attention on concatenated inputs and conditions.

        Args:
            num_params (int): Number of parameters. Input dimension is num_params + 1.
            num_hidden (int): Dimension for hidden layers in MLP and attention.
            num_layers (int): Number of hidden layers (applies to MLP after attention).
            activation (Type or str): Activation function. Defaults to ReLU.
            last_activation (Type or str): Activation for the last layer. Defaults to Identity.
            num_conditions (int): Number of conditions.
            batch_norm (bool): Whether to use batch normalization in MLP.
            dropout (float): Dropout rate.
            num_heads (int): Number of attention heads for self-attention.
        """
        super().__init__()
        self.num_params = num_params
        self.num_hidden = num_hidden
        self.num_layers = num_layers
        self.activation = activation
        self.last_activation = last_activation
        self.num_conditions = num_conditions
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.num_heads = num_heads

        self.input_dim = self.num_params + 1 + self.num_conditions  # params, time, conditions
        self.output_dim = self.num_params

        # Ensure num_hidden is divisible by num_heads for TransformerEncoderLayer
        if self.num_hidden % self.num_heads != 0:
            raise ValueError(f"num_hidden ({self.num_hidden}) must be divisible by num_heads ({self.num_heads})")

        self._build_model()

    def _build_model(self):
        h = self.num_hidden
        activation_fn = self.activation
        last_activation_fn = self.last_activation

        if isinstance(activation_fn, str):
            activation_fn = getattr(tnn, activation_fn)
        if isinstance(last_activation_fn, str):
            last_activation_fn = getattr(tnn, last_activation_fn)

        # 1. Input Linear Layer to project to num_hidden (if input_dim != num_hidden)
        # This is needed because TransformerEncoderLayer expects embed_dim
        # to be the feature dimension for its self-attention and FFN.
        self.input_projection = Linear(self.input_dim, h) if self.input_dim != h else Identity()

        # 2. Self-Attention Layer (TransformerEncoderLayer for simplicity)
        # We use a single TransformerEncoderLayer which contains both Multi-Head Self-Attention
        # and a Feed-Forward Network. This effectively replaces the initial MLP layers with attention.
        self.self_attention_layer = TransformerEncoderLayer(
            d_model=h,
            nhead=self.num_heads,
            dim_feedforward=h * 4,  # Standard practice for Transformer FFN
            dropout=self.dropout,
            activation=activation_fn.__name__.lower(),  # TransformerEncoderLayer expects string for activation
            batch_first=True  # Input is (batch_size, sequence_length, features)
        )

        # 3. MLP layers after self-attention (if num_layers > 1, as one layer is implicitly in attention)
        mlp_layers = []
        # The TransformerEncoderLayer already has a feed-forward network.
        # We model `num_layers` after the attention block. If `num_layers` is 1,
        # the TransformerEncoderLayer itself acts as the main processing block.
        # If num_layers > 1, we add additional MLP layers.
        # We start with `h` input features, as that's the output of the attention layer.

        # The TransformerEncoderLayer has its own FFN. If num_layers=1, this is effectively our "hidden layer".
        # If num_layers > 1, we add (num_layers - 1) additional MLP blocks.
        if self.num_layers > 1:
            for _ in range(self.num_layers - 1):
                mlp_layers.extend([Linear(h, h)])
                if self.batch_norm:
                    mlp_layers.append(BatchNorm1d(h))
                mlp_layers.append(activation_fn())
                if self.dropout > 0:
                    mlp_layers.append(Dropout(self.dropout))

        self.mlp_after_attention = Sequential(*mlp_layers)

        # 4. Output Linear Layer
        self.output_layer = Sequential(
            Linear(h, self.output_dim),
            last_activation_fn()
        )

    def forward(self, x: Tensor, t: Tensor, *conditions: Tensor) -> Tensor:
        """ Forward pass.

        Args:
            x (Tensor): Input parameters tensor of shape (batch_size, num_params).
            t (Tensor): Time tensor of shape (batch_size, 1).
            conditions (Tensor): Optional condition tensors, each of shape (batch_size, num_condition_features).

        Returns:
            Tensor: Output tensor of shape (batch_size, num_params).
        """
        # Concatenate inputs and conditions
        # Shape: (batch_size, num_params + 1 + num_conditions)
        concatenated_input = cat([x, t, *conditions], dim=-1)

        # Project to num_hidden dimension
        # Shape: (batch_size, num_hidden)
        projected_input = self.input_projection(concatenated_input)

        # Add a "sequence length" dimension for TransformerEncoderLayer (sequence_length=1)
        # Shape: (batch_size, 1, num_hidden)
        # This treats each concatenated input as a single "token" in a sequence of length 1.
        input_for_attention = projected_input.unsqueeze(1)

        # Apply self-attention
        # Output shape: (batch_size, 1, num_hidden)
        attention_output = self.self_attention_layer(input_for_attention)

        # Remove the "sequence length" dimension
        # Shape: (batch_size, num_hidden)
        attention_output = attention_output.squeeze(1)

        # Apply subsequent MLP layers
        mlp_output = self.mlp_after_attention(attention_output)

        # Final output layer
        output = self.output_layer(mlp_output)

        return output

    def __repr__(self):
        return f"SelfAttentionMLP(num_params={self.num_params}, num_hidden={self.num_hidden}, " \
               f"num_layers={self.num_layers}, activation={self.activation.__name__}, " \
               f"last_activation={self.last_activation.__name__}, num_conditions={self.num_conditions}, " \
               f"batch_norm={self.batch_norm}, dropout={self.dropout}, num_heads={self.num_heads})"


# Example Usage:
if __name__ == "__main__":
    # Basic test
    import torch
    model = SelfAttentionMLP(num_params=2, num_hidden=64, num_layers=3, num_conditions=0, num_heads=8, dropout=0.1)
    print(model)

    x_input = torch.randn(16, 2)  # Batch size 16, 2 parameters
    t_input = torch.randn(16, 1)  # Batch size 16, 1 time value

    output = model(x_input, t_input)
    print("Output shape (no conditions):", output.shape)  # Expected: torch.Size([16, 2])

    # Test with conditions
    model_with_cond = SelfAttentionMLP(num_params=3, num_hidden=128, num_layers=2, num_conditions=10, num_heads=8,
                                       batch_norm=True)
    print("\n", model_with_cond)

    x_input_cond = torch.randn(8, 3)
    t_input_cond = torch.randn(8, 1)
    cond1 = torch.randn(8, 5)
    cond2 = torch.randn(8, 5)  # Two condition tensors summing to 10 features

    output_cond = model_with_cond(x_input_cond, t_input_cond, cond1, cond2)
    print("Output shape (with conditions):", output_cond.shape)  # Expected: torch.Size([8, 3])

    # Test edge case: num_layers=1 (attention block effectively is the hidden layer)
    model_one_layer = SelfAttentionMLP(num_params=2, num_hidden=32, num_layers=1, num_conditions=0, num_heads=4)
    print("\n", model_one_layer)
    output_one_layer = model_one_layer(torch.randn(4, 2), torch.randn(4, 1))
    print("Output shape (num_layers=1):", output_one_layer.shape)

    # Test with input_dim == num_hidden (input_projection becomes Identity)
    model_identity_proj = SelfAttentionMLP(num_params=2, num_hidden=3, num_layers=2, num_conditions=0,
                                           num_heads=3)  # input_dim = 2+1 = 3
    print("\n", model_identity_proj)
    output_identity_proj = model_identity_proj(torch.randn(4, 2), torch.randn(4, 1))
    print("Output shape (input_dim=num_hidden):", output_identity_proj.shape)