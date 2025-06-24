import torch
from torch import Tensor, cat, nn as tnn
from torch.nn import Module, Tanh, Identity, Linear, Sequential, BatchNorm1d, Dropout, ReLU, TransformerEncoderLayer, \
    TransformerDecoderLayer
from typing import Type, Union, List


class CrossAttentionMLP(Module):
    def __init__(self, num_params: int = 2, num_hidden: int = 32, num_layers: int = 2,
                 activation: Union[Type, str] = ReLU, last_activation: Union[Type, str] = Identity,
                 num_conditions: int = 0, batch_norm: bool = False, dropout: float = 0.0,
                 num_heads: int = 4):
        """ MLP-like model with cross-attention between input (x, t) and conditions.

        Args:
            num_params (int): Number of parameters. Input (x, t) dimension is num_params + 1.
            num_hidden (int): Dimension for hidden layers in MLP and attention.
            num_layers (int): Number of hidden layers (applies to MLP after cross-attention).
            activation (Type or str): Activation function for FFNs. Defaults to ReLU.
            last_activation (Type or str): Activation for the last layer. Defaults to Identity.
            num_conditions (int): Total number of features in all conditions combined.
                                  Crucially, this is the *feature dimension* of the conditions.
                                  If multiple conditions are passed, they are concatenated.
            batch_norm (bool): Whether to use batch normalization in MLP.
            dropout (float): Dropout rate.
            num_heads (int): Number of attention heads for cross-attention.
        """
        super().__init__()
        self.num_params = num_params
        self.num_hidden = num_hidden
        self.num_layers = num_layers
        self.activation = activation
        self.last_activation = last_activation
        assert num_conditions >= 0, "num_conditions must be non-negative"
        self.num_conditions = num_conditions
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.num_heads = num_heads

        self.input_query_dim = self.num_params + 1  # Dimension of x (params) + t (time)
        self.input_key_value_dim = self.num_conditions  # Dimension of concatenated conditions

        # Ensure num_hidden is divisible by num_heads for Transformer layers
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

        # 1. Linear layer to project input (x, t) to num_hidden (for Query)
        self.input_query_projection = Linear(self.input_query_dim, h)

        # 2. Linear layer to project conditions to num_hidden (for Key and Value)
        # If no conditions, this should ideally be an Identity or not used.
        self.conditions_kv_projection = Linear(self.input_key_value_dim,
                                               h) if self.input_key_value_dim > 0 else Identity()

        # 3. Cross-Attention Layer (TransformerDecoderLayer with self-attention disabled for Q)
        # We use TransformerDecoderLayer. It has:
        #   - self-attention (Q=Q) -> disabled here by providing no memory_mask
        #   - cross-attention (Q vs K,V from memory)
        #   - FFN
        # The key idea is that the *memory* is the conditions.
        self.cross_attention_layer = TransformerDecoderLayer(
            d_model=h,
            nhead=self.num_heads,
            dim_feedforward=h * 4,  # Standard for Transformer FFN
            dropout=self.dropout,
            activation=activation_fn.__name__.lower(),  # TransformerEncoderLayer expects string for activation
            batch_first=True  # Input is (batch_size, sequence_length, features)
        )

        # 4. MLP layers after cross-attention
        mlp_layers = []
        # The TransformerDecoderLayer has its own FFN. If num_layers=1, this is effectively our "hidden layer".
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

        # 5. Output Linear Layer
        self.output_layer = Sequential(
            Linear(h, self.num_params),
            last_activation_fn()
        )

    def forward(self, x: Tensor, t: Tensor, *conditions: Tensor) -> Tensor:
        """ Forward pass.

        Args:
            x (Tensor): Input parameters tensor of shape (batch_size, num_params).
            t (Tensor): Time tensor of shape (batch_size, 1).
            conditions (Tensor): Optional condition tensors, each of shape (batch_size, num_condition_features_i).
                                 These will be concatenated to form the memory for cross-attention.

        Returns:
            Tensor: Output tensor of shape (batch_size, num_params).
        """
        # 1. Prepare Query (Q) from x and t
        # Shape: (batch_size, num_params + 1)
        query_input = cat([x, t], dim=-1)
        # Project to num_hidden: (batch_size, num_hidden)
        query_projected = self.input_query_projection(query_input)
        # Add sequence length dim for Transformer: (batch_size, 1, num_hidden)
        query_for_attention = query_projected.unsqueeze(1)

        # 2. Prepare Key (K) and Value (V) from conditions
        # Concatenate all condition tensors
        # Shape: (batch_size, total_num_conditions)
        concatenated_conditions = cat(conditions, dim=-1)
        # Project to num_hidden: (batch_size, num_hidden)
        memory_projected = self.conditions_kv_projection(concatenated_conditions)
        # Add sequence length dim for Transformer: (batch_size, 1, num_hidden)
        # Each concatenated condition set is a single 'memory' token for each batch item.
        memory_for_attention = memory_projected.unsqueeze(1)


        # 3. Apply Cross-Attention
        # `tgt` (target) is the query (x, t)
        # `memory` is the conditions (K, V)
        # `tgt_mask`, `memory_mask`, `tgt_key_padding_mask`, `memory_key_padding_mask` are None here
        # to simplify. If conditions had variable lengths or were padded, these masks would be needed.
        # The first output from TransformerDecoderLayer (tgt) corresponds to the output of the
        # self-attention (if enabled) and cross-attention. We are implicitly disabling the
        # self-attention within TransformerDecoderLayer by not providing `tgt_mask`.
        # However, it's safer to ensure the input to `TransformerDecoderLayer` is treated correctly.
        # A simple `TransformerDecoderLayer` will still run `self_attn` even if `tgt_mask` is `None`.
        # For *pure* cross-attention, we'd need to manually build `MultiheadAttention` or use
        # `TransformerDecoder` (which applies self-attention then cross-attention).
        # To avoid the self-attention part of TransformerDecoderLayer (which operates on queries),
        # we can set the `is_causal` flag if using PyTorch 2.0+ `nn.TransformerDecoderLayer` with `is_causal=True`
        # and `tgt_mask=None`, it would act like a self-attention.
        # However, the simpler interpretation for this use case is that the `TransformerDecoderLayer`
        # is processing the query (`x`, `t`) and using `memory` (`conditions`) for cross-attention.
        # The key for pure cross-attention is that the `MultiheadAttention` module gets different
        # `query`, `key`, `value` inputs. `TransformerDecoderLayer` handles this where `query` is `tgt`
        # and `key/value` are `memory`.

        attention_output = self.cross_attention_layer(tgt=query_for_attention, memory=memory_for_attention)

        # Remove the "sequence length" dimension
        # Shape: (batch_size, num_hidden)
        attention_output = attention_output.squeeze(1)

        # Apply subsequent MLP layers
        mlp_output = self.mlp_after_attention(attention_output)

        # Final output layer
        output = self.output_layer(mlp_output)

        return output

    def __repr__(self):
        return f"CrossAttentionMLP(num_params={self.num_params}, num_hidden={self.num_hidden}, " \
               f"num_layers={self.num_layers}, activation={self.activation.__name__}, " \
               f"last_activation={self.last_activation.__name__}, num_conditions={self.num_conditions}, " \
               f"batch_norm={self.batch_norm}, dropout={self.dropout}, num_heads={self.num_heads})"

# Example Usage:
if __name__ == "__main__":
    # Basic test
    # num_conditions is the *total* feature dimension of all conditions.
    # If you pass two conditions of size 5 each, num_conditions should be 10.
    model = CrossAttentionMLP(num_params=2, num_hidden=64, num_layers=3, num_conditions=10, num_heads=8, dropout=0.1)
    print(model)

    x_input = torch.randn(16, 2)  # Batch size 16, 2 parameters
    t_input = torch.randn(16, 1)  # Batch size 16, 1 time value
    cond_a = torch.randn(16, 5)   # First condition tensor
    cond_b = torch.randn(16, 5)   # Second condition tensor

    output = model(x_input, t_input, cond_a, cond_b)
    print("Output shape (with conditions):", output.shape) # Expected: torch.Size([16, 2])

    # Test with no conditions (num_conditions=0)
    # The model still builds a `conditions_kv_projection` but it will be `Identity` or handle zero dim.
    # The `memory_for_attention` will be `None`, and `TransformerDecoderLayer` should handle this
    # by skipping cross-attention. It will still apply self-attention on `tgt`.
    model_no_cond = CrossAttentionMLP(num_params=3, num_hidden=32, num_layers=2, num_conditions=0, num_heads=4, batch_norm=True)
    print("\n", model_no_cond)

    x_input_no_cond = torch.randn(8, 3)
    t_input_no_cond = torch.randn(8, 1)

    # Test edge case: num_layers=1
    model_one_layer = CrossAttentionMLP(num_params=2, num_hidden=32, num_layers=1, num_conditions=5, num_heads=4)
    print("\n", model_one_layer)
    output_one_layer = model_one_layer(torch.randn(4,2), torch.randn(4,1), torch.randn(4,5))
    print("Output shape (num_layers=1):", output_one_layer.shape)

    # Test with conditions_kv_projection becoming Identity
    # If num_conditions == num_hidden, the projection can be Identity
    model_identity_kv = CrossAttentionMLP(num_params=2, num_hidden=5, num_layers=2, num_conditions=5, num_heads=5)
    print("\n", model_identity_kv)
    output_identity_kv = model_identity_kv(torch.randn(4,2), torch.randn(4,1), torch.randn(4,5))
    print("Output shape (conditions_kv_projection=Identity):", output_identity_kv.shape)