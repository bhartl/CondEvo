from torch import Tensor, cat, nn as tnn
from torch.nn import Module, Tanh, Identity, Linear, Sequential, BatchNorm1d, Dropout, ReLU
from typing import Type, Union


class MLP(Module):
    def __init__(self, num_params=2, num_hidden=32, num_layers=2, activation: Union[Type, str] = ReLU,
                 last_activation: Union[Type, str] = Identity, num_conditions: int = 0,
                 batch_norm=False, dropout=0.0):
        """ Default model for diffusion.

        Args:
            num_params (int, optional): Number of parameters. Defaults to 2. Input dimension is num_params + 1.
            num_hidden (int, optional): Number of hidden units. Defaults to 32.
            num_layers (int, optional): Number of hidden layers. Defaults to 2.
            activation (Type, optional): Activation function. Defaults to nn.ReLU.
            last_activation (Type, optional): Activation function for the last layer. Defaults to nn.Identity.
            num_conditions (int, optional): Number of conditions. Defaults to 0.
            batch_norm (bool, optional): Whether to use batch normalization. Defaults to False.
            dropout (float, optional): Dropout rate. Defaults to 0.0.
        """
        super(MLP, self).__init__()
        self.num_params = num_params
        self.num_hidden = num_hidden
        self.num_layers = num_layers
        self.activation = activation
        self.last_activation = last_activation
        self.num_conditions = num_conditions
        self.batch_norm = batch_norm
        self.dropout = dropout

        self.layers = None
        self._build_model()

    def _build_model(self):
        h, foo, foo_l = self.num_hidden, self.activation, self.last_activation
        if isinstance(foo, str):
            foo = getattr(tnn, foo)
        if isinstance(foo_l, str):
            foo_l = getattr(tnn, foo_l)

        input_len = self.num_params + 1 + self.num_conditions  # params ,time, conditions
        output_len = self.num_params

        def get_block(in_features, out_features, batch_norm, dropout, activation):
            block = [Linear(in_features, out_features)]
            if batch_norm:
                block.append(BatchNorm1d(out_features))
            block.append(activation())
            if dropout > 0:
                block.append(Dropout(dropout))
            return block

        self.layers = Sequential(
            *get_block(input_len, h, self.batch_norm, self.dropout, foo),
            *[Sequential(*get_block(h, h, self.batch_norm, self.dropout, foo)) for _ in range(self.num_layers - 1)],
            *get_block(h, output_len, batch_norm=False, dropout=self.dropout, activation=foo_l)
        )

        return self.layers

    def forward(self, x: Tensor, t: Tensor, *conditions: Tensor) -> Tensor:
        """ Forward pass.

            Transforms a tensor of shape (batch_size, num_params + 1)
            to a tensor of shape (batch_size, num_params). """
        x = cat([x, t, *conditions], dim=-1)
        return self.layers(x)

    def __repr__(self):
        return f"MLP(num_params={self.num_params}, num_hidden={self.num_hidden}, num_layers={self.num_layers}, " \
               f"activation={self.activation}, last_activation={self.last_activation}, " \
               f"num_conditions={self.num_conditions}, batch_norm={self.batch_norm}, dropout={self.dropout})"
