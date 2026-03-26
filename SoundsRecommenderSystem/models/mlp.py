import torch
import torch.nn as nn
from typing import List, Optional, Type


class MLP(nn.Module):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_layers: List[int],
        activation_fn: Type[nn.Module] = nn.ReLU,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layers = hidden_layers
        self.activation_fn = activation_fn
        self.dropout = dropout

        self.model = self._init_mlp()

    def _init_mlp(self):
        layer_sizes = [self.input_size] + self.hidden_layers + [self.output_size]
        layers = []
        for i in range(len(layer_sizes) - 1):
            in_feat, out_feat = layer_sizes[i], layer_sizes[i + 1]
            layers.append(nn.Linear(in_features=in_feat, out_features=out_feat))

            is_last_layer = i == len(layer_sizes) - 2
            if not is_last_layer:
                layers.append(self.activation_fn())
                if self.dropout > 0:
                    layers.append(nn.Dropout(self.dropout))
        layers.append(nn.Sigmoid())

        model = nn.Sequential(*layers)
        return model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
