import layers as lr
import torch
import torch.nn as nn


class NeuralNetwork(nn.Module):  # type: ignore[misc]
    """
    Use a list of Layer instances to produce a sequential model.
    """

    def __init__(self, layers: list[lr.Layer]):
        super().__init__()
        self.layers = nn.ModuleList(layers)  # registers layers as submodules

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            tensor = layer(tensor)  # call each layer's forward
        return tensor
