import layers as lr
import torch
import torch.nn as nn
from validate_data import validate_pytorch_tensor


class NeuralNetwork(nn.Module):  # type: ignore[misc]
    """
    Use a list of Layer instances to produce a sequential model.
    """

    def __init__(self, layers: list[lr.Layer], lb: torch.Tensor, ub: torch.Tensor):
        super().__init__()
        self.layers = nn.ModuleList(layers)  # registers layers as submodules

        # Validate input lower and upper bounds
        nn_in_features = layers[0].in_features
        validate_pytorch_tensor(lb, (nn_in_features,))
        validate_pytorch_tensor(ub, (nn_in_features,))
        self.lb = lb
        self.ub = ub

        # Register as buffers (not trainable parameters)
        self.register_buffer("lb", lb)
        self.register_buffer("ub", ub)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        # Normalisation ### temp hardcoded for now, but should add a normalisation.py
        ### that selects methods and then the method function in set in __init__
        tensor = 2.0 * (tensor - self.lb) / (self.ub - self.lb) - 1.0

        for layer in self.layers:
            tensor = layer(tensor)  # call each layer's forward
        return tensor
