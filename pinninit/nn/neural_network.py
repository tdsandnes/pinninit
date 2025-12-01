from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from ..utils import execute_from_registry
from .activation_function import ACTIVATION_FUNC_REGISTRY
from .initialise_biases import BIAS_INIT_REGISTRY
from .initialise_weights import WEIGHT_INIT_REGISTRY

if TYPE_CHECKING:
    from ..io.input_parameters import LayersParams

### Maybe eventually add multiple NN subclasses if we want multiple options


class NeuralNetwork(nn.Module):
    """
    Use a list of Layer instances to produce a sequential model.
    """

    lb: torch.Tensor
    ub: torch.Tensor

    def __init__(
        self,
        layers: list[nn.Module],
        layers_params: LayersParams,
        lb: torch.Tensor,
        ub: torch.Tensor,
    ):
        super().__init__()
        self.layers = nn.ModuleList(layers)  # registers layers as submodules
        activation_function_labels = layers_params.activation_functions
        init_weight_lables = layers_params.init_weights
        init_bias_labels = layers_params.init_biases

        # Set activation functions and inital weights and biases for each layer
        activation_funcs = []
        for i, layer in enumerate(self.layers):
            # Look up activation function in registry
            activation_funcs.append(
                execute_from_registry(
                    activation_function_labels[i], ACTIVATION_FUNC_REGISTRY
                )
            )

            # Define weights and biases as trainable parameters
            execute_from_registry(
                init_weight_lables[i], WEIGHT_INIT_REGISTRY, layer.weight
            )
            execute_from_registry(init_bias_labels[i], BIAS_INIT_REGISTRY, layer.bias)

        self.activation_funcs = activation_funcs

        ### hardcoded normalisation for now
        # Register as buffers (not trainable parameters)
        self.register_buffer("lb", lb)
        self.register_buffer("ub", ub)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        # Normalisation ### temp hardcoded for now, but should add a normalisation.py
        ### that selects methods and then the method function in set in __init__
        lb = self.lb.unsqueeze(0)
        ub = self.ub.unsqueeze(0)
        tensor = 2.0 * (tensor - lb) / (ub - lb) - 1.0

        for i, layer in enumerate(self.layers):
            tensor = layer(tensor)  # call each layer's forward
            tensor = self.activation_funcs[i](tensor)  # Call activation function

        return tensor
