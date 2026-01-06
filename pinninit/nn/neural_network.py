from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

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
        output_field_label_to_rank: dict[str, int],
        layers: list[nn.Module],
        layers_params: LayersParams,
        lb: torch.Tensor,
        ub: torch.Tensor,
    ):
        super().__init__()
        self.output_field_label_to_rank = output_field_label_to_rank
        self.layers = nn.ModuleList(layers)  # registers layers as submodules
        activation_function_labels = layers_params.activation_functions
        init_weight_lables = layers_params.init_weights
        init_bias_labels = layers_params.init_biases

        # Lower and Upper bounds
        self.lb = lb
        self.ub = ub

        # Set activation functions and inital weights and biases for each layer
        activation_funcs = []
        for i, layer in enumerate(self.layers):
            ### Maybe have to do this differently in case for general layer that might
            ### not have weights and biases
            assert isinstance(
                layer.weight, torch.Tensor
            ), f"Expected layer weight to be torch.Tensor, got {type(layer.weight)}"
            assert isinstance(
                layer.bias, torch.Tensor
            ), f"Expected layer bias to be torch.Tensor, got {type(layer.bias)}"

            # Look up activation function in registry
            activation_function_object = ACTIVATION_FUNC_REGISTRY[
                activation_function_labels[i]
            ]()
            activation_funcs.append(activation_function_object)

            # Define weights and biases as trainable parameters
            WEIGHT_INIT_REGISTRY[init_weight_lables[i]](layer.weight)
            BIAS_INIT_REGISTRY[init_bias_labels[i]](layer.bias)

        self.activation_funcs = activation_funcs

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
