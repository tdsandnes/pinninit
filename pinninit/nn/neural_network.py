from __future__ import annotations

from typing import TYPE_CHECKING

from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from enum import Enum

from .activation_function import ACTIVATION_FUNC_REGISTRY
from .initialise_biases import BIAS_INIT_REGISTRY
from .initialise_weights import WEIGHT_INIT_REGISTRY
from .input_transformation import InputTransformation

if TYPE_CHECKING:
    from ..data_structure.coords_dataclass import CoordsProtocol
    from ..io.input_parameters import LayersParams


class NNTypeLabels(str, Enum):
    FEEDFORWARD = "feedforward"
    ### e.g. PIRATENET = "piratenet"


class NeuralNetwork(nn.Module, ABC):
    """Base class for all network architectures."""
    
    output_field_labels: tuple[str, ...]
    
    @abstractmethod
    def evaluate(self, coords: CoordsProtocol) -> torch.Tensor:
        pass


class FeedforwardNN(NeuralNetwork):
    lb: torch.Tensor
    ub: torch.Tensor

    def __init__(
        self,
        output_field_labels: tuple[str],
        layers: list[nn.Module],
        layers_params: LayersParams,
        lb: torch.Tensor,
        ub: torch.Tensor,
        input_transformation: InputTransformation | None = None,
    ):
        super().__init__()
        self.output_field_labels = output_field_labels
        self.layers = nn.ModuleList(layers)  # registers layers as submodules
        activation_function_labels = layers_params.activation_functions
        init_weight_labels = layers_params.init_weights
        init_bias_labels = layers_params.init_biases

        # Lower and Upper bounds
        self.register_buffer("lb", lb.view(1, -1))
        self.register_buffer("ub", ub.view(1, -1))

        # Set activation functions and inital weights and biases for each layer
        for i, layer in enumerate(self.layers):
            ### Maybe have to do this differently in case for general layer that might
            ### not have weights and biases
            assert isinstance(
                layer.weight, torch.Tensor
            ), f"Expected layer weight to be torch.Tensor, got {type(layer.weight)}"
            assert isinstance(
                layer.bias, torch.Tensor
            ), f"Expected layer bias to be torch.Tensor, got {type(layer.bias)}"
            WEIGHT_INIT_REGISTRY[init_weight_labels[i]](layer.weight)
            BIAS_INIT_REGISTRY[init_bias_labels[i]](layer.bias)

        self.activation_funcs = nn.ModuleList(
            [
                ACTIVATION_FUNC_REGISTRY[activation_function_labels[i]]()
                for i in range(len(self.layers))
            ]
        )

        # Input transformations
        self.input_transformation = input_transformation

    def evaluate(self, coords: CoordsProtocol) -> torch.Tensor:
        if self.input_transformation is not None:
            tensor = self.input_transformation.transform(coords)
        else:
            tensor = coords.internal
        return self.forward(tensor)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            tensor = layer(tensor)  # call each layer's forward
            tensor = self.activation_funcs[i](tensor)  # Call activation function

        return tensor

# Mapping from NN type labels to classes
NN_TYPE_REGISTRY: dict[NNTypeLabels, type[NeuralNetwork]] = {
    NNTypeLabels.FEEDFORWARD: FeedforwardNN,
}
