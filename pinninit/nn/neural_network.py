from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from .activation_function import ACTIVATION_FUNC_REGISTRY
from .initialise_biases import BIAS_INIT_REGISTRY
from .initialise_weights import WEIGHT_INIT_REGISTRY
from .input_transformation import InputTransformation

if TYPE_CHECKING:
    from ..config.config import NNConfig
    from ..data_structure.coords_dataclass import CoordsProtocol


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
    def __init__(
        self,
        output_field_labels: tuple[str, ...],
        layers: list[nn.Module],
        nn_config: NNConfig,
        input_transformation: InputTransformation | None = None,
    ):
        super().__init__()
        self.output_field_labels = output_field_labels
        self.layers = nn.ModuleList(layers)
        activation_function_labels = nn_config.activation_functions
        init_weight_labels = nn_config.init_weights
        init_bias_labels = nn_config.init_biases

        for i, layer in enumerate(self.layers):
            assert isinstance(layer.weight, torch.Tensor)
            assert isinstance(layer.bias, torch.Tensor)
            WEIGHT_INIT_REGISTRY[init_weight_labels[i]](layer.weight)
            BIAS_INIT_REGISTRY[init_bias_labels[i]](layer.bias)

        self.activation_funcs = nn.ModuleList(
            [
                ACTIVATION_FUNC_REGISTRY[activation_function_labels[i]]()
                for i in range(len(self.layers))
            ]
        )
        self.input_transformation = input_transformation

    def evaluate(self, coords: CoordsProtocol) -> torch.Tensor:
        if self.input_transformation is not None:
            tensor = self.input_transformation.transform(coords)
        else:
            tensor = coords.internal
        return self.forward(tensor)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            tensor = layer(tensor)
            tensor = self.activation_funcs[i](tensor)
        return tensor


# Mapping from NN type labels to classes
NN_TYPE_REGISTRY: dict[NNTypeLabels, type[NeuralNetwork]] = {
    NNTypeLabels.FEEDFORWARD: FeedforwardNN,
}
