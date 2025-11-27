from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from ..checks.validate_data import validate_pytorch_tensor
from ..utils import execute_from_registry, lookup_from_registry
from .activation_function import ACTIVATION_FUNC_REGISTRY
from .initialise_biases import BIAS_INIT_REGISTRY
from .initialise_weights import WEIGHT_INIT_REGISTRY

if TYPE_CHECKING:
    from .activation_function import ActivationFuncLabels
    from .initialise_biases import BiasInitLabels
    from .initialise_weights import WeightInitLabels


class LayerTypeLabels(str, Enum):
    """Enumeration of layer type labels."""

    LINEAR = "linear"


class Layer(nn.Module, ABC):
    """Base class for all layers."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation_func_label: ActivationFuncLabels,
        weight_init_label: WeightInitLabels,
        bias_init_label: BiasInitLabels,
    ):
        super().__init__()
        self.activation_func_label = activation_func_label
        self.weight_init_label = weight_init_label
        self.in_features = in_features
        self.out_features = out_features

        # Look up activation function in registry
        self.activation_func = lookup_from_registry(
            activation_func_label, ACTIVATION_FUNC_REGISTRY
        )

        # Define weights and biases as trainable parameters
        weights = execute_from_registry(
            weight_init_label, WEIGHT_INIT_REGISTRY, in_features, out_features
        )
        biases = execute_from_registry(
            bias_init_label, BIAS_INIT_REGISTRY, out_features
        )
        validate_pytorch_tensor(weights, (in_features, out_features))
        validate_pytorch_tensor(biases, (out_features,))
        self.weights = nn.Parameter(weights)
        self.biases = nn.Parameter(biases)

    @abstractmethod
    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Subclasses must implement this method to forward pass through the layer.."""
        pass


class LinearLayer(Layer):
    """
    Standard fully-connected layer + activation + optional weight init.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation_func_label: ActivationFuncLabels,
        weight_init_label: WeightInitLabels,
        bias_init_label: BiasInitLabels,
    ):
        super().__init__(
            in_features,
            out_features,
            activation_func_label,
            weight_init_label,
            bias_init_label,
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the layer.

        Parameters:
        ----------
        tensor : torch.Tensor
            The input tensor. Size (n_points, in_features)

        Returns:
        torch.Tensor
        ----------
            The output of the layer. Size (n_points, out_features)
        """
        # Validate input tensors
        n_points = tensor.shape[0]
        validate_pytorch_tensor(tensor, (n_points, self.in_features))
        validate_pytorch_tensor(self.weights, (self.in_features, self.out_features))
        validate_pytorch_tensor(self.biases, (self.out_features,))

        linear_combination = (
            tensor @ self.weights.T + self.biases
        )  # matrix multiply + bias

        activated_output = self.activation_func(linear_combination)
        assert isinstance(
            activated_output, torch.Tensor
        ), f"Activation function returned {type(activated_output)}"
        return activated_output


# Mapping from layer type labels to classes
LAYER_TYPE_REGISTRY = {
    LayerTypeLabels.LINEAR: LinearLayer,
}
