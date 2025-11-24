from enum import Enum

import activation_function as af
import bias_init as bi
import torch
import torch.nn as nn
import weight_init as wi
from utils import lookup_func_in_registry


class LayerTypeLabels(str, Enum):
    """Enumeration of layer type labels."""

    LINEAR = "linear"


class Layer(nn.Module):  # type: ignore[misc]
    """Base class for all layers."""

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        activation_func_label: af.ActivationFuncLabels,
        weight_init_label: wi.WeightInitLabels,
        bias_init_label: wi.BiasInitLabels,
    ):
        super().__init__()
        self.activation_func_label = activation_func_label
        self.weight_init_label = weight_init_label
        self.dim_in = dim_in
        self.dim_out = dim_out

        # Look up functions in registry
        self.activation_func = lookup_func_in_registry(
            activation_func_label, af.ACTIVATION_FUNC_REGISTRY
        )
        self.weight_init_func = lookup_func_in_registry(
            weight_init_label, wi.WEIGHT_INIT_REGISTRY
        )
        self.bias_init_func = lookup_func_in_registry(
            bias_init_label, bi.BIAS_INIT_REGISTRY
        )

        # Define weights and biases as trainable parameters
        self.weights = nn.Parameter(self.weight_init_func(dim_in, dim_out))
        self.biases = nn.Parameter(self.bias_init_func(dim_out))

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class LinearLayer(Layer):
    """
    Standard fully-connected layer + activation + optional weight init.
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        lower_bound: torch.Tensor,
        upper_bound: torch.Tensor,
        activation_func_label: af.ActivationFuncLabels,
        weight_init_label: wi.WeightInitLabels,
        bias_init_label: wi.BiasInitLabels,
    ):
        super().__init__(
            dim_in, dim_out, activation_func_label, weight_init_label, bias_init_label
        )
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the layer.

        Parameters:
        ----------
        tensor : torch.Tensor
            The input tensor.

        Returns:
        torch.Tensor
        ----------
            The output of the layer.
        """
        # Normalisation ### temp hardcoded for now, but might add a normalisation.py
        x = (
            2.0 * (tensor - self.lower_bound) / (self.upper_bound - self.lower_bound)
            - 1.0
        )
        y = x @ self.weights.T + self.biases  # matrix multiply + bias

        act_y = self.activation_func(y)
        assert isinstance(
            act_y, torch.Tensor
        ), f"Activation function returned {type(act_y)}"
        return act_y


# Mapping from layer type labels to classes
LAYER_TYPE_REGISTRY = {
    LayerTypeLabels.LINEAR: LinearLayer,
}
