from enum import Enum

import torch


class ActivationFuncLabels(str, Enum):
    """Enumeration of activation function labels."""

    TANH = "tanh"


def tanh_activation_func(x: torch.Tensor) -> torch.Tensor:
    """Hyperbolic tangent activation function."""
    return torch.tanh(x)


# Mapping from activation function labels to python functions
ACTIVATION_FUNC_REGISTRY = {
    ActivationFuncLabels.TANH: tanh_activation_func,
}
