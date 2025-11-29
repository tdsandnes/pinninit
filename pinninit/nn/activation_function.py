from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Callable


class ActivationFuncLabels(str, Enum):
    """Enumeration of activation function labels."""

    NONE = "none"
    TANH = "tanh"


def no_activation_func(x: torch.Tensor) -> torch.Tensor:
    """No activation function."""
    return x


def tanh_activation_func(x: torch.Tensor) -> torch.Tensor:
    """Hyperbolic tangent activation function."""
    return torch.tanh(x)


# Mapping from activation function labels to python functions
ACTIVATION_FUNC_REGISTRY: dict[
    ActivationFuncLabels, Callable[[torch.Tensor], torch.Tensor]
] = {
    ActivationFuncLabels.NONE: no_activation_func,
    ActivationFuncLabels.TANH: tanh_activation_func,
}
