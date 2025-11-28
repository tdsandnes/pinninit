from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Callable


class ActivationFuncLabels(str, Enum):
    """Enumeration of activation function labels."""

    TANH = "tanh"


def tanh_activation_func(x: torch.Tensor) -> torch.Tensor:
    """Hyperbolic tangent activation function."""
    return torch.tanh(x)


# Mapping from activation function labels to python functions
ACTIVATION_FUNC_REGISTRY: dict[
    ActivationFuncLabels, Callable[[torch.Tensor], torch.Tensor]
] = {
    ActivationFuncLabels.TANH: tanh_activation_func,
}
