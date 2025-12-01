from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

import torch.nn as nn

if TYPE_CHECKING:
    from collections.abc import Callable


class ActivationFuncLabels(str, Enum):
    """Enumeration of activation function labels."""

    NONE = "none"
    TANH = "tanh"


# Mapping from activation function labels to python functions
ACTIVATION_FUNC_REGISTRY: dict[
    ActivationFuncLabels, Callable[..., nn.Module] | type[nn.Module]
] = {
    ActivationFuncLabels.NONE: nn.Identity,
    ActivationFuncLabels.TANH: nn.Tanh,
}
