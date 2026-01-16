from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

import torch.nn as nn

if TYPE_CHECKING:
    pass


class ActivationFuncLabels(str, Enum):
    """Enumeration of activation function labels."""

    NONE = "none"
    TANH = "tanh"
    SILU = "silu"


# Mapping from activation function labels to python functions
ACTIVATION_FUNC_REGISTRY: dict[ActivationFuncLabels, type[nn.Module]] = {
    ActivationFuncLabels.NONE: nn.Identity,
    ActivationFuncLabels.TANH: nn.Tanh,
    ActivationFuncLabels.SILU: nn.SiLU,
}
