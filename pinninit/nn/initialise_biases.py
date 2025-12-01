from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

import torch
import torch.nn.init as init

if TYPE_CHECKING:
    from collections.abc import Callable


class BiasInitLabels(str, Enum):
    """Enumeration of bias initialisation method labels."""

    ZEROS = "zeros"


def zeros_bias_init(bias: torch.Tensor) -> None:
    """
    Initialise the bias of a layer with zeros using PyTorch's built-in function.
    """
    init.zeros_(bias)


# Mapping from bias initialisation method labels to python functions
BIAS_INIT_REGISTRY: dict[BiasInitLabels, Callable[[torch.Tensor], None]] = {
    BiasInitLabels.ZEROS: zeros_bias_init,
}
