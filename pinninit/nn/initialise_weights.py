from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

import torch
import torch.nn.init as init

if TYPE_CHECKING:
    from collections.abc import Callable


class WeightInitLabels(str, Enum):
    """Enumeration of weight initialisation method labels."""

    XAVIER = "xavier"


def xavier_weight_init(weight: torch.Tensor) -> None:
    """
    Initialise the weights of a layer using PyTorch's Xavier uniform initialisation.
    """
    # Apply PyTorch Xavier uniform init
    init.xavier_uniform_(weight)


# Mapping from weight initialisation method labels to python functions
WEIGHT_INIT_REGISTRY: dict[WeightInitLabels, Callable[[torch.Tensor], None]] = {
    WeightInitLabels.XAVIER: xavier_weight_init,
}
