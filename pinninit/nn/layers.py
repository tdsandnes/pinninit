from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

import torch.nn as nn

if TYPE_CHECKING:
    from collections.abc import Callable


class LayerTypeLabels(str, Enum):
    """Enumeration of layer type labels."""

    LINEAR = "linear"


# Mapping from layer type labels to classes
LAYER_TYPE_REGISTRY: dict[
    LayerTypeLabels, Callable[..., nn.Module] | type[nn.Module]
] = {
    LayerTypeLabels.LINEAR: nn.Linear,
}
