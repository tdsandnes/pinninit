from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

import torch.optim as optim

if TYPE_CHECKING:
    from collections.abc import Callable


class OptimizerLabels(str, Enum):
    """Enumeration of optimizer labels."""

    ADAM = "adam"


# Mapping from optimizer labels to classes
OPTIMIZER_REGISTRY: dict[OptimizerLabels, Callable[..., optim.Optimizer]] = {
    OptimizerLabels.ADAM: optim.Adam,
}
