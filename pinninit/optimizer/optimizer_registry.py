from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

from .optimizer import Optimizer
from .pytorch_optimizers import LBFGS, Adam
from .soap import SOAP

if TYPE_CHECKING:
    from collections.abc import Callable


class OptimizerLabels(str, Enum):
    """Enumeration of optimizer labels."""

    ADAM = "adam"
    LBFGS = "lbfgs"
    SOAP = "soap"


# Mapping from optimizer labels to classes
OPTIMIZER_REGISTRY: dict[OptimizerLabels, Callable[..., Optimizer]] = {
    OptimizerLabels.ADAM: Adam,
    OptimizerLabels.LBFGS: LBFGS,
    OptimizerLabels.SOAP: SOAP,
}
