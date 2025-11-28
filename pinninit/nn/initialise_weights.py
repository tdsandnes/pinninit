from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Callable


class WeightInitLabels(str, Enum):
    """Enumeration of weight initialisation method labels."""

    XAVIER = "xavier"


def xavier_weight_init(in_features: int, out_features: int) -> torch.Tensor:
    """
    Initialise the weights of a layer using Xavier initialisation.

    Note that the shape of the tensor is (out_features, in_features) to match
    PyTorch conventions.

    Parameters:
    ----------
    in_features : int
        Number of input features.
    out_features : int
        Number of output features.

    Returns:
    ----------
    torch.Tensor
        A tensor with the initialised weights.
    """
    xavier_stddev = torch.sqrt(torch.tensor(2.0 / (in_features + out_features)))
    return torch.randn(out_features, in_features) * xavier_stddev


# Mapping from weight initialisation method labels to python functions
WEIGHT_INIT_REGISTRY: dict[WeightInitLabels, Callable[[int, int], torch.Tensor]] = {
    WeightInitLabels.XAVIER: xavier_weight_init,
}
