from __future__ import annotations

from enum import Enum

import torch


class WeightInitLabels(str, Enum):
    """Enumeration of weight initialisation method labels."""

    XAVIER = "xavier"


def xavier_weight_init(in_features: int, out_features: int) -> torch.Tensor:
    """
    Initialise the weights of a layer using Xavier initialisation.

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
    return torch.randn(in_features, out_features) * xavier_stddev


# Mapping from weight initialisation method labels to python functions
WEIGHT_INIT_REGISTRY = {
    WeightInitLabels.XAVIER: xavier_weight_init,
}
