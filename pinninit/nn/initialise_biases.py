from __future__ import annotations

from enum import Enum

import torch


class BiasInitLabels(str, Enum):
    """Enumeration of bias initialisation method labels."""

    ZEROS = "zeros"


def zeros_bias_init(out_features: int) -> torch.Tensor:
    """
    Initialise the bias of a layer with zeros.

    Parameters:
    ----------
    out_features : int
        Number of output features.

    Returns:
    ----------
    torch.Tensor
        A tensor with the initialised biases.
    """
    return torch.zeros(out_features)


# Mapping from bias initialisation method labels to python functions
BIAS_INIT_REGISTRY = {
    BiasInitLabels.ZEROS: zeros_bias_init,
}
