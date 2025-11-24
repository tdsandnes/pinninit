from enum import Enum

import torch


class BiasInitLabels(str, Enum):
    """Enumeration of bias initialisation method labels."""

    ZEROS = "zeros"


def zeros_bias_init(dim_out: int) -> torch.Tensor:
    """
    Initialise the bias of a layer with zeros.

    Parameters:
    ----------
    dim_out : int
        Number of output neurons.

    Returns:
    ----------
    torch.Tensor
        A tensor with the initialised biases.
    """
    return torch.zeros(dim_out)


# Mapping from bias initialisation method labels to python functions
BIAS_INIT_REGISTRY = {
    BiasInitLabels.ZEROS: zeros_bias_init,
}
