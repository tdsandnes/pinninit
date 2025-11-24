from enum import Enum

import torch


class WeightInitLabels(str, Enum):
    """Enumeration of weight initialisation method labels."""

    XAVIER = "xavier"


def xavier_weight_init(dim_in: int, dim_out: int) -> torch.Tensor:
    """
    Initialise the weights of a layer using Xavier initialisation.

    Parameters:
    ----------
    dim_in : int
        Number of input neurons.
    dim_out : int
        Number of output neurons.

    Returns:
    ----------
    torch.Tensor
        A tensor with the initialised weights.
    """
    xavier_stddev = torch.sqrt(torch.tensor(2.0 / (dim_in + dim_out)))
    return torch.randn(dim_in, dim_out) * xavier_stddev


# Mapping from weight initialisation method labels to python functions
WEIGHT_INIT_REGISTRY = {
    WeightInitLabels.XAVIER: xavier_weight_init,
}
