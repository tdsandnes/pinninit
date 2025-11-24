from enum import Enum

import torch


class WeightInitLabels(str, Enum):
    """Enumeration of weight initialisation method labels."""

    XAVIER = "xavier"


def xavier_weight_init(layer_size: tuple[int, int]) -> torch.Tensor:
    """
    Initialize the weights of a layer using Xavier initialisation.

    Parameters:
    ----------
    layer_size : (tuple)
        The layer's dimensions (in_dim, out_dim).

    Returns:
    ----------
    torch.Tensor
        A tensor with the initialised weights.
    """
    in_dim, out_dim = layer_size
    xavier_stddev = torch.sqrt(torch.tensor(2.0 / (in_dim + out_dim)))
    return torch.randn(in_dim, out_dim) * xavier_stddev


# Mapping from weight initialisation method labels to python functions
WEIGHT_INIT_REGISTRY = {
    WeightInitLabels.XAVIER: xavier_weight_init,
}
