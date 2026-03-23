# initialise_weights.py
from __future__ import annotations

import math
from enum import Enum
from typing import TYPE_CHECKING

import torch
import torch.nn.init as init

if TYPE_CHECKING:
    from collections.abc import Callable


class WeightInitLabels(str, Enum):
    """Enumeration of weight initialisation method labels."""

    XAVIER = "xavier"
    SIREN_FIRST = "siren_first"
    SIREN = "siren"


def xavier_weight_init(weight: torch.Tensor) -> None:
    """
    Initialise the weights of a layer using PyTorch's Xavier uniform initialisation.
    """
    # Apply PyTorch Xavier uniform init
    init.xavier_uniform_(weight)


def siren_first_weight_init(weight: torch.Tensor) -> None:
    """SIREN first layer weight initialisation (Sitzmann et al. 2020).
    https://colab.research.google.com/github/vsitzmann/siren/blob/master/explore_siren.ipynb#scrollTo=MKTWL-i5xWQC"""
    n = weight.shape[1]  # in_features
    bound = 1.0 / n
    init.uniform_(weight, -bound, bound)


def siren_weight_init(weight: torch.Tensor) -> None:
    """SIREN hidden layer weight initialisation (Sitzmann et al. 2020).
    https://colab.research.google.com/github/vsitzmann/siren/blob/master/explore_siren.ipynb#scrollTo=MKTWL-i5xWQC"""
    n = weight.shape[1]  # in_features
    omega = 1.0  ### hardcoded hidden layer omega for now
    bound = math.sqrt(6.0 / n) / omega
    init.uniform_(weight, -bound, bound)


# Mapping from weight initialisation method labels to python functions
WEIGHT_INIT_REGISTRY: dict[WeightInitLabels, Callable[[torch.Tensor], None]] = {
    WeightInitLabels.XAVIER: xavier_weight_init,
    WeightInitLabels.SIREN_FIRST: siren_first_weight_init,
    WeightInitLabels.SIREN: siren_weight_init,
}
