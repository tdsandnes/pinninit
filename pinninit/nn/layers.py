from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Self


class LayerTypeLabels(str, Enum):
    """Enumeration of layer type labels."""

    LINEAR = "linear"
    SIREN_FIRST = "siren_first"
    SIREN = "siren"


class SirenLinear(nn.Module):
    """
    Linear layer with omega-scaled weights for SIREN networks y = omega * W x + b.
    Slightly modified PyTorch nn.Linear class.
    """

    def __init__(
        self, in_features: int, out_features: int, omega: float, is_first: bool
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.omega = omega
        self.is_first = is_first
        self.weight = Parameter(torch.empty(out_features, in_features))
        self.bias = Parameter(torch.empty(out_features))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Runs the forward pass.
        """
        return F.linear(input, self.omega * self.weight, self.bias)

    def extra_repr(self) -> str:
        """
        Return the extra representation of the module.
        """
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"omega={self.omega}, is_first={self.is_first}"
        )

    @classmethod
    def first_layer(cls: type[Self], in_features: int, out_features: int) -> Self:
        # omega isn't configurable: it's 30 for the first layer
        return cls(in_features, out_features, omega=30.0, is_first=True)

    @classmethod
    def hidden_layer(cls: type[Self], in_features: int, out_features: int) -> Self:
        # omega isn't configurable: it's 1 for the hidden layers
        return cls(in_features, out_features, omega=1.0, is_first=False)


LAYER_TYPE_REGISTRY: dict[LayerTypeLabels, Callable[..., nn.Module]] = {
    LayerTypeLabels.LINEAR: nn.Linear,
    LayerTypeLabels.SIREN_FIRST: SirenLinear.first_layer,
    LayerTypeLabels.SIREN: SirenLinear.hidden_layer,
}
