from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any, Self


class LayerTypeLabels(str, Enum):
    """Enumeration of layer type labels."""

    LINEAR = "linear"
    SIREN_FIRST = "siren_first"
    SIREN = "siren"


class SirenLinear(nn.Module):
    """
    Linear layer with omega-scaled weights for SIREN networks y = omega * W *x + b.
    Slightly modified Pytorch nn.Linear class.
    """

    __constants__ = ["in_features", "out_features", "omega", "is_first"]

    in_features: int
    out_features: int
    omega: float
    is_first: bool
    weight: torch.Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        omega: float,
        is_first: bool = False,
        bias: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.omega = omega
        self.is_first = is_first
        self.weight = Parameter(
            torch.empty(
                (out_features, in_features),
                device=device,
                dtype=dtype,
            )
        )
        if bias:
            self.bias = Parameter(
                torch.empty(
                    out_features,
                    device=device,
                    dtype=dtype,
                )
            )
        else:
            self.register_parameter("bias", None)

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
            f"omega={self.omega}, is_first={self.is_first}, "
            f"bias={self.bias is not None}"
        )

    @classmethod
    def first_layer(
        cls: type[Self], in_features: int, out_features: int, **kwargs: Any
    ) -> SirenLinear:
        # omega isn't configurable: it's 30 for the first layer
        return cls(in_features, out_features, omega=30.0, is_first=True, **kwargs)

    @classmethod
    def hidden_layer(
        cls: type[Self], in_features: int, out_features: int, **kwargs: Any
    ) -> SirenLinear:
        # omega isn't configurable: it's 1 for the hidden layers
        return cls(in_features, out_features, omega=1.0, is_first=False, **kwargs)


LAYER_TYPE_REGISTRY: dict[LayerTypeLabels, Callable[..., nn.Module]] = {
    LayerTypeLabels.LINEAR: nn.Linear,
    LayerTypeLabels.SIREN_FIRST: SirenLinear.first_layer,
    LayerTypeLabels.SIREN: SirenLinear.hidden_layer,
}
