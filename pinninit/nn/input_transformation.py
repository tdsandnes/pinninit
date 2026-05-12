from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Callable

    from ..data_structure.coords_dataclass import CoordsProtocol



class InputTransformationLabels(str, Enum):
    """Enumeration of input transformation labels."""

    SCALE_LINEAR = "scale_linear"
    PERIODIC_W23 = "periodic_w23"

class InputTransformation:
    """Base class for input transformations."""

    def transform(self, coords: CoordsProtocol) -> torch.Tensor:
        raise NotImplementedError
    
    @property
    def out_features(self) -> int:
        raise NotImplementedError

class ScaleLinear(InputTransformation):
    """
    Maps coordinates linearly from [lb, ub] to [out_lb, out_ub].
    """
    def __init__(
        self,
        lb: torch.Tensor,
        ub: torch.Tensor,
        out_lb: float = -1.0,
        out_ub: float = 1.0,
    ) -> None:
        self.out_lb = out_lb
        self.out_ub = out_ub
        self.lb = lb.view(1, -1)
        self.ub = ub.view(1, -1)

    @property
    def out_features(self) -> int:
        return self.lb.shape[-1]

    def transform(self, coords: CoordsProtocol) -> torch.Tensor:
        return self.out_lb + (self.out_ub - self.out_lb) * ((coords.internal - self.lb) / (self.ub - self.lb))


class PeriodicW23(InputTransformation):
    """
    Hard periodic BCs via fixed Fourier embedding (Wang et al., 2023).

    For each periodic axis with period T, sets w = 2*pi/T and embeds as
    [cos(w*x), sin(w*x)] using raw (un-normalised) coordinates so the period
    is exact. Non-periodic axes are normalised to [-1, 1] and passed through.
    """

    def __init__(
        self,
        internal_labels: list[str],
        periodic_coords: dict[str, float],
        lb: torch.Tensor,
        ub: torch.Tensor,
        scale_non_periodic: bool = True,
    ) -> None:
        for label in periodic_coords:
            if label not in internal_labels:
                raise ValueError(
                    f"Periodic coord '{label}' not found in internal_labels: "
                    f"{internal_labels}"
                )
        self.internal_labels = internal_labels
        self.periodic_coords = periodic_coords
        self.lb = lb.view(1, -1)
        self.ub = ub.view(1, -1)
        self.scale_non_periodic = scale_non_periodic

    @property
    def out_features(self) -> int:
        n_non_periodic = len(self.internal_labels) - len(self.periodic_coords)
        return 2 * len(self.periodic_coords) + n_non_periodic

    def transform(self, coords: CoordsProtocol) -> torch.Tensor:
        x = coords.internal
        embeddings: list[torch.Tensor] = []

        for label, period in self.periodic_coords.items():
            idx = self.internal_labels.index(label)
            arg = (2.0 * torch.pi / period) * x[:, idx : idx + 1]
            embeddings.append(torch.cos(arg))
            embeddings.append(torch.sin(arg))

        for i, label in enumerate(self.internal_labels):
            if label not in self.periodic_coords:
                x_i = x[:, i : i + 1]
                if self.scale_non_periodic:
                    lb = self.lb[:, i : i + 1]
                    ub = self.ub[:, i : i + 1]
                    embeddings.append(2.0 * (x_i - lb) / (ub - lb) - 1.0)
                else:
                    embeddings.append(x_i)

        return torch.cat(embeddings, dim=-1)


INPUT_TRANSFORMATION_REGISTRY: dict[InputTransformationLabels, Callable[..., InputTransformation]] = {
    InputTransformationLabels.SCALE_LINEAR: ScaleLinear,
    InputTransformationLabels.PERIODIC_W23: PeriodicW23,
}
