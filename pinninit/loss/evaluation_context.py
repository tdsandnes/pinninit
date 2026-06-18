from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch.utils.data import DataLoader

    from ..data_structure.coords_dataclass import CoordsProtocol
    from ..data_structure.data_container import DataContainer
    from ..data_structure.field_dataclass import FieldProtocol


@dataclass(frozen=False, slots=True)
class EvaluationContext:
    """
    Shared evaluation state for loss computation.
    """

    output_data: DataContainer
    loss_function: Callable[..., torch.Tensor]
    dataloader: DataLoader[CoordsProtocol]
    input_data: DataContainer | None = None
    raw_point_weights: FieldProtocol | None = None
    point_weight_modifier: FieldProtocol | None = None

    coords: CoordsProtocol = field(init=False)

    def __post_init__(self) -> None:
        # Get shared coords from output_data (always present)
        self.coords = self.output_data.coords

        # When input_data is provided, enforce that both containers share coords
        if (
            self.input_data is not None
            and self.input_data.coords is not self.output_data.coords
        ):
            raise ValueError(
                "input_data and output_data must share identical coordinates."
            )

        if self.raw_point_weights is not None:
            self._validate_point_weights(self.raw_point_weights)
        if self.point_weight_modifier is not None:
            self._validate_point_weights(self.point_weight_modifier)

    def _validate_point_weights(self, field: FieldProtocol) -> None:
        # Must use same coords as data containers
        if field.coords is not self.coords:
            raise ValueError("point_weights field coords mismatch")

        # Must be scalar field
        if field.rank != 0:
            raise ValueError("point_weights must be a scalar field (rank=0)")

    def update_point_weights(
        self, new_weights: torch.Tensor, coords: CoordsProtocol | None = None
    ) -> None:
        """
        Update point weights in-place using FieldProtocol.
        """
        if self.raw_point_weights is None:
            raise ValueError("point_weights field is not initialised")

        coords = coords or self.coords

        # Check shape
        if new_weights.shape != (coords.n_points,):
            raise ValueError(
                f"Expected weights shape ({coords.n_points},), got {new_weights.shape}"
            )

        # Update Field
        self.raw_point_weights.update_from_tensor(
            new_tensor=new_weights,
            coords=coords,
            detach_stale_values=True,
            detach_new_values=True,
        )

    def get_raw_point_weights(
        self, coords: CoordsProtocol | None = None
    ) -> torch.Tensor:
        """
        Get the raw point weights, without applying the coord-based modifier.

        These are the weights owned and updated by adaptive trainer modules
        (e.g. RBA). Modules must read and write the raw weights so the modifier
        is not compounded across updates.
        """
        if self.raw_point_weights is None:
            raise ValueError("point_weights field is not initialised")

        coords = coords or self.coords

        return self.raw_point_weights.tensor(coords).detach()

    def get_point_weights(self, coords: CoordsProtocol | None = None) -> torch.Tensor:
        """
        Get the point weights used by the loss function.

        These are the raw weights multiplied by the coord-based modifier (if a
        point-weight equation is configured).
        """
        coords = coords or self.coords

        point_weights = self.get_raw_point_weights(coords)

        if self.point_weight_modifier is not None:
            point_weights = (
                point_weights * self.point_weight_modifier.tensor(coords).detach()
            )

        return point_weights
