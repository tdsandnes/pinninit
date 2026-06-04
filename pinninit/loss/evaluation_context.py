from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ..data_structure.coords_dataclass import CoordsProtocol
    from ..data_structure.data_container import DataContainer
    from ..data_structure.field_dataclass import FieldProtocol


@dataclass(frozen=False, slots=True)
class EvaluationContext:
    """
    Shared evaluation state for loss computation.
    """

    input_data: DataContainer
    output_data: DataContainer
    point_weights: FieldProtocol | None = None

    coords: CoordsProtocol = field(init=False)

    def __post_init__(self) -> None:
        """
        Enforce consistency between input and output data containers.
        """
        input_coords = self.input_data.coords
        output_coords = self.output_data.coords

        if (
            input_coords is not None
            and output_coords is not None
            and input_coords is not output_coords
        ):
            raise ValueError(
                "Input and output DataContainer must share identical coordinates."
            )
        self.coords = self.input_data.coords

        if self.point_weights is not None:
            self._validate_point_weights(self.point_weights)

    def _validate_point_weights(self, field: FieldProtocol) -> None:
        # Must use same coords as data containers
        if field.coords != self.coords:
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
        if self.point_weights is None:
            raise ValueError("point_weights field is not initialised")

        coords = coords or self.coords

        # Check shape
        if new_weights.shape != (coords.n_points,):
            raise ValueError(
                f"Expected weights shape ({coords.n_points},), got {new_weights.shape}"
            )

        # Update Field
        self.point_weights.update_from_tensor(
            new_tensor=new_weights,
            coords=coords,
            detach_stale_values=True,
            detach_new_values=True,
        )
