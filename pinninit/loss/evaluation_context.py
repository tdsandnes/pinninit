from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from ..data_structure.field import make_field_dataclass

if TYPE_CHECKING:
    from ..data_structure.batching import Batches
    from ..data_structure.coords import Coords
    from ..data_structure.data_container import DataContainer
    from ..data_structure.field import Field
    from .loss_function import LossFunction


@dataclass(frozen=False, slots=True)
class EvaluationContext:
    """
    The points of one loss term for one role (training or validation), and
    what is needed to evaluate the loss on them: the model output and derived
    fields, the input data (e.g. to compare against), the loss function, the
    batches and the per-point weights.
    """

    output_data: DataContainer
    loss_function: LossFunction
    batches: Batches
    input_data: DataContainer | None = None
    # Optional fixed per-point multiplier on the raw point weights
    point_weight_modifier: Field | None = None

    coords: Coords = field(init=False)
    # Per-point weights, starting at 1 and updated by adaptive trainer modules
    raw_point_weights: Field = field(init=False)

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

        ScalarField = make_field_dataclass(0, type(self.coords))
        self.raw_point_weights = ScalarField(
            label="point_weights",
            coords=self.coords,
            value=torch.ones(
                self.coords.n_points, dtype=self.coords.dtype, device=self.coords.device
            ),
        )

        if self.point_weight_modifier is not None:
            self._validate_point_weights(self.point_weight_modifier)

    def _validate_point_weights(self, field: Field) -> None:
        # Must use same coords as data containers
        if field.coords is not self.coords:
            raise ValueError("point_weights field coords mismatch")

        # Must be scalar field
        if field.rank != 0:
            raise ValueError("point_weights must be a scalar field (rank=0)")

    def update_point_weights(
        self, new_weights: torch.Tensor, coords: Coords | None = None
    ) -> None:
        """Set the raw point weights at coords (e.g. a batch), or at all points
        if coords is None."""
        self.raw_point_weights.update_from_tensor(new_weights.detach(), coords)

    def get_raw_point_weights(self, coords: Coords | None = None) -> torch.Tensor:
        """
        Get the raw point weights at coords, or at all points if coords is
        None, without applying the coord-based modifier.

        These are the weights owned and updated by adaptive trainer modules
        (e.g. RBA). Modules must read and write the raw weights so the modifier
        is not compounded across updates.
        """
        return self.raw_point_weights.tensor(coords).detach()

    def get_point_weights(self, coords: Coords | None = None) -> torch.Tensor:
        """
        Get the point weights used by the loss function, at coords or at all
        points if coords is None.

        These are the raw weights multiplied by the coord-based modifier (if a
        point-weight equation is configured).
        """
        point_weights = self.get_raw_point_weights(coords)

        if self.point_weight_modifier is not None:
            point_weights = (
                point_weights * self.point_weight_modifier.tensor(coords).detach()
            )

        return point_weights
