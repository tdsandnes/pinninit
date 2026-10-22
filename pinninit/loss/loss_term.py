from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ..data_structure.coords import Coords
    from .evaluation_context import EvaluationContext


class LossTermLabels(str, Enum):
    """Enumeration of loss term labels."""

    DATA = "data"
    PHYSICS = "physics"


class LossTerm(ABC):
    """Base class for loss terms"""

    def __init__(
        self,
        field_label: str,
    ) -> None:
        self.field_label = field_label

    def check(self, eval_context: EvaluationContext) -> None:
        """Check the loss term can be computed in eval_context."""
        if self.field_label not in eval_context.output_data.field_labels:
            raise ValueError(
                f"Loss term field '{self.field_label}' is not an output or "
                "derived field."
            )

    @abstractmethod
    def compute_residual(
        self,
        eval_context: EvaluationContext,
        eval_coords: Coords,
    ) -> torch.Tensor:
        """Subclasses must implement this method to compute residual."""
        pass

    def compute_loss_and_residuals(
        self,
        eval_context: EvaluationContext,
        eval_coords: Coords,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        residuals = self.compute_residual(eval_context, eval_coords)

        # Pass to loss function
        loss = eval_context.loss_function(
            residuals=residuals, eval_context=eval_context, eval_coords=eval_coords
        )

        # Return both loss and residuals
        return loss, residuals


class DataLoss(LossTerm):
    """Data loss term"""

    def check(self, eval_context: EvaluationContext) -> None:
        super().check(eval_context)
        if (
            eval_context.input_data is None
            or self.field_label not in eval_context.input_data.field_labels
        ):
            raise ValueError(
                f"DataLoss on field '{self.field_label}' needs input data for "
                "that field."
            )

    def compute_residual(
        self,
        eval_context: EvaluationContext,
        eval_coords: Coords,
    ) -> torch.Tensor:
        if eval_context.input_data is None:
            raise ValueError(
                f"DataLoss on field '{self.field_label}' requires input_data, "
                "but EvaluationContext.input_data is None."
            )

        target = eval_context.input_data.get_field_component_tensor(
            self.field_label, eval_coords
        )
        predicted = eval_context.output_data.get_field_component_tensor(
            self.field_label, eval_coords
        )
        return predicted - target


class PhysicsLoss(LossTerm):
    """Physics loss term"""

    def compute_residual(
        self,
        eval_context: EvaluationContext,
        eval_coords: Coords,
    ) -> torch.Tensor:
        return eval_context.output_data.get_field_component_tensor(
            self.field_label, eval_coords
        )


# Mapping from loss term labels to classes
LOSS_TERM_REGISTRY: dict[LossTermLabels, type[LossTerm]] = {
    LossTermLabels.DATA: DataLoss,
    LossTermLabels.PHYSICS: PhysicsLoss,
}
