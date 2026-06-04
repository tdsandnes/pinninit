from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch

from .loss_function import LOSS_FUNCTION_REGISTRY

if TYPE_CHECKING:
    from ..data_structure.coords_dataclass import CoordsProtocol
    from .evaluation_context import EvaluationContext


class LossTerm(ABC):
    """Base class for loss terms"""

    def __init__(
        self,
        field_label: str,
    ):
        self.field_label = field_label

    @abstractmethod
    def compute_residual(
        self,
        eval_context: EvaluationContext,
        eval_coords: CoordsProtocol,
    ) -> torch.Tensor:
        """Subclasses must implement this method to compute residual."""
        pass

    def compute_loss(
        self,
        eval_context: EvaluationContext,
        eval_coords: CoordsProtocol,
    ) -> torch.Tensor:
        residuals = self.compute_residual(eval_context, eval_coords)
        point_weights = None
        if eval_context.point_weights is not None:
            point_weights = eval_context.point_weights.component_tensor(eval_coords)
            if point_weights.shape[0] != residuals.shape[0]:
                raise ValueError(
                    "point_weights and residuals must have the same length."
                )

        # Get loss function
        loss_func = LOSS_FUNCTION_REGISTRY[eval_context.loss_function_label]

        return loss_func(residuals=residuals, point_weights=point_weights)


class DataLoss(LossTerm):
    """Data loss term"""

    def compute_residual(
        self,
        eval_context: EvaluationContext,
        eval_coords: CoordsProtocol,
    ) -> torch.Tensor:
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
        eval_coords: CoordsProtocol,
    ) -> torch.Tensor:
        return eval_context.output_data.get_field_component_tensor(
            self.field_label, eval_coords
        )
