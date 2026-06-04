from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch

from .loss_function import LOSS_FUNCTION_REGISTRY, LossFunctionLabels

if TYPE_CHECKING:
    from .evaluation_context import EvaluationContext


class LossTerm(ABC):
    """Base class for loss terms"""

    def __init__(
        self,
        field_label: str,
        component_label: str,
        loss_function_label: LossFunctionLabels,
    ):
        self.field_label = field_label
        self.component_label = component_label
        self.loss_function_label = loss_function_label
        self.loss_func = LOSS_FUNCTION_REGISTRY[loss_function_label]

    @abstractmethod
    def compute_residual(
        self,
        eval_context: EvaluationContext,
    ) -> torch.Tensor:
        """Subclasses must implement this method to compute residual."""
        pass

    def compute_loss(
        self,
        eval_context: EvaluationContext,
    ) -> torch.Tensor:
        residuals = self.compute_residual(eval_context)
        point_weights = None
        if eval_context.point_weights is not None:
            point_weights = eval_context.point_weights.tensor(eval_context.coords)
            if point_weights.shape != residuals.shape:
                raise ValueError(
                    "point_weights and residuals must have the same shape."
                )
        return self.loss_func(residuals=residuals, point_weights=point_weights)


class DataLoss(LossTerm):
    """Data loss term"""

    def compute_residual(
        self,
        eval_context: EvaluationContext,
    ) -> torch.Tensor:
        target = eval_context.input_data.get_field_component(
            self.field_label, self.component_label, eval_context.coords
        )
        predicted = eval_context.output_data.get_field_component(
            self.field_label, self.component_label, eval_context.coords
        )
        return predicted - target


class PhysicsLoss(LossTerm):
    """Physics loss term"""

    def compute_residual(
        self,
        eval_context: EvaluationContext,
    ) -> torch.Tensor:
        return eval_context.output_data.get_field_component(
            self.field_label, self.component_label, eval_context.coords
        )
