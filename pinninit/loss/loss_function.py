from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Protocol

import torch

if TYPE_CHECKING:
    from ..data_structure.coords import Coords
    from .evaluation_context import EvaluationContext


class LossFunctionLabels(str, Enum):
    """Enumeration of loss function labels."""

    MSE = "mse"  # sum(res^2) / n_points
    WEIGHTED_MSE = "weighted_mse"  # sum(w * res^2) / sum(w)
    WEIGHTED_POINT_AVERAGED_MSE = "weighted_point_avg_mse"  # sum(w * res^2) / n_points


class LossFunction(Protocol):
    """A loss function: the loss of each residual component, shape
    (n_components,), from residuals of shape (n_points, n_components) at
    eval_coords."""

    def __call__(
        self,
        residuals: torch.Tensor,
        eval_context: EvaluationContext,
        eval_coords: Coords,
    ) -> torch.Tensor: ...


def mse_loss(
    residuals: torch.Tensor,
    eval_context: EvaluationContext,
    eval_coords: Coords,
) -> torch.Tensor:
    """Mean squared residual. Ignores point weights."""
    return (residuals**2).sum(dim=0) / residuals.shape[0]


def weighted_mse_loss(
    residuals: torch.Tensor,
    eval_context: EvaluationContext,
    eval_coords: Coords,
) -> torch.Tensor:
    """Squared residuals averaged with the point weights as weights."""
    point_weights = _point_weights(residuals, eval_context, eval_coords)
    return (point_weights * residuals**2).sum(dim=0) / point_weights.sum()


def weighted_point_avg_mse_loss(
    residuals: torch.Tensor,
    eval_context: EvaluationContext,
    eval_coords: Coords,
) -> torch.Tensor:
    """Squared residuals times the point weights, averaged over the points (so
    not divided by the total weight)."""
    point_weights = _point_weights(residuals, eval_context, eval_coords)
    return (point_weights * residuals**2).mean(dim=0)


def _point_weights(
    residuals: torch.Tensor, eval_context: EvaluationContext, eval_coords: Coords
) -> torch.Tensor:
    """Point weights at eval_coords, shape (n_points, 1) to multiply residuals."""
    point_weights = eval_context.get_point_weights(eval_coords).unsqueeze(-1)
    if point_weights.shape[0] != residuals.shape[0]:
        raise ValueError(
            f"point_weights length {point_weights.shape[0]} "
            f"must match residual length {residuals.shape[0]}"
        )
    return point_weights


# Mapping from loss function labels to python functions
LOSS_FUNCTION_REGISTRY: dict[LossFunctionLabels, LossFunction] = {
    LossFunctionLabels.MSE: mse_loss,
    LossFunctionLabels.WEIGHTED_MSE: weighted_mse_loss,
    LossFunctionLabels.WEIGHTED_POINT_AVERAGED_MSE: weighted_point_avg_mse_loss,
}
