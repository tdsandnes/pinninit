from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Callable

    from ..data_structure.coords_dataclass import CoordsProtocol
    from .evaluation_context import EvaluationContext


class LossFunctionLabels(str, Enum):
    """Enumeration of loss function labels."""

    MSE = "mse"  # sum(w * res^2) / sum(weight) standard MSE when all weights = 1
    WEIGHTED_POINT_AVERAGED_MSE = "weighted_point_avg_mse"  # sum(w * res^2) / n_points


def mse_loss(
    residuals: torch.Tensor,
    eval_context: EvaluationContext,
    eval_coords: CoordsProtocol,
) -> torch.Tensor:
    """
    Weight mean square loss between predicted and exact function values.
    """
    # Get weights of individual points
    point_weights = eval_context.get_point_weights(eval_coords).unsqueeze(-1)

    if point_weights.shape[0] != residuals.shape[0]:
        raise ValueError(
            f"point_weights length {point_weights.shape[0]} "
            f"must match residual length {residuals.shape[0]}"
        )
    return (point_weights * residuals**2).sum(dim=0) / point_weights.sum()


def weighted_point_avg_mse_loss(
    residuals: torch.Tensor,
    eval_context: EvaluationContext,
    eval_coords: CoordsProtocol,
) -> torch.Tensor:
    """
    Weight mean square loss, but not divided by total weight.
    """
    # Get weights of individual points
    point_weights = eval_context.get_point_weights(eval_coords).unsqueeze(-1)

    if point_weights.shape[0] != residuals.shape[0]:
        raise ValueError(
            f"point_weights length {point_weights.shape[0]} "
            f"must match residual length {residuals.shape[0]}"
        )
    return (point_weights * residuals**2).mean(dim=0)


# Mapping from loss function labels to python functions
LOSS_FUNCTION_REGISTRY: dict[
    LossFunctionLabels,
    Callable[..., torch.Tensor],
] = {
    LossFunctionLabels.MSE: mse_loss,
    LossFunctionLabels.WEIGHTED_POINT_AVERAGED_MSE: weighted_point_avg_mse_loss,
}
