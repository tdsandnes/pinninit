from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Callable


class LossFunctionLabels(str, Enum):
    """Enumeration of loss function labels."""

    MSE = "mse"  # Mean Square Error: sum(res^2) / n_points
    WEIGHTED_MEAN_MSE = "weighted_mean_mse"  # sum(w * res^2) / sum(weight)
    WEIGHTED_POINT_AVERAGED_MSE = (
        "weighted_point_avg_mse_loss"  # sum(w * res^2) / n_points
    )


def mse_loss(
    residuals: torch.Tensor,
    point_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Mean Squared Error (MSE) loss between predicted and exact function values.
    """
    if point_weights is not None:
        raise ValueError("point_weights should not be provided for MSE loss")

    return (residuals**2).mean()


def weighted_mean_mse_loss(
    residuals: torch.Tensor,
    point_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Weight mean square loss between predicted and exact function values.
    """
    if point_weights is None:
        raise ValueError("point_weights must be provided for weighted MSE loss")

    if point_weights.shape[0] != residuals.shape[0]:
        raise ValueError(
            f"point_weights length {point_weights.shape[0]} "
            f"must match residual length {residuals.shape[0]}"
        )
    return (point_weights * residuals**2).sum() / point_weights.sum()


def weighted_point_avg_mse_loss(
    residuals: torch.Tensor,
    point_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Weight mean square loss, but not divided by total weight.
    """
    if point_weights is None:
        raise ValueError("point_weights must be provided for weighted MSE loss")

    if point_weights.shape[0] != residuals.shape[0]:
        raise ValueError(
            f"point_weights length {point_weights.shape[0]} "
            f"must match residual length {residuals.shape[0]}"
        )
    return (point_weights * residuals**2).mean()


# Mapping from loss function labels to python functions
LOSS_FUNCTION_REGISTRY: dict[
    LossFunctionLabels,
    Callable[..., torch.Tensor],
] = {
    LossFunctionLabels.MSE: mse_loss,
    LossFunctionLabels.WEIGHTED_MEAN_MSE: weighted_mean_mse_loss,
    LossFunctionLabels.WEIGHTED_POINT_AVERAGED_MSE: weighted_point_avg_mse_loss,
}
