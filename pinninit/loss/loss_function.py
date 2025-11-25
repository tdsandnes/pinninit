from enum import Enum

import torch
from loss_utils import is_loss_input_residual


class LossFunctionLabels(str, Enum):
    """Enumeration of loss function labels."""

    MSE = "mse"  # Mean Square Error


def mse_loss(
    f_pred: torch.Tensor | None,
    f_target: torch.Tensor | None,
    f_residual: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Mean Squared Error (MSE) loss between predicted and exact function values.

    There are two valid ways to call this function:
        (1) Provide both f_pred and f_target
        (2) Provide f_residual only

    Parameters
    ----------
    f_pred : torch.Tensor
        Predicted function values.
    f_target : torch.Tensor
        Target function values.
    f_residual : torch.Tensor, optional
        Precomputed residual (f_pred - f_target)

    Returns
    -------
    torch.Tensor
        MSE loss.
    """
    # Use residual value if provided. Otherwise, use f_pred and f_target.
    loss: torch.Tensor
    if is_loss_input_residual(f_pred, f_target, f_residual):
        assert f_residual is not None
        loss = (f_residual**2).mean()
    else:
        assert f_pred is not None and f_target is not None
        loss = ((f_pred - f_target) ** 2).mean()

    return loss


# Mapping from loss function labels to python functions
LOSS_FUNCTION_REGISTRY = {
    LossFunctionLabels.MSE: mse_loss,
}
