from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

import torch

from ..utils import lookup_from_registry
from .loss_function import LOSS_FUNCTION_REGISTRY, LossFunctionLabels
from .loss_utils import is_loss_input_residual

if TYPE_CHECKING:
    from collections.abc import Callable


class LossTermLabels(str, Enum):
    """Enumeration of loss term labels."""

    DATA = "data"
    PHYSICS = "physics"


class LossTerm:
    """Base class for loss terms"""

    def __init__(self, loss_function_label: LossFunctionLabels):
        self.loss_function_label = loss_function_label
        # Look up the loss function in registry
        self.loss_func = lookup_from_registry(
            loss_function_label, LOSS_FUNCTION_REGISTRY
        )

    def compute_loss(
        self,
        f_pred: torch.Tensor | None = None,
        f_target: torch.Tensor | None = None,
        f_residual: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Computes loss using loss function.

        There are two valid ways to call this function:
            (1) Provide both f_pred and f_target
            (2) Provide f_residual only

        Parameters
        ----------
        f_pred : torch.Tensor, optional
            Predicted function values.
        f_target : torch.Tensor, optional
            Target function values.
        f_residual : torch.Tensor, optional
            Precomputed residual (f_pred - f_target)

        Returns
        -------
        torch.Tensor
            Loss corresponding to the specific loss term.
        """
        # Use residual value if provided. Otherwise, use f_pred and f_target.
        if is_loss_input_residual(f_pred, f_target, f_residual):
            loss = self.loss_func(f_residual)
        else:
            loss = self.loss_func(f_pred, f_target)

        assert isinstance(loss, torch.Tensor), f"Loss returned {type(loss)}"
        return loss

    def compute_weight(self) -> torch.Tensor:
        """
        Computes the weight of the loss term.

        ### Temp: currently just return scalar tensor 1. Ideally want this to go to a
        ### weights.py file where we can add various constant and adaptive methods.
        """
        return torch.tensor(1)

    def compute_weighted_loss(
        self,
        f_pred: torch.Tensor | None = None,
        f_target: torch.Tensor | None = None,
        f_residual: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Computes product of loss and the weight of the loss term.

        There are two valid ways to call this function:
            (1) Provide both f_pred and f_target
            (2) Provide f_residual only

        Parameters
        ----------
        f_pred : torch.Tensor, optional
            Predicted function values.
        f_target : torch.Tensor, optional
            Target function values.
        f_residual : torch.Tensor, optional
            Precomputed residual (f_pred - f_target)

        Returns
        -------
        torch.Tensor
            Weighted loss term.
        """
        weight = self.compute_weight()
        loss = self.compute_loss(f_pred, f_target, f_residual)
        return weight * loss


class DataLoss(LossTerm):
    """Data loss term"""

    def __init__(self, loss_function_label: LossFunctionLabels):
        super().__init__(loss_function_label)

    def compute_predicted_value(self, coords: torch.Tensor, net: None) -> torch.Tensor:
        ### not sure on this, will have to come back when networks are added
        ### In some way the loss term class will know what field(s) to use
        ###field_label = "temp"  ### better function name
        return torch.tensor(
            1
        )  ### temp: Something like this net.get_predicted_field(field_label, coords)

    def compute_target_value(self, coords: torch.Tensor) -> torch.Tensor:
        ### not sure on this, will have to come back when data handling is added
        ### Note sure whether data would already only be for the specific field, or we
        ### use the label here to extract right data
        ### In some way the loss term class will know what field(s) to use
        ###field_label = "temp"  ### better function name
        ### data = self.data  ### or something like that. Only set for data loss
        return torch.tensor(
            1
        )  ### temp: data[field_label, coords]  ### something like this? Maybe here the
        ### label needs to be an int and called field_idx

    def compute_residual(self, coords: torch.Tensor, net: None) -> torch.Tensor:
        ### In some way the loss term class will know what field(s) to use
        # Residual can be computed from f_pred - f_target
        f_pred = self.compute_predicted_value(coords, net)
        f_target = self.compute_target_value(coords)
        return f_pred - f_target


class PhysicsLoss(LossTerm):
    """Physics loss term"""

    def __init__(self, loss_function_label: LossFunctionLabels):
        super().__init__(loss_function_label)

    def compute_predicted_value(self, coords: torch.Tensor, net: None) -> torch.Tensor:
        ### not sure if this makes sense for phys loss
        return torch.tensor(1)  ### temp: Maybe error instead?

    def compute_target_value(self, coords: torch.Tensor) -> torch.Tensor:
        ### not sure if this makes sense for phys loss
        return torch.tensor(1)  ### temp: Maybe error instead?

    def compute_residual(self, coords: torch.Tensor, net: None) -> torch.Tensor:
        ### In some way the loss term class will know what eqn(s) to use
        ### eqn_label = "temp"  ### better function name
        return torch.tensor(
            1
        )  ### temp: phys.compute_eqn_residual(eqn_label, coords, net)


# Mapping from loss term labels to classes
LOSS_TERM_REGISTRY: dict[LossTermLabels, Callable[..., LossTerm] | type[LossTerm]] = {
    LossTermLabels.DATA: DataLoss,
    LossTermLabels.PHYSICS: PhysicsLoss,
}
