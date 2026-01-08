from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING

import torch

from .loss_function import LOSS_FUNCTION_REGISTRY, LossFunctionLabels
from .loss_utils import is_loss_input_residual

if TYPE_CHECKING:
    from ..data_structure.data_container import CoordDataContainer


class LossTermLabels(str, Enum):
    """Enumeration of loss term labels."""

    DATA = "data"
    PHYSICS = "physics"


class LossTerm(ABC):
    """Base class for loss terms"""

    def __init__(
        self,
        field_label: str,
        component_idx: int,
        loss_function_label: LossFunctionLabels,
    ):
        self.field_label = field_label
        self.component_idx = component_idx
        self.loss_function_label = loss_function_label
        self.loss_func = LOSS_FUNCTION_REGISTRY[loss_function_label]

    @abstractmethod
    def compute_loss(
        self,
        input_data: CoordDataContainer,
        output_data: CoordDataContainer,
    ) -> torch.Tensor:
        """Subclasses must implement this method to compute loss."""
        pass

    @abstractmethod
    def compute_residual(
        self,
        input_data: CoordDataContainer,
        output_data: CoordDataContainer,
    ) -> torch.Tensor:
        """Subclasses must implement this method to compute residual."""
        pass

    def _compute_loss(
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
        if is_loss_input_residual(
            f_pred=f_pred, f_target=f_target, f_residual=f_residual
        ):
            loss = self.loss_func(f_residual=f_residual)
        else:
            loss = self.loss_func(f_pred=f_pred, f_target=f_target)

        assert isinstance(loss, torch.Tensor), f"Loss returned {type(loss)}"
        return loss


class DataLoss(LossTerm):
    """Data loss term"""

    def __init__(
        self,
        field_label: str,
        component_idx: int,
        loss_function_label: LossFunctionLabels,
    ):
        super().__init__(field_label, component_idx, loss_function_label)

    def compute_target_value(
        self,
        input_data: CoordDataContainer,
    ) -> torch.Tensor:
        return input_data.get_field(self.field_label).component_tensor[
            :, self.component_idx
        ]

    def compute_predicted_value(
        self,
        output_data: CoordDataContainer,
    ) -> torch.Tensor:
        return output_data.get_field(self.field_label).component_tensor[
            :, self.component_idx
        ]

    def compute_residual(
        self,
        input_data: CoordDataContainer,
        output_data: CoordDataContainer,
    ) -> torch.Tensor:
        f_target = self.compute_target_value(input_data)
        f_pred = self.compute_predicted_value(output_data)
        return f_pred - f_target

    def compute_loss(
        self,
        input_data: CoordDataContainer,
        output_data: CoordDataContainer,
    ) -> torch.Tensor:
        f_target = self.compute_target_value(input_data)
        f_pred = self.compute_predicted_value(output_data)
        return self._compute_loss(f_target=f_target, f_pred=f_pred)


class PhysicsLoss(LossTerm):
    """Physics loss term"""

    def __init__(
        self,
        field_label: str,
        component_idx: int,
        loss_function_label: LossFunctionLabels,
    ):
        super().__init__(field_label, component_idx, loss_function_label)

    def compute_residual(
        self,
        input_data: CoordDataContainer,
        output_data: CoordDataContainer,
    ) -> torch.Tensor:
        return output_data.get_field(self.field_label).component_tensor[
            :, self.component_idx
        ]

    def compute_loss(
        self,
        input_data: CoordDataContainer,
        output_data: CoordDataContainer,
    ) -> torch.Tensor:
        f_residual = self.compute_residual(input_data, output_data)
        return self._compute_loss(f_residual=f_residual)


# Mapping from loss term labels to classes
LOSS_TERM_REGISTRY: dict[LossTermLabels, type[LossTerm]] = {
    LossTermLabels.DATA: DataLoss,
    LossTermLabels.PHYSICS: PhysicsLoss,
}
