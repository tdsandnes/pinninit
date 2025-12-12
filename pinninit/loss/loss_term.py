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

    ### str for now but make all these enum
    def __init__(
        self,
        loss_function_label: LossFunctionLabels,
        weight_method_label: str,
        weight_value_label: str,
    ):
        self.loss_function_label = loss_function_label
        # Look up the loss function in registry
        self.loss_func = LOSS_FUNCTION_REGISTRY[loss_function_label]
        self.weight_method_label = weight_method_label
        ### Temp until better way of handling different methods
        self.weight_value_label = weight_value_label

    @abstractmethod
    def compute_loss(
        self,
        input_data: CoordDataContainer,
        output_data: CoordDataContainer,
    ) -> torch.Tensor:
        """Subclasses must implement this method to compute loss."""
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

    def _compute_weight(self) -> torch.Tensor:
        """
        Computes the weight of the loss term.

        ### Temp: currently just return scalar self.weight_value_label. Ideally want
        ### this to go to a loss_weights.py file where we can add various constant
        and adaptive methods.
        """
        ### Temp until better way of handling different methods
        return torch.tensor(self.weight_value_label)

    def _compute_weighted_loss(
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
        weight = self._compute_weight()
        loss = self._compute_loss(
            f_pred=f_pred, f_target=f_target, f_residual=f_residual
        )
        return weight * loss


class DataLoss(LossTerm):
    """Data loss term"""

    def __init__(
        self,
        field_label: str,
        field_index: int,
        loss_function: LossFunctionLabels,
        weight_method: str,
        weight_value: str,
    ):
        super().__init__(loss_function, weight_method, weight_value)

        # Extract loss term params
        self.field_label = field_label
        self.field_index = field_index

    def compute_target_value(
        self,
        input_data: CoordDataContainer,
    ) -> torch.Tensor:
        if self.field_label in input_data.fields:
            field = input_data.fields[self.field_label].tensor
        elif self.field_label in input_data.derivable_fields:
            field = input_data.derivable_fields[self.field_label]
        else:
            raise KeyError(
                f"Field label '{self.field_label}' not found in dictionaries of input "
                "or derived input fields."
            )

        if field.dim() == 1:
            return field
        elif field.dim() == 2:
            return field[:, self.field_index]
        else:
            raise ValueError(
                "Current implimentation is only compatible with scalar and vector "
                " fields."
            )

    def compute_predicted_value(
        self,
        output_data: CoordDataContainer,
    ) -> torch.Tensor:
        if self.field_label in output_data.fields:
            field = output_data.fields[self.field_label].tensor
        elif self.field_label in output_data.derivable_fields:
            field = output_data.derivable_fields[self.field_label]
        else:
            raise KeyError(
                f"Field label '{self.field_label}' not found in dictionaries of output "
                "or derived output fields."
            )

        if field.dim() == 1:
            return field
        elif field.dim() == 2:
            return field[:, self.field_index]
        else:
            raise ValueError(
                "Current implimentation is only compatible with scalar and vector "
                " fields."
            )

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
        return self._compute_weighted_loss(f_target=f_target, f_pred=f_pred)


class PhysicsLoss(LossTerm):
    """Physics loss term"""

    def __init__(
        self,
        equation_label: str,
        residual_label: str,
        residual_index: int,
        loss_function: LossFunctionLabels,
        weight_method: str,
        weight_value: str,
    ):
        super().__init__(loss_function, weight_method, weight_value)

        # Extract loss term params
        self.equation_label = equation_label
        self.residual_label = residual_label
        self.residual_index = residual_index

    def compute_residual(
        self,
        output_data: CoordDataContainer,
    ) -> torch.Tensor:
        if self.residual_label in output_data.fields:
            field = output_data.fields[self.residual_label].tensor
        elif self.residual_label in output_data.derivable_fields:
            field = output_data.derivable_fields[self.residual_label]
        else:
            raise KeyError(
                f"Residual label '{self.residual_label}' not found in dictionaries "
                "of output or derived output fields."
            )

        if field.dim() == 1:
            return field
        elif field.dim() == 2:
            return field[:, self.residual_index]
        else:
            raise ValueError(
                "Current implimentation is only compatible with scalar and vector "
                " fields."
            )

    def compute_loss(
        self,
        input_data: CoordDataContainer,
        output_data: CoordDataContainer,
    ) -> torch.Tensor:
        f_residual = self.compute_residual(output_data)
        return self._compute_weighted_loss(f_residual=f_residual)


# Mapping from loss term labels to classes
LOSS_TERM_REGISTRY: dict[LossTermLabels, type[LossTerm]] = {
    LossTermLabels.DATA: DataLoss,
    LossTermLabels.PHYSICS: PhysicsLoss,
}
