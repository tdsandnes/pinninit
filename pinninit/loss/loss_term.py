from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING

import torch

from .loss_function import LOSS_FUNCTION_REGISTRY, LossFunctionLabels
from .loss_utils import is_loss_input_residual

if TYPE_CHECKING:
    from ..io.input_parameters import EquationsParams


class LossTermLabels(str, Enum):
    """Enumeration of loss term labels."""

    DATA = "data"
    PHYSICS = "physics"


class LossTerm(ABC):
    """Base class for loss terms"""

    input_eqns_needed: set[str]
    output_eqns_needed: set[str]

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

        # Init the sets of equations needed for loss term group
        self.input_eqns_needed = set()
        self.output_eqns_needed = set()

    @abstractmethod
    def compute_loss(
        self,
        input_fields: dict[str, torch.Tensor],
        output_fields: dict[str, torch.Tensor],
        derived_input_fields: dict[str, torch.Tensor],
        derived_output_fields: dict[str, torch.Tensor],
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
        input_field_labels: set[str],
        output_field_labels: set[str],
        derived_input_field_eqn_dict: dict[str, set[str]],
        derived_output_field_eqn_dict: dict[str, set[str]],
    ):
        super().__init__(loss_function, weight_method, weight_value)

        # Extract loss term params
        self.field_label = field_label
        self.field_index = field_index

        # Work out what equations are needed to evaluate the loss term
        self._set_equations_needed(
            input_field_labels,
            output_field_labels,
            derived_input_field_eqn_dict,
            derived_output_field_eqn_dict,
        )

    def _set_equations_needed(
        self,
        input_field_labels: set[str],
        output_field_labels: set[str],
        derived_input_field_eqn_dict: dict[str, set[str]],
        derived_output_field_eqn_dict: dict[str, set[str]],
    ) -> None:
        # For it to be possible to evaluate the data loss term, the field that the loss
        # term aims to compare between input and output must be in both (1) either
        # input_field_labels or derived_input_field_eqn_dict; (2) either
        # output_field_labels or derived_output_field_eqn_dict

        field_label = self.field_label
        derived_input_labels = derived_input_field_eqn_dict.keys()
        derived_output_labels = derived_output_field_eqn_dict.keys()

        # Work out which equations need to be applied to input and output params to be
        # able to evaluate loss term
        input_eqns_needed = set()
        output_eqns_needed = set()
        # Check that field is in input or derived input labels
        if field_label in derived_input_labels:
            input_eqns_needed = derived_input_field_eqn_dict[field_label]

        elif field_label not in input_field_labels:
            raise ValueError(
                f"Data loss field '{field_label}' was not found in either: "
                f"{input_field_labels =} or {derived_input_labels = }."
            )

        # Check that field is in output or derived output labels
        if field_label in derived_output_labels:
            output_eqns_needed = derived_output_field_eqn_dict[field_label]

        elif field_label not in output_field_labels:
            raise ValueError(
                f"Data loss field '{field_label}' was not found in either:"
                f" {output_field_labels =} or {derived_output_labels = } "
            )

        # Set the labels of equations needed to be applied to input and output
        # data before loss term can be calculated
        self.input_eqns_needed = input_eqns_needed
        self.output_eqns_needed = output_eqns_needed

    def compute_target_value(
        self,
        input_fields: dict[str, torch.Tensor],
        derived_input_fields: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if self.field_label in input_fields:
            field = input_fields[self.field_label]
        elif self.field_label in derived_input_fields:
            field = derived_input_fields[self.field_label]
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
        output_fields: dict[str, torch.Tensor],
        derived_output_fields: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if self.field_label in output_fields:
            field = output_fields[self.field_label]
        elif self.field_label in derived_output_fields:
            field = derived_output_fields[self.field_label]
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
        input_fields: dict[str, torch.Tensor],
        output_fields: dict[str, torch.Tensor],
        derived_input_fields: dict[str, torch.Tensor],
        derived_output_fields: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        f_target = self.compute_target_value(input_fields, derived_input_fields)
        f_pred = self.compute_predicted_value(output_fields, derived_output_fields)
        return f_pred - f_target

    def compute_loss(
        self,
        input_fields: dict[str, torch.Tensor],
        output_fields: dict[str, torch.Tensor],
        derived_input_fields: dict[str, torch.Tensor],
        derived_output_fields: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        f_target = self.compute_target_value(input_fields, derived_input_fields)
        f_pred = self.compute_predicted_value(output_fields, derived_output_fields)
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
        output_field_labels: set[str],
        input_const_labels: set[str],
        learnt_const_labels: set[str],
        coord_axes_labels: set[str],
        derived_output_field_eqn_dict: dict[str, set[str]],
        equations_params: EquationsParams,
    ):
        super().__init__(loss_function, weight_method, weight_value)

        # Extract loss term params
        self.equation_label = equation_label
        self.residual_label = residual_label
        self.residual_index = residual_index

        # Extract equation parameters for loss term equation
        equations_by_label = {eq["label"]: eq for eq in equations_params.equations}
        self.loss_equation = equations_by_label[self.equation_label]
        self.loss_equation_input_args = self.loss_equation["input_args"]
        self.loss_equation_output_args = self.loss_equation["output_args"]

        # Work out what equations are needed to evaluate the loss term
        nonderived_labels = (
            output_field_labels
            | input_const_labels
            | learnt_const_labels
            | coord_axes_labels
        )
        self._set_equations_needed(nonderived_labels, derived_output_field_eqn_dict)

    def _set_equations_needed(
        self,
        nonderived_labels: set[str],
        derived_output_field_eqn_dict: dict[str, set[str]],
    ) -> None:
        # For it to be possible to evaluate the physics loss term, the input args of the
        # equation that the loss term evaluates must be contained within a set of
        # "allowed" non-derived labels and the params that can be derived from these
        # "allowed" params through the use of an equation

        field_labels = self.loss_equation_input_args
        derived_output_labels = derived_output_field_eqn_dict.keys()

        # Work out which equations need to be applied to be able to evaluate loss term
        eqns_needed = {self.equation_label}

        for field_label in field_labels:
            # Check that field is in allowed nonderived_labels or derived output labels
            if field_label in derived_output_labels:
                eqns_needed.update(derived_output_field_eqn_dict[field_label])

            elif field_label not in nonderived_labels:
                raise ValueError(
                    f"Physics loss field '{field_label}' was not found in either:"
                    f"{nonderived_labels =} or {derived_output_labels = } "
                )

        # Set the labels of all equations needed to evaluate physics loss term
        self.input_eqns_needed = set()
        self.output_eqns_needed = eqns_needed

    def compute_residual(
        self,
        output_fields: dict[str, torch.Tensor],
        derived_output_fields: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if self.residual_label in output_fields:
            field = output_fields[self.residual_label]
        elif self.residual_label in derived_output_fields:
            field = derived_output_fields[self.residual_label]
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
        input_fields: dict[str, torch.Tensor],
        output_fields: dict[str, torch.Tensor],
        derived_input_fields: dict[str, torch.Tensor],
        derived_output_fields: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        f_residual = self.compute_residual(output_fields, derived_output_fields)
        return self._compute_weighted_loss(f_residual=f_residual)


# Mapping from loss term labels to classes
LOSS_TERM_REGISTRY: dict[LossTermLabels, type[LossTerm]] = {
    LossTermLabels.DATA: DataLoss,
    LossTermLabels.PHYSICS: PhysicsLoss,
}
