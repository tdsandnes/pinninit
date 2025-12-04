from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

import torch

from .loss_function import LOSS_FUNCTION_REGISTRY, LossFunctionLabels
from .loss_utils import is_loss_input_residual

if TYPE_CHECKING:
    from typing import Any

    from ..io.input_parameters import EquationsParams


class LossTermLabels(str, Enum):
    """Enumeration of loss term labels."""

    DATA = "data"
    PHYSICS = "physics"


class LossTerm:
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
        ### loss_weights.py file where we can add various constant and adaptive methods.
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

    def __init__(
        self,
        data_loss_term: dict[str, Any],
        input_field_labels: set[str],
        output_field_labels: set[str],
        derived_input_field_eqn_dict: dict[str, set[str]],
        derived_output_field_eqn_dict: dict[str, set[str]],
    ):
        super().__init__(
            data_loss_term["loss_function"],
            data_loss_term["weight_method"],
            data_loss_term["weight_value"],
        )

        # Extract loss term params
        self.label = data_loss_term["field_label"]

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

        field_label = self.label
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

    def __init__(
        self,
        physics_loss_term: dict[str, Any],
        output_field_labels: set[str],
        input_const_labels: set[str],
        learnt_const_labels: set[str],
        coord_axes_labels: set[str],
        derived_output_field_eqn_dict: dict[str, set[str]],
        equations_params: EquationsParams,
    ):
        super().__init__(
            physics_loss_term["loss_function"],
            physics_loss_term["weight_method"],
            physics_loss_term["weight_value"],
        )

        # Extract loss term params
        self.label = physics_loss_term["label"]
        self.equation_label = physics_loss_term["equation_label"]

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
        eqns_needed = set(self.equation_label)

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
        self.eqns_needed = eqns_needed

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
LOSS_TERM_REGISTRY: dict[LossTermLabels, type[LossTerm]] = {
    LossTermLabels.DATA: DataLoss,
    LossTermLabels.PHYSICS: PhysicsLoss,
}
