from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ..differential_equations.operators import Operators
from ..io.input_equations import EQUATION_REGISTRY
from .loss_term import LossTerm

if TYPE_CHECKING:
    from typing import Any

    from ..geometry.coordinate_system import CoordSystem
    from ..nn.neural_network import NeuralNetwork


class LossTermCoordGroup:
    """Grouping of loss terms based on the coords that they're evaluated at"""

    def __init__(
        self,
        coord_system: CoordSystem,
        coords: dict[str, torch.Tensor],
        geometric_axes_labels: tuple[str, ...],
        index_to_coord_dict: dict[int, str],
        device: torch.device,
    ):
        # Init as empty group
        self.loss_terms: list[LossTerm] = []

        # Store coords and operators
        self.coords = coords
        self.device = device
        self.operators = Operators(
            coord_system,
            coords,
            geometric_axes_labels,
            device,
        )

        # Make tensors for model inputs
        coords_list = []

        for i in range(len(coords)):
            coord_label = index_to_coord_dict[i]
            coords_list.append(coords[coord_label])

        # Make tensor
        self.coords_tensor = torch.stack(coords_list, dim=1)

    def add(self, loss_term: LossTerm) -> None:
        """Add a specific loss term to the group."""
        self.loss_terms.append(loss_term)

    def remove(self, loss_term: LossTerm) -> None:
        """Remove a specific loss term from the group."""
        self.loss_terms.remove(loss_term)

    def equations_needed(self) -> tuple[set[str], set[str]]:
        """All equations needed to evaluate loss terms in group"""
        input_eqns_needed = set()
        output_eqns_needed = set()
        for loss_term in self.loss_terms:
            input_eqns_needed.update(loss_term.input_eqns_needed)
            output_eqns_needed.update(loss_term.output_eqns_needed)

        return input_eqns_needed, output_eqns_needed

    def _evaluate_eqns_loop(
        self,
        eqns_to_calculate: set[str],
        available_params: dict[str, torch.Tensor],
        equations_by_label: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        """
        Loop to evaluate equations until all outputs are computed.

        Parameters
        ----------
        eqns_to_calculate : set[str]
            Set of equation labels to evaluate.
        available_params : dict[str, torch.Tensor]
            Dictionary containing available parameters.
        equations_by_label : dict[str, str]
            Dictionary containing information about each equation in the parameter file.

        Returns
        -------
        derived_fields : dict[str, torch.Tensor]
            Dictionary of all derived outputs from evaluated equations.
        """
        derived_fields: dict[str, torch.Tensor] = {}
        eqns_remaining = eqns_to_calculate.copy()

        # Loop until all equations have been evaluated
        while eqns_remaining:
            evaluated_new = set()
            for eqn in list(eqns_remaining):
                eqn_input_labels = equations_by_label[eqn]["input_args"]
                eqn_output_labels = equations_by_label[eqn]["output_args"]
                eqn_inputs = []

                # Check whether we have access to all input args of eqn
                can_evaluate = True
                for label in eqn_input_labels:
                    if label in available_params:
                        eqn_inputs.append(available_params[label])
                    elif label in derived_fields:
                        eqn_inputs.append(derived_fields[label])
                    else:
                        can_evaluate = False
                        break

                if can_evaluate:
                    # Evaluate eqn and store all output values
                    eqn_outputs = EQUATION_REGISTRY[eqn](*eqn_inputs, self.operators)
                    for i, eqn_output in enumerate(eqn_outputs):
                        derived_fields[eqn_output_labels[i]] = eqn_output
                    evaluated_new.add(eqn)

            # Remove any evaluated eqns from the set of eqns left to evaluate
            eqns_remaining -= evaluated_new

            # Raise error if we loop through all equations and didn't evaluate a new
            # equation and there are still more equations to evaluate.
            if not evaluated_new:
                raise KeyError(
                    f"Unable to evaluate equations {eqns_remaining} for "
                    f"{self.__class__.__name__}."
                )

        return derived_fields

    def apply_model(
        self,
        model: NeuralNetwork,
        equations_by_label: dict[str, Any],
        output_field_labels: set[str],
        input_fields: dict[str, torch.Tensor],
        input_consts: dict[str, torch.Tensor],
        learnt_consts: dict[str, torch.Tensor],
    ) -> None:
        """Applies NN model and evaluates all equations needed for loss term group"""
        # Get model output
        output_fields_tuple = model(self.coords_tensor)

        # Make dict of model output consistently assigning output value to sorted label
        sorted_output_field_labels = sorted(output_field_labels)
        output_fields = {}
        for i, label in enumerate(sorted_output_field_labels):
            output_fields[label] = output_fields_tuple[:, i]

        # Store parameters
        self.output_fields = output_fields
        self.input_fields = input_fields
        self.input_consts = input_consts
        self.learnt_consts = learnt_consts

        # Parameters that can be used in input equations
        input_available_params = input_fields | input_consts | self.coords

        # Parameters that can be used in output equations
        output_available_params = (
            output_fields | input_consts | learnt_consts | self.coords
        )

        # Work out what equations need to be evaluated
        input_eqns_needed, output_eqns_needed = self.equations_needed()

        # Get derived fields
        self.derived_input_fields = self._evaluate_eqns_loop(
            input_eqns_needed, input_available_params, equations_by_label
        )
        self.derived_output_fields = self._evaluate_eqns_loop(
            output_eqns_needed, output_available_params, equations_by_label
        )

    def compute_loss(self) -> torch.Tensor:
        loss = torch.tensor(0.0, device=self.device)
        for loss_term in self.loss_terms:
            loss += loss_term.compute_loss(
                self.input_fields,
                self.output_fields,
                self.derived_input_fields,
                self.derived_output_fields,
            )

        return loss
