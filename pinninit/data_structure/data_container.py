from typing import Any

import torch

from ..io.input_equations import EQUATION_REGISTRY
from .coords_dataclass import CoordsProtocol
from .data_structure_utils import generate_tensor_component_labels
from .field_dataclass import FieldProtocol, make_field_dataclass


class CoordDataContainer:
    def __init__(
        self,
        coords: CoordsProtocol,
        field_labels_and_ranks: dict[str, int],
        const_labels: tuple[str, ...],
        equations: dict[str, dict[str, Any]],
    ):
        self.coords = coords
        self.device = coords.device
        self.n_points = self.coords.n_points
        self.dim_geometry = self.coords.coord_system.dim_geometry
        self.field_labels_and_ranks = field_labels_and_ranks
        self.field_labels = field_labels_and_ranks.keys()
        self.const_labels = const_labels
        ### Eventually make class in equations/equations.py to
        # standardise equation handling
        self.equations = equations

        # Bool to keep track of whether container has been updated
        # without derived fields having been recalculated
        self.updated_without_recalculation = True

        # Empty dict of fields and field classes
        self.field_dataclasses: dict[int, type[FieldProtocol]] = {}
        self.fields = self._init_fields(self.field_labels_and_ranks)

        # Empty dict of consts
        self.consts: dict[str, torch.Tensor] = {}

        # Work out what other fields can be derived from the equations supplied
        self._set_derivable_fields()
        ### Comment this out. Temp for now jus treat derivable fields as tensors
        ###self.derivable_fields =
        # ###self._init_fields(self.derivable_field_labels_and_ranks)
        self.derivable_fields: dict[str, torch.Tensor] = {}

    def _init_fields(
        self, field_labels_and_ranks: dict[str, int]
    ) -> dict[str, FieldProtocol]:
        fields = {}
        # Loop through all fields in field labels
        for field_label, rank in field_labels_and_ranks.items():
            # Get or create Field dataclass for this rank
            Field = self.field_dataclasses.setdefault(
                rank,
                make_field_dataclass(rank=rank, coords=self.coords, device=self.device),
            )

            # Components must correspond to those expected from rank and coords
            expected_labels = generate_tensor_component_labels(
                self.coords.geometric_labels, rank
            )
            init_field_dict = {
                label: torch.zeros(self.coords.n_points, device=self.device)
                for label in expected_labels
            }

            # Init field and add to dict
            fields[field_label] = Field(**init_field_dict)

        return fields

    def update_field(self, field_label: str, field_dict: dict[str, Any]) -> None:
        # Field must be in available labels
        if field_label not in self.field_labels_and_ranks:
            raise ValueError(
                f"Invalid field label {field_label} provided. "
                f"Available options are {self.field_labels_and_ranks.keys()}."
            )

        # Update field with input field dict=
        self.fields[field_label].update(field_dict)
        self.updated_without_recalculation = True

    def update_const(self, const_label: str, value: torch.Tensor) -> None:
        # Const must be in available labels
        if const_label not in self.const_labels:
            raise ValueError(
                f"Invalid const label {const_label} provided. "
                f"Available options are {self.const_labels}."
            )

        self.consts[const_label] = value
        self.updated_without_recalculation = True

    def _set_derivable_fields(self) -> None:
        """From the equations, gets all new parameters that can be calculated
        from a list of other parameters (based on their labels), or quantities that can
        be calculated from them, through other equations."""
        # Labels of all coords, fields, and consts
        all_labels = (
            set(self.coords.coord_labels)
            | set(self.fields.keys())
            | set(self.consts.keys())
        )

        # Start from empty dict
        derivable_fields: dict[str, int] = {}
        equations_needed: set[str] = set()

        # Loop through all equations and see if any satisfy conditions.
        # If new params have been added, it's possible that other equations are
        # now available. Therefore keep looping until no new params are added.
        added_new = True
        while added_new:
            added_new = False
            # Loop through all equations and see if any satisfy conditions
            for equation_label, equation in self.equations.items():
                eqn_input_labels = equation["input_args"]
                eqn_output_labels_and_ranks = equation["output_args"]

                # If all input args satisfy conditions
                if all(arg in all_labels for arg in eqn_input_labels):
                    # Add equation to the list of equations needed
                    equations_needed.add(equation_label)

                    # Add all output args to derived_dict...
                    for output_arg in eqn_output_labels_and_ranks:
                        # ... if they're not already in all_labels
                        if output_arg not in all_labels:
                            # Add entry to dict
                            rank = eqn_output_labels_and_ranks[output_arg]
                            derivable_fields[output_arg] = rank

                            # Update all_labels
                            all_labels.add(output_arg)

                            # Added at least one new value, so pass through outer
                            # for loop at least one more time
                            added_new = True

        self.derivable_field_labels_and_ranks = derivable_fields
        self.equations_needed = equations_needed

    def evaluate_equations(self) -> None:
        """
        Loop to evaluate equations until all outputs are computed.
        """
        # Labels of all coords, fields, and consts
        all_labels = (
            set(self.coords.coord_labels)
            | set(self.fields.keys())
            | set(self.consts.keys())
        )

        eqns_remaining = self.equations_needed.copy()

        # Loop until all equations have been evaluated
        while eqns_remaining:
            evaluated_new = set()
            for eqn in list(eqns_remaining):
                eqn_input_labels = self.equations[eqn]["input_args"]
                eqn_output_labels = list(self.equations[eqn]["output_args"].keys())

                # If all input args satisfy conditions
                if all(arg in all_labels for arg in eqn_input_labels):
                    eqn_inputs: list[Any] = []
                    # Check whether we have access to all input args of eqn
                    for label in eqn_input_labels:
                        if label in self.coords.coord_labels:
                            eqn_inputs.append(torch.min(getattr(self.coords, label)))
                        elif label in self.fields:
                            field = self.fields[label]
                            eqn_inputs.append(field.tensor)
                        elif label in self.consts:
                            eqn_inputs.append(self.consts[label])
                        elif label in self.derivable_fields:
                            ###field = self.derivable_fields[label]
                            ###eqn_inputs.append(field.tensor)
                            eqn_inputs.append(self.derivable_fields[label])
                        else:
                            raise KeyError(
                                "Not all params are available to evaluate equation."
                            )

                    # Evaluate eqn and store all output values
                    eqn_outputs = EQUATION_REGISTRY[eqn](
                        *eqn_inputs, self.coords.operators
                    )

                    ###if len(eqn_output_labels) == 1:
                    ####    label = eqn_output_labels[0]
                    ###    self.update_field(label, eqn_outputs)

                    # Update all_labels
                    ###    all_labels.add(label)

                    ###else:
                    ###    for i, eqn_output in enumerate(eqn_outputs):
                    ###        label = eqn_output_labels[i]
                    ###        self.update_field(label, eqn_output)

                    # Update all_labels
                    ###        all_labels.add(label)

                    ### Do this instead for now
                    if len(eqn_output_labels) == 1:
                        self.derivable_fields[eqn_output_labels[0]] = eqn_outputs

                        all_labels.add(eqn_output_labels[0])
                    else:
                        for i, eqn_output in enumerate(eqn_outputs):
                            self.derivable_fields[eqn_output_labels[i]] = eqn_output
                            all_labels.add(eqn_output_labels[i])

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
