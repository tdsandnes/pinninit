from typing import Any

import torch

from ..io.input_equations import EQUATION_REGISTRY
from ..io.input_parameters import EquationsParams, FieldsParams
from .coords_dataclass import CoordsProtocol
from .data_structure_utils import generate_tensor_component_labels
from .field_dataclass import FieldProtocol, make_field_dataclass


class CoordDataContainer:
    def __init__(
        self,
        coords: CoordsProtocol,
        fields_params: list[dict[str, Any]],
        const_labels: tuple[str, ...],
        available_fields_params: FieldsParams,
        available_equations_params: EquationsParams,
    ):
        self.coords = coords
        self.device = coords.device
        self.n_points = self.coords.n_points
        self.dim_geometry = self.coords.coord_system.dim_geometry
        self.fields_params = fields_params
        self.field_labels = {field["label"] for field in self.fields_params}
        self.const_labels = const_labels

        # Total available fields that can be derived
        self.available_fields_params = available_fields_params
        # and the equations that can be used to derive fields
        self.available_equations_params = available_equations_params

        # Empty dict of fields and field classes
        self.field_dataclasses: dict[int, type[FieldProtocol]] = {}
        self.fields = self._init_fields(self.fields_params)

        # Empty dict of consts
        self.consts: dict[str, torch.Tensor] = {}

        ### Comment this out. Temp for now jus treat derivable fields as tensors
        ###self.derivable_fields =
        # ###self._init_fields(self.derivable_field_labels_and_ranks)
        self.derivable_fields: dict[str, torch.Tensor] = {}

    def _init_fields(
        self, fields_params: list[dict[str, Any]]
    ) -> dict[str, FieldProtocol]:
        fields = {}
        # Loop through all fields
        for field in fields_params:
            field_label = field["label"]
            rank = field["rank"]
            # Get or create Field dataclass for this rank
            Field = self.field_dataclasses.setdefault(
                rank,
                make_field_dataclass(rank=rank, coords=self.coords, device=self.device),
            )

            # Components must correspond to those expected from rank and coords
            expected_components = generate_tensor_component_labels(
                self.coords.geometric_labels, rank
            )
            init_field_dict = {
                label: torch.zeros(self.coords.n_points, device=self.device)
                for label in expected_components
            }

            # Init field and add to dict
            fields[field_label] = Field(**init_field_dict)

        return fields

    def update_field(self, field_label: str, field_dict: dict[str, Any]) -> None:
        # Field must be in available labels

        if field_label not in self.field_labels:
            raise ValueError(
                f"Invalid field label {field_label} provided. "
                f"Available options are {self.field_labels}."
            )

        # Update field with input field dict=
        self.fields[field_label].update(field_dict)

    def update_const(self, const_label: str, value: torch.Tensor) -> None:
        # Const must be in available labels
        if const_label not in self.const_labels:
            raise ValueError(
                f"Invalid const label {const_label} provided. "
                f"Available options are {self.const_labels}."
            )

        self.consts[const_label] = value

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
        # dict to look up field from label
        fields = (
            self.available_fields_params.input_fields
            + self.available_fields_params.output_fields
            + self.available_fields_params.derivable_fields
        )
        fields_dict = {f["label"]: f for f in fields}

        # Start from empty dict
        derivable_fields_params = []
        equations = []

        # Loop through all equations and see if any satisfy conditions.
        # If new params have been added, it's possible that other equations are
        # now available. Therefore keep looping until no new params are added.
        added_new = True
        while added_new:
            added_new = False
            # Loop through all equations and see if any satisfy conditions
            for equation in self.available_equations_params.equations:
                eqn_input_labels = equation["input_args"]
                eqn_output_labels = equation["output_args"]

                # Only proceed if all input args are satisfied
                if not all(arg in all_labels for arg in eqn_input_labels):
                    continue  # skip to next equation

                output_arg_added = False

                # Add all output args to derived_dict...
                for output_arg in eqn_output_labels:
                    # ... if they're not already in all_labels
                    if output_arg in all_labels:
                        continue  # skip to next output arg

                    # Add output arg to derivable fields if it's in the available fields
                    if output_arg in fields_dict:
                        field = fields_dict[output_arg]
                        derivable_fields_params.append(field)
                        output_arg_added = True

                        # Update set of field labels
                        self.field_labels.add(output_arg)

                        # Update all_labels
                        all_labels.add(output_arg)

                        # Added at least one new value, so pass through outer
                        # for loop at least one more time
                        added_new = True
                    else:
                        break  # skip to next equation

                if output_arg_added:
                    # Add equation to the list of equations needed, if any output
                    # args were new
                    equations.append(equation)

        self.derivable_fields_params = derivable_fields_params
        ### This would be the places to init derived fields as Fields
        self.equations = equations

    def evaluate_equations(self) -> None:
        """
        Loop to evaluate equations until all outputs are computed.
        """
        self._set_derivable_fields()

        # Labels of all coords, fields, and consts
        all_labels = (
            set(self.coords.coord_labels)
            | set(self.fields.keys())
            | set(self.consts.keys())
        )

        eqn_dict = {eq["label"]: eq for eq in self.equations}
        eqns_remaining = set(eqn_dict.keys())

        # Loop until all equations have been evaluated
        while eqns_remaining:
            evaluated_new = set()
            for eqn_label in list(eqns_remaining):
                equation = eqn_dict[eqn_label]
                eqn_input_labels = equation["input_args"]
                eqn_output_labels = equation["output_args"]

                # Only proceed if all input args are satisfied
                if not all(arg in all_labels for arg in eqn_input_labels):
                    continue  # skip to next equation

                eqn_inputs: list[Any] = []
                # Check whether we have access to all input args of eqn
                for label in eqn_input_labels:
                    if label in self.coords.coord_labels:
                        eqn_inputs.append(getattr(self.coords, label))
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
                eqn_outputs = EQUATION_REGISTRY[eqn_label](
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

                evaluated_new.add(eqn_label)

            # Remove any evaluated eqns from the set of eqns left to evaluate
            eqns_remaining -= evaluated_new

            # Raise error if we loop through all equations and didn't evaluate a new
            # equation and there are still more equations to evaluate.
            if not evaluated_new:
                raise KeyError(
                    f"Unable to evaluate equations {eqns_remaining} for "
                    f"{self.__class__.__name__}."
                )
