from collections.abc import Callable
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
        available_fields_params: FieldsParams,
        available_equations_params: EquationsParams,
    ):
        self.coords = coords
        self.device = coords.device
        self.n_points = self.coords.n_points
        self.dim_geometry = self.coords.coord_system.dim_geometry

        # Total available fields that can be derived
        self.available_fields_params = available_fields_params
        # and the equations that can be used to derive fields
        self.available_equations_params = available_equations_params

        all_available_input_fields = self.available_fields_params.input_fields
        all_available_output_fields = self.available_fields_params.output_fields
        all_available_derivable_fields = self.available_fields_params.derivable_fields
        all_available_fields = (
            all_available_input_fields
            + all_available_output_fields
            + all_available_derivable_fields
        )
        self.field_label_to_rank = {
            field["label"]: field["rank"] for field in all_available_fields
        }

        # Empty dict of fields and field classes
        # Note that _fields can be internally inconsistent, but fields property cannot
        self._fields: dict[str, FieldProtocol] = {}
        self.field_dataclasses: dict[int, type[FieldProtocol]] = {}

        # Labels of derived fields and bool to keep track of if these are consistent
        # with other fields and consts that might have recently been updated/added
        self.derived_field_labels: set[str] = set()
        self.internal_consistency = True

        # Initialise consts
        self.consts: dict[str, torch.Tensor] = {}

    @property
    def fields(self) -> dict[str, FieldProtocol]:
        # Ensure derived fields are up-to-date before returning
        self._ensure_consistency()
        return self._fields

    def get_field(self, field_label: str) -> FieldProtocol:
        # Ensure derived fields are up-to-date before returning
        self._ensure_consistency()
        return self._get_field(field_label)

    def _ensure_consistency(self) -> None:
        if not self.internal_consistency:
            self._set_derived_fields()

    def _init_empty_field(self, field_label: str, rank: int) -> None:
        if field_label in self._fields:
            raise ValueError(
                f"Cannot initialise field {field_label}, which already exists in "
                "data container."
            )

        # Get or create Field dataclass for this rank
        Field = self.field_dataclasses.setdefault(
            rank,
            make_field_dataclass(rank=rank, coords=self.coords, device=self.device),
        )

        # Components must correspond to those expected from rank and coords
        labels, _ = generate_tensor_component_labels(self.coords.geometric_labels, rank)
        init_field_dict = {
            label: torch.zeros(self.coords.n_points, device=self.device)
            for label in labels
        }

        self._fields[field_label] = Field(**init_field_dict)

    def _get_field(self, field_label: str) -> FieldProtocol:
        if field_label in self._fields:
            return self._fields[field_label]
        else:
            raise KeyError(
                f"Field label {field_label} not found in fields {self._fields.keys()}."
            )

    def _set_field(
        self, field_label: str, update: Callable[[FieldProtocol], None]
    ) -> None:
        # Field must in available labels
        if field_label not in self.field_label_to_rank:
            raise ValueError(
                f"Invalid field label {field_label} provided. "
                f"Available options are {self.field_label_to_rank.keys()}."
            )

        # Init field if it hasn't already been initialised
        if field_label not in self._fields:
            rank = self.field_label_to_rank[field_label]
            self._init_empty_field(field_label, rank)

        # Update field
        update(self._fields[field_label])
        self.internal_consistency = False

    def set_field_from_component_dict(
        self, field_label: str, field_component_dict: dict[str, Any]
    ) -> None:
        self._set_field(
            field_label, lambda f: f.update_from_component_dict(field_component_dict)
        )

    def set_field_from_component_tensor(
        self, field_label: str, field_component_tensor: torch.Tensor
    ) -> None:
        self._set_field(
            field_label,
            lambda f: f.update_from_component_tensor(field_component_tensor),
        )

    def set_field_from_tensor(
        self, field_label: str, field_tensor: torch.Tensor
    ) -> None:
        self._set_field(field_label, lambda f: f.update_from_tensor(field_tensor))

    def set_const(self, const_label: str, value: torch.Tensor) -> None:
        # Const must be in available labels
        if const_label not in self.consts:
            raise ValueError(
                f"Invalid const label {const_label} provided. "
                f"Available options are {self.consts.keys()}."
            )

        self.consts[const_label] = value
        self.internal_consistency = False

    def _set_derived_fields(self) -> None:
        """
        Loop to evaluate equations until all outputs are computed.
        """

        # Labels of all coords, fields, and consts.
        # Not including derived fields, which we are recalculating here.
        base_field_labels = set(self._fields.keys()) - self.derived_field_labels
        currently_available_labels = (
            set(self.coords.coord_labels) | base_field_labels | set(self.consts.keys())
        )

        available_equations = self.available_equations_params.equations
        eqn_dict = {eq["label"]: eq for eq in available_equations}
        eqns_remaining = set(eqn_dict.keys())

        preexisting_derived_fields_remaining = self.derived_field_labels.copy()

        # Loop through all equations and see if any satisfy conditions.
        # If new params have been added, it's possible that other equations are
        # now available. Therefore keep looping until no new params are added.
        added_new_field = True
        equations_used = set()
        while eqns_remaining and added_new_field:
            added_new_field = False
            equations_evaluated_this_round = set()

            for eqn_label in list(eqns_remaining):
                equation = eqn_dict[eqn_label]
                eqn_input_labels = equation["input_args"]
                eqn_output_labels = equation["output_args"]

                # Only proceed if all input args are currently available
                if not all(
                    arg in currently_available_labels for arg in eqn_input_labels
                ):
                    continue  # skip to next equation

                # Get all input args of eqn
                eqn_inputs: list[Any] = []
                for label in eqn_input_labels:
                    if label in self.coords.coord_labels:
                        eqn_inputs.append(getattr(self.coords, label))
                    elif label in self._fields:
                        field = self._fields[label]
                        eqn_inputs.append(field.tensor)
                    elif label in self.consts:
                        eqn_inputs.append(self.consts[label])
                    else:
                        raise KeyError(
                            "Not all params are available to evaluate equation."
                        )

                # Evaluate equation
                eqn_outputs = EQUATION_REGISTRY[eqn_label](
                    *eqn_inputs, self.coords.operators
                )

                # Make single outputs into tuple for easier looping through outputs
                eqn_outputs = (
                    eqn_outputs if isinstance(eqn_outputs, tuple) else (eqn_outputs,)
                )

                # Make sure number of outputs is as expected
                if len(eqn_outputs) != len(eqn_output_labels):
                    raise ValueError(
                        f"The number of equation outputs, "
                        f" {len(eqn_outputs)}, doesn't match the expected number, "
                        f"{len(eqn_output_labels)}, for equation {eqn_label}."
                    )

                # Loop through equation outputs
                for label, eqn_output in zip(
                    eqn_output_labels, eqn_outputs, strict=True
                ):
                    # Update field if it is already in container
                    if label in self.derived_field_labels:
                        self._fields[label].update_from_tensor(eqn_output)
                        preexisting_derived_fields_remaining.discard(label)

                    # else init the field and set it
                    else:
                        rank = self.field_label_to_rank[label]
                        self._init_empty_field(label, rank)
                        self._fields[label].update_from_tensor(eqn_output)
                        self.derived_field_labels.add(label)

                    currently_available_labels.add(label)

                added_new_field = True
                equations_evaluated_this_round.add(eqn_label)

            # Remove any evaluated eqns from the set of eqns left to evaluate
            eqns_remaining -= equations_evaluated_this_round
            equations_used.update(equations_evaluated_this_round)

        # Store equations used
        self.equations_used = equations_used

        # If any pre-existing fields haven't been updated, remove the field from the
        # container for internal consistency
        for field_label in preexisting_derived_fields_remaining:
            del self._fields[field_label]
            self.derived_field_labels.remove(field_label)

        # We now have internal consistency
        self.internal_consistency = True
