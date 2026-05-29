from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from ..io.input_equations import EQUATION_REGISTRY
from ..io.input_parameters import EquationsParams, FieldsParams
from .coords_dataclass import CoordsProtocol
from .field_dataclass import (
    FieldProtocol,
    generate_tensor_component_labels,
    make_field_dataclass,
)


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
        self._derived_field_labels: set[str] = set()
        self.internal_consistency = True
        self.field_dataclasses: dict[int, type[FieldProtocol]] = {}

        # Initialise consts
        self.consts: dict[str, torch.Tensor] = {}

        # Fields that will be accessed (e.g., by loss terms)
        self._required_fields: set[str] = set()

        # Output to equation lookup dict
        self._output_to_equation: dict[str, dict[str, Any]] = {}
        for eq in self.available_equations_params.equations:
            for output in eq["output_args"]:
                self._output_to_equation[output] = eq

    def get_field(self, field_label: str) -> FieldProtocol:
        # If field doesn't exist or if it's derived and dirty, compute it
        if field_label not in self._fields or (
            field_label in self._derived_field_labels and not self.internal_consistency
        ):
            self._set_derived_fields(required_fields=self._required_fields or None)

        return self._get_field(field_label)

    def _init_empty_field(self, field_label: str, rank: int) -> None:
        if field_label in self._fields:
            raise ValueError(
                f"Cannot initialise field {field_label}, which already exists."
            )

        # Get or create Field dataclass for this rank
        Field = self.field_dataclasses.setdefault(
            rank,
            make_field_dataclass(rank=rank, coords=self.coords, device=self.device),
        )

        # Components use coords.n_points which respects batching
        labels = generate_tensor_component_labels(self.coords.geometric_labels, rank)
        init_field_dict = {
            label: torch.zeros(self.coords.n_points, device=self.device)
            for label in labels
        }

        self._fields[field_label] = Field(**init_field_dict)

    def _get_field(self, label: str) -> FieldProtocol:
        """Get a field by label."""
        if label not in self._fields:
            raise KeyError(f"Field {label} not found.")
        return self._fields[label]

    def _clear_derived_fields(self) -> None:
        for label in self._derived_field_labels:
            self._fields.pop(label, None)
        self._derived_field_labels.clear()

    def _set_field(
        self, field_label: str, update: Callable[[FieldProtocol], None]
    ) -> None:
        # Field must be in available labels
        if field_label not in self.field_label_to_rank:
            raise ValueError(
                f"Invalid field label {field_label} provided. "
                f"Available options are {list(self.field_label_to_rank.keys())}."
            )

        # Init field if it hasn't already been initialised
        if field_label not in self._fields:
            rank = self.field_label_to_rank[field_label]
            self._init_empty_field(field_label, rank)

        # Update field
        update(self._fields[field_label])
        if field_label not in self._derived_field_labels:
            self.internal_consistency = False

    def set_field_from_component_dict(
        self, field_label: str, field_component_dict: dict[str, Any]
    ) -> None:
        # Clear derived fields that depend on this field
        if field_label not in self._derived_field_labels:
            self._clear_derived_fields()

        self._set_field(
            field_label, lambda f: f.update_from_component_dict(field_component_dict)
        )

    def set_field_from_component_tensor(
        self, field_label: str, field_component_tensor: torch.Tensor
    ) -> None:
        # Clear derived fields that depend on this field
        if field_label not in self._derived_field_labels:
            self._clear_derived_fields()

        self._set_field(
            field_label,
            lambda f: f.update_from_component_tensor(field_component_tensor),
        )

    def set_field_from_tensor(
        self, field_label: str, field_tensor: torch.Tensor
    ) -> None:
        # Clear derived fields that depend on this field
        if field_label not in self._derived_field_labels:
            self._clear_derived_fields()

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

    def set_batch(self, indices: slice | torch.Tensor) -> None:
        """Set batch view."""
        self.coords.set_batch(indices)

        # Clear derived fields (need recomputation with new batch)
        self._clear_derived_fields()

        # Remove any batch-only fields that don't match current batch
        fields_to_remove = []
        for label, field in self._fields.items():
            if field.is_batch_only:
                fields_to_remove.append(label)

        for label in fields_to_remove:
            self._fields.pop(label)

        self.internal_consistency = False

    def clear_batch(self) -> None:
        """Clear batch view."""
        self.coords.clear_batch()

        # Clear all batch-only fields
        fields_to_remove = []
        for label, field in self._fields.items():
            if field.is_batch_only:
                fields_to_remove.append(label)

        for label in fields_to_remove:
            self._fields.pop(label)

        self._derived_field_labels.clear()
        self.internal_consistency = False

    def register_required_field(self, field_label: str) -> None:
        """Register a field that will be accessed (e.g., by loss terms)."""
        self._required_fields.add(field_label)

    def _set_derived_fields(self, required_fields: set[str] | None = None) -> None:
        """Loop to evaluate equations until required outputs are computed.

        Args:
            required_fields: If provided, only compute these specific derived fields
                            and their dependencies. If None, compute all possible.
        """
        # Clear previously derived fields
        self._clear_derived_fields()

        # Labels of all coords, fields, and consts
        currently_available_labels = (
            set(self.coords.coord_labels)
            | set(self._fields.keys())
            | set(self.consts.keys())
        )

        if required_fields is None:
            required_fields = {
                output
                for eq in self.available_equations_params.equations
                for output in eq["output_args"]
            }

        # Resolve all fields we need to compute (including intermediates)
        fields_to_compute = self._resolve_dependencies(
            required_fields, currently_available_labels
        )
        fields_still_needed = fields_to_compute - currently_available_labels

        available_equations = self.available_equations_params.equations
        eqn_dict = {eq["label"]: eq for eq in available_equations}
        eqns_remaining = set(eqn_dict.keys())

        # Temporary storage for intermediate fields
        local_derived_fields: dict[str, torch.Tensor] = {}

        added_new_field = True
        equations_used = set()
        while eqns_remaining and added_new_field and fields_still_needed:
            added_new_field = False
            equations_evaluated_this_round = set()

            for eqn_label in list(eqns_remaining):
                equation = eqn_dict[eqn_label]
                eqn_input_labels = equation["input_args"]
                eqn_output_labels = equation["output_args"]

                # Skip if this equation doesn't produce any needed fields
                if not any(
                    output in fields_still_needed for output in eqn_output_labels
                ):
                    continue

                # Only proceed if all input args are currently available
                if not all(
                    arg in currently_available_labels for arg in eqn_input_labels
                ):
                    continue

                # Get all input args of eqn
                eqn_inputs: list[Any] = []
                for label in eqn_input_labels:
                    if label in self.coords.coord_labels:
                        eqn_inputs.append(self.coords.get_coord(label))
                    elif label in self._fields:
                        eqn_inputs.append(self._fields[label].tensor)
                    elif label in local_derived_fields:
                        eqn_inputs.append(local_derived_fields[label])
                    elif label in self.consts:
                        eqn_inputs.append(self.consts[label])
                    else:
                        raise KeyError(
                            f"Input {label} not available for equation {eqn_label}"
                        )

                # Evaluate equation
                eqn_outputs = EQUATION_REGISTRY[eqn_label](
                    *eqn_inputs, self.coords.operators
                )
                eqn_outputs = (
                    eqn_outputs if isinstance(eqn_outputs, tuple) else (eqn_outputs,)
                )

                # Validate number of outputs
                if len(eqn_outputs) != len(eqn_output_labels):
                    raise ValueError(
                        f"Equation {eqn_label} returned {len(eqn_outputs)} outputs, "
                        f"expected {len(eqn_output_labels)}"
                    )

                # Store outputs
                for label, eqn_output in zip(
                    eqn_output_labels, eqn_outputs, strict=True
                ):
                    if label not in self._fields and label in required_fields:
                        rank = self.field_label_to_rank[label]
                        self._init_empty_field(label, rank)
                        self._fields[label].update_from_tensor(eqn_output)
                        self._derived_field_labels.add(label)
                        currently_available_labels.add(label)
                        fields_still_needed.discard(label)
                    elif label not in self._fields:
                        local_derived_fields[label] = eqn_output
                        currently_available_labels.add(label)
                        fields_still_needed.discard(label)

                added_new_field = True
                equations_evaluated_this_round.add(eqn_label)

            eqns_remaining -= equations_evaluated_this_round
            equations_used.update(equations_evaluated_this_round)

        self.equations_used = equations_used
        self.internal_consistency = True

    def _resolve_dependencies(
        self, required_fields: set[str], available_labels: set[str]
    ) -> set[str]:
        """Resolve all dependencies needed to compute required fields."""
        fields_to_compute = set()
        fields_to_check = list(required_fields)

        # Iteratively find dependencies
        while fields_to_check:
            field = fields_to_check.pop()

            # Skip if already available or already marked for computation
            if field in available_labels or field in fields_to_compute:
                continue

            # Check if this field can be derived
            if field not in self._output_to_equation:
                # Field cannot be derived, skip
                continue

            # Mark this field for computation
            fields_to_compute.add(field)

            # Add its dependencies to check
            equation = self._output_to_equation[field]
            for input_field in equation["input_args"]:
                if input_field not in available_labels:
                    fields_to_check.append(input_field)

        return fields_to_compute
