from __future__ import annotations

from collections.abc import Callable

import torch

from .coords_dataclass import Coords
from .equation import Equation
from .equation_evaluator import EquationEvaluator
from .field_dataclass import Field, make_field_dataclass


class DataContainer:
    def __init__(
        self,
        coords: Coords,
        primary_field_label_to_rank: dict[str, int],
        derived_field_label_to_rank: dict[str, int] | None = None,
        const_labels: set[str] | None = None,
        equations: list[Equation] | None = None,
    ) -> None:
        self.coords = coords
        self.coord_labels = coords.coord_labels
        self.device = coords.device
        self.equations = equations or []
        self.equation_evaluator = EquationEvaluator(self)

        # Store field labels
        derived_field_label_to_rank = derived_field_label_to_rank or {}
        self.primary_field_labels = set(primary_field_label_to_rank)
        self.derived_field_labels = set(derived_field_label_to_rank)

        # Init dict of empty fields
        self._field_rank_to_classes: dict[int, type[Field]] = {}
        self.primary_fields: dict[str, Field] = {
            label: self._empty_field_of_rank(label, rank)
            for label, rank in primary_field_label_to_rank.items()
        }
        self.derived_fields: dict[str, Field] = {
            label: self._empty_field_of_rank(label, rank)
            for label, rank in derived_field_label_to_rank.items()
        }

        # Init dict of empty consts
        self.consts: dict[str, torch.Tensor | None] = dict.fromkeys(
            const_labels or set()
        )

        # Keep track of the active set of coords used when updating fields or consts
        self.primary_fields_active_coords: dict[str, Coords | None] = dict.fromkeys(
            self.primary_field_labels, None
        )
        self.derived_fields_active_coords: dict[str, Coords | None] = dict.fromkeys(
            self.derived_field_labels, None
        )
        self.consts_active_coords: dict[str, Coords | None] = dict.fromkeys(
            const_labels or set(), None
        )

        # Cache for accessing fields
        self._fields_cache: dict[str, Field] | None = None
        self._fields_active_coords_cache: dict[str, Coords | None] | None = None

    @property
    def field_labels(self) -> set[str]:
        return self.primary_field_labels | self.derived_field_labels

    @property
    def fields(self) -> dict[str, Field]:
        if self._fields_cache is None:
            self._fields_cache = {**self.primary_fields, **self.derived_fields}
        return self._fields_cache

    @property
    def fields_active_coords(self) -> dict[str, Coords | None]:
        if self._fields_active_coords_cache is None:
            self._fields_active_coords_cache = {
                **self.primary_fields_active_coords,
                **self.derived_fields_active_coords,
            }
        return self._fields_active_coords_cache

    def get_field(self, label: str) -> Field:
        if label not in self.field_labels:
            raise KeyError(f"Field '{label}' not found.")

        return self.fields[label]

    def get_field_tensor(
        self,
        label: str,
        coords: Coords | None = None,
        enforce_ancestry: bool = True,
    ) -> torch.Tensor:
        coords = coords or self.coords
        active_coords = self._get_active_coords(label)
        self._validate_eval_coords(label, active_coords, coords, enforce_ancestry)
        return self.get_field(label).tensor(coords)

    def get_field_component(
        self,
        field_label: str,
        component_label: str,
        coords: Coords | None = None,
        enforce_ancestry: bool = True,
    ) -> torch.Tensor:
        coords = coords or self.coords
        active_coords = self._get_active_coords(field_label)
        self._validate_eval_coords(field_label, active_coords, coords, enforce_ancestry)
        return self.get_field(field_label).get_component(component_label, coords)

    def get_field_component_tensor(
        self,
        label: str,
        coords: Coords | None = None,
        enforce_ancestry: bool = True,
    ) -> torch.Tensor:
        coords = coords or self.coords
        active_coords = self._get_active_coords(label)
        self._validate_eval_coords(label, active_coords, coords, enforce_ancestry)
        return self.get_field(label).component_tensor(coords)

    def get_const(
        self,
        label: str,
        coords: Coords | None = None,
        enforce_ancestry: bool = True,
    ) -> torch.Tensor:
        coords = coords or self.coords
        active_coords = self._get_active_coords(label)
        self._validate_eval_coords(label, active_coords, coords, enforce_ancestry)

        value = self.consts[label]
        if value is None:
            raise RuntimeError(
                f"Constant '{label}' has active coordinates tracked, "
                f"but its tensor value is unexpectedly None."
            )

        return value

    def _set_field(
        self,
        label: str,
        coords: Coords | None,
        allow_derived: bool,
        update: Callable[[Field, Coords], None],
    ) -> None:
        # Guard against setting derived field outside of equation evaluator
        if label in self.derived_field_labels and not allow_derived:
            raise ValueError(
                f"Field '{label}' is a derived field and cannot be set directly."
            )

        # Update field
        coords = coords or self.coords
        self._validate_eval_coords(label, self.coords, coords, enforce_ancestry=True)
        update(self.get_field(label), coords)

        # Update active coords for the field
        if label in self.primary_field_labels:
            self.primary_fields_active_coords[label] = coords
        elif label in self.derived_field_labels:
            self.derived_fields_active_coords[label] = coords

        # Reset field cache
        self._fields_cache = None
        self._fields_active_coords_cache = None

    def set_field_from_tensor(
        self,
        label: str,
        tensor: torch.Tensor,
        coords: Coords | None = None,
        detach_stale_values: bool = True,
        allow_derived: bool = False,
    ) -> None:
        self._set_field(
            label,
            coords,
            allow_derived,
            lambda field, c: field.update_from_tensor(
                tensor, coords=c, detach_stale_values=detach_stale_values
            ),
        )

    def set_field_from_component_tensor(
        self,
        label: str,
        component_tensor: torch.Tensor,
        coords: Coords | None = None,
        detach_stale_values: bool = True,
        allow_derived: bool = False,
    ) -> None:
        self._set_field(
            label,
            coords,
            allow_derived,
            lambda field, c: field.update_from_component_tensor(
                component_tensor, coords=c, detach_stale_values=detach_stale_values
            ),
        )

    def set_field_from_component_dict(
        self,
        label: str,
        component_dict: dict[str, torch.Tensor],
        coords: Coords | None = None,
        detach_stale_values: bool = True,
        allow_derived: bool = False,
    ) -> None:
        self._set_field(
            label,
            coords,
            allow_derived,
            lambda field, c: field.update_from_component_dict(
                component_dict, coords=c, detach_stale_values=detach_stale_values
            ),
        )

    def set_const(
        self, label: str, value: torch.Tensor, coords: Coords | None = None
    ) -> None:
        # Check that input is a scalar
        if value.ndim != 0:
            raise ValueError(
                f"Constant '{label}' must be a scalar tensor, got shape {value.shape}."
            )

        # Update constant
        if label not in self.consts:
            raise KeyError(f"Const '{label}' not found.")
        self.consts[label] = value

        # Update active coords for the constant
        coords = coords or self.coords
        self._validate_eval_coords(label, self.coords, coords, enforce_ancestry=True)
        self.consts_active_coords[label] = coords

    def evaluate_derived_fields(self, coords: Coords | None = None) -> None:
        coords = coords or self.coords
        self.equation_evaluator.evaluate(coords)

    def validate_eval_coords_primary_fields(self, coords: Coords) -> None:
        for label in self.primary_field_labels:
            active_coords = self.primary_fields_active_coords[label]

            if active_coords is None:
                raise ValueError(f"Primary field '{label}' has not been initialised.")

            if not active_coords.is_ancestor_of(coords):
                raise ValueError(
                    f"Evaluation coords for '{label}' must be the active coords or a "
                    "descendant of them."
                )

    def _empty_field_of_rank(self, label: str, rank: int) -> Field:
        # Create or retrieve Field dataclass for this rank
        field_class = self._field_rank_to_classes.get(rank)
        if field_class is None:
            field_class = make_field_dataclass(
                rank=rank, coords=self.coords, device=self.device
            )
            self._field_rank_to_classes[rank] = field_class

        # Return an empty Field instance with the correct label
        return field_class(label=label)

    def _validate_eval_coords(
        self,
        label: str,
        active_coords: Coords | None,
        coords: Coords,
        enforce_ancestry: bool,
    ) -> None:
        if active_coords is None:
            raise ValueError(f"'{label}' has not been initialised.")

        # Bypasses the tree search if not enforcing coord ancestry
        if not enforce_ancestry:
            return

        if not active_coords.is_ancestor_of(coords):
            raise ValueError(
                f"Evaluation coords for '{label}' must be the active coords "
                "or a descendant of them."
            )

    def _get_active_coords(self, label: str) -> Coords | None:
        # Check for label in fields and consts
        if label in self.field_labels:
            return self.fields_active_coords[label]
        elif label in self.consts:
            return self.consts_active_coords[label]
        else:
            raise KeyError(f"Label '{label}' not found in fields or consts.")
