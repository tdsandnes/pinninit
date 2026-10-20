from __future__ import annotations

from collections import Counter

import torch

from ..differential_equations.operators import Operators
from .coords import Coords
from .equation import Equation, needed_equations, sort_equations
from .field import Field, make_field_dataclass


class DataContainer:
    """
    The fields and constants defined on one set of coords.

    Primary fields are set directly (e.g. from data or the network output).
    Derived fields are set by evaluating equations.

    Each field is stored on the coords it was last set on: the full coords or a
    batch of them. It can only be read on those coords or a batch derived from
    them, so values from another batch are never read by mistake, and setting a
    field on a batch only costs memory and time for the batch.
    """

    def __init__(
        self,
        coords: Coords,
        primary_field_label_to_rank: dict[str, int],
        derived_field_label_to_rank: dict[str, int] | None = None,
        const_labels: set[str] | None = None,
        equations: list[Equation] | None = None,
    ) -> None:
        derived_field_label_to_rank = derived_field_label_to_rank or {}
        const_labels = const_labels or set()
        _check_labels_unique(
            [
                *coords.coord_labels,
                *primary_field_label_to_rank,
                *derived_field_label_to_rank,
                *const_labels,
            ]
        )

        self.coords = coords
        self.coord_labels = coords.coord_labels
        self.device = coords.device
        self.primary_field_labels = set(primary_field_label_to_rank)
        self.derived_field_labels = set(derived_field_label_to_rank)

        # The Field class of each field, and the field once it has been set
        field_label_to_rank = primary_field_label_to_rank | derived_field_label_to_rank
        self.field_classes: dict[str, type[Field]] = {
            label: make_field_dataclass(rank, type(coords))
            for label, rank in field_label_to_rank.items()
        }
        self.fields: dict[str, Field] = {}

        # Constant values, None until set
        self.consts: dict[str, torch.Tensor | None] = dict.fromkeys(const_labels)

        # Equations that compute the derived fields, sorted so each equation comes
        # after the equations it depends on
        self.equations = sort_equations(equations or [])
        for equation in self.equations:
            self._check_equation(equation)

    @property
    def field_labels(self) -> set[str]:
        return set(self.field_classes)

    def get_field(self, label: str) -> Field:
        if label not in self.field_classes:
            raise KeyError(f"Field '{label}' not found.")
        if label not in self.fields:
            raise ValueError(f"Field '{label}' has not been initialised.")
        return self.fields[label]

    def get_field_tensor(
        self, label: str, coords: Coords | None = None
    ) -> torch.Tensor:
        field, coords = self._field_to_read(label, coords)
        return field.tensor(coords)

    def get_field_component(
        self, field_label: str, component_label: str, coords: Coords | None = None
    ) -> torch.Tensor:
        field, coords = self._field_to_read(field_label, coords)
        return field.get_component(component_label, coords)

    def get_field_component_tensor(
        self, label: str, coords: Coords | None = None
    ) -> torch.Tensor:
        field, coords = self._field_to_read(label, coords)
        return field.component_tensor(coords)

    def get_const(self, label: str) -> torch.Tensor:
        if label not in self.consts:
            raise KeyError(f"Const '{label}' not found.")
        value = self.consts[label]
        if value is None:
            raise ValueError(f"Constant '{label}' has not been initialised.")
        return value

    def set_field_from_tensor(
        self,
        label: str,
        tensor: torch.Tensor,
        coords: Coords | None = None,
        allow_derived: bool = False,
    ) -> None:
        field = self._new_field(label, coords, allow_derived)
        field.update_from_tensor(tensor)
        self.fields[label] = field

    def set_field_from_component_tensor(
        self,
        label: str,
        component_tensor: torch.Tensor,
        coords: Coords | None = None,
        allow_derived: bool = False,
    ) -> None:
        field = self._new_field(label, coords, allow_derived)
        field.update_from_component_tensor(component_tensor)
        self.fields[label] = field

    def set_field_from_component_dict(
        self,
        label: str,
        component_dict: dict[str, torch.Tensor],
        coords: Coords | None = None,
        allow_derived: bool = False,
    ) -> None:
        field = self._new_field(label, coords, allow_derived)
        field.update_from_component_dict(component_dict)
        self.fields[label] = field

    def set_const(self, label: str, value: torch.Tensor) -> None:
        if label not in self.consts:
            raise KeyError(f"Const '{label}' not found.")
        if value.ndim != 0:
            raise ValueError(
                f"Constant '{label}' must be a scalar tensor, got shape {value.shape}."
            )
        self.consts[label] = value

    def clear_fields(self) -> None:
        """Forget the values of all fields, e.g. to free the autograd graphs
        they hold."""
        self.fields.clear()

    def evaluate_derived_fields(
        self, coords: Coords | None = None, labels: list[str] | None = None
    ) -> None:
        """Evaluate the equations at coords and store the results. If labels
        are given, only the equations those fields need are evaluated."""
        coords = self.coords if coords is None else coords
        operators = Operators(coords)
        equations = self.equations
        if labels is not None:
            equations = needed_equations(equations, labels)
        for equation in equations:
            outputs = equation.evaluate(self, coords, operators)
            for label, tensor in outputs.items():
                self.set_field_from_tensor(label, tensor, coords, allow_derived=True)

    def _check_equation(self, equation: Equation) -> None:
        """Check an equation computes derived fields of this container, from
        its coords, fields and consts."""
        not_derived = set(equation.output_labels) - self.derived_field_labels
        if not_derived:
            raise ValueError(
                f"Equation '{equation.label}' outputs {sorted(not_derived)}, "
                "which are not derived fields."
            )

        for label, rank in equation.output_label_to_rank.items():
            field_rank = self.field_classes[label].rank
            if rank != field_rank:
                raise ValueError(
                    f"Equation '{equation.label}' outputs '{label}' with rank "
                    f"{rank}, but the field has rank {field_rank}."
                )

        known_labels = {*self.coord_labels, *self.field_labels, *self.consts}
        unknown = sorted(set(equation.input_labels) - known_labels)
        if unknown:
            raise ValueError(
                f"Equation '{equation.label}' takes {unknown}, which are not "
                "coords, fields or consts."
            )

    def _new_field(
        self, label: str, coords: Coords | None, allow_derived: bool
    ) -> Field:
        """Check a field can be set on coords, and return a new field on them."""
        if label not in self.field_classes:
            raise KeyError(f"Field '{label}' not found.")
        if label in self.derived_field_labels and not allow_derived:
            raise ValueError(
                f"Field '{label}' is a derived field and cannot be set directly."
            )

        coords = self.coords if coords is None else coords
        if not self.coords.is_ancestor_of(coords):
            raise ValueError(
                f"Evaluation coords for '{label}' must be the active coords "
                "or a descendant of them."
            )
        return self.field_classes[label](label=label, coords=coords)

    def _field_to_read(self, label: str, coords: Coords | None) -> tuple[Field, Coords]:
        """Check a field can be read on coords, and return it with the coords."""
        field = self.get_field(label)
        coords = self.coords if coords is None else coords
        if not field.coords.is_ancestor_of(coords):
            raise ValueError(
                f"Evaluation coords for '{label}' must be the active coords "
                "or a descendant of them."
            )
        return field, coords


def _check_labels_unique(labels: list[str]) -> None:
    """Coord, field and const labels share one namespace in equation arguments."""
    duplicates = sorted(label for label, count in Counter(labels).items() if count > 1)
    if duplicates:
        raise ValueError(
            f"Labels {duplicates} are used more than once across coords, fields "
            "and consts."
        )
