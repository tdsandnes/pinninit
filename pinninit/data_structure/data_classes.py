from __future__ import annotations

from dataclasses import field, make_dataclass
from typing import TYPE_CHECKING, Protocol

import torch

from ..geometry.coordinate_system import CoordSystem
from .data_structure_utils import generate_tensor_component_labels

if TYPE_CHECKING:
    from typing import Any


class CoordsProtocol(Protocol):
    # Attributes
    geometric_labels: tuple[str, ...]
    coord_system: CoordSystem
    sorted_coord_labels: tuple[str, ...]
    label_to_index: dict[str, int]
    n_points: int

    # Methods
    def __post_init__(self) -> None: ...
    def _validate_input(self) -> None: ...
    def tensor(self) -> torch.Tensor: ...
    def geometric(self) -> torch.Tensor: ...


class FieldProtocol(Protocol):
    # Attributes
    rank: int
    component_labels: tuple[str, ...]
    coords: CoordsProtocol

    # Methods
    def __post_init__(self) -> None: ...
    def _validate_input(self) -> None: ...
    def _stack_tensor_field(
        self, labels: tuple[str, ...], rank: int
    ) -> torch.Tensor: ...
    def tensor(self) -> torch.Tensor: ...


def make_coords_dataclass(coord_labels: list[str]) -> type[CoordsProtocol]:
    # Define components
    components: list[tuple[str, Any]] = []
    # Add coordinate labels
    components += [(label, torch.Tensor) for label in coord_labels]
    # Add coordinate system
    components.append(("coord_system", CoordSystem))

    def __post_init__(self: CoordsProtocol) -> None:
        # Geometric coordinates (sorted based on coord system metadata)
        self.geometric_labels = self.coord_system.axes_labels_selected

        # Validate input
        self._validate_input()

        # Store dicts for consistent ordering (sorted alphabetically)
        self.sorted_coord_labels = tuple(sorted(coord_labels))
        self.label_to_index = {
            label: i for i, label in enumerate(self.sorted_coord_labels)
        }

    # Add methods
    def _validate_input(self: Any) -> None:
        # Validate coord components
        lengths = []
        for label in coord_labels:
            tensor = getattr(self, label)
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"Coordinate {label} is not a torch.Tensor")
            if tensor.dim() != 1:
                raise ValueError(
                    f"Coordinate {label} must be 1D, got shape " f"{tensor.shape}"
                )
            lengths.append(tensor.size(0))

        if not lengths:
            raise ValueError("No coordinates provided")

        if len(set(lengths)) > 1:
            raise ValueError(
                f"All coordinates must have the same length, " f"got {lengths}"
            )

        # Store length
        self.n_points = lengths[0]

        # Validate that geometric coords are subset of all coords
        if not set(self.geometric_labels).issubset(set(coord_labels)):
            raise ValueError(
                "Geometric coordinate labels are not a subset of " "coordinate labels"
            )

    def tensor(self: CoordsProtocol) -> torch.Tensor:
        # Returns tensor of coords, ordered by sorted_coord_labels
        return torch.stack(
            [getattr(self, label) for label in self.sorted_coord_labels], dim=1
        )

    def geometric(self: CoordsProtocol) -> torch.Tensor:
        # Returns tensor of geometric coords, ordered by coord system metadata
        return torch.stack(
            [getattr(self, label) for label in self.geometric_labels], dim=1
        )

    # Make dataclass instance
    Coords = make_dataclass(
        "Coords",
        components,
        namespace={
            "_validate_input": _validate_input,
            "__post_init__": __post_init__,
            "tensor": tensor,
            "geometric": geometric,
        },
    )

    return Coords


def make_field_dataclass(rank: int, coords: CoordsProtocol) -> type[FieldProtocol]:
    # Define components
    components: list[tuple[str, Any, Any | None]] = []
    # Add tensor labels
    components += [
        (label, torch.Tensor, None)
        for label in generate_tensor_component_labels(coords.geometric_labels, rank)
    ]
    # Add rank
    components.append(("rank", int, field(default=rank)))
    # Add coordinate system
    components.append(("coords", type(coords), field(default=coords)))

    def __post_init__(self: FieldProtocol) -> None:
        # Store the ordered component labels
        self.component_labels = generate_tensor_component_labels(
            self.coords.geometric_labels, self.rank
        )

        # Validate input
        self._validate_input()

    # Add methods
    def _validate_input(self: Any) -> None:
        # Don't allow negative rank
        if self.rank < 0:
            raise ValueError(
                f"Tensor rank must be greater than or equal to 0," f"got {self.rank}"
            )

        # Validate components
        if set(self.component_labels) != {
            label
            for label, _ in self.__dataclass_fields__.items()
            if label not in ["rank", "coords"]
        }:
            raise ValueError(
                f"Field labels do not match those expected from "
                f"coordinates geometric labels, {self.component_labels}"
            )

        for label in self.component_labels:
            tensor = getattr(self, label)
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"Field component {label} is not a torch.Tensor")
            if tensor.shape != (self.coords.n_points,):
                raise ValueError(
                    f"Field component {label} must have shape matching "
                    f"coordinate points ({self.coords.n_points},), got "
                    f"shape {tensor.shape}"
                )

    def _stack_tensor_field(
        self: FieldProtocol, labels: tuple[str, ...], rank: int
    ) -> torch.Tensor:
        # Recursively stack tensor
        if rank == 1:
            # Break recursion after rank 1 is reached
            return torch.stack([getattr(self, label) for label in labels], dim=1)
        else:
            # Recursively group by first index
            grouped = []
            for coord in self.coords.geometric_labels:
                # rest of label string for labels that start with coord
                sub_labels = [label[1:] for label in labels if label.startswith(coord)]
                # recursively stack along remaining coords
                grouped.append(self._stack_tensor_field(tuple(sub_labels), rank - 1))
            return torch.stack(grouped, dim=1)

    def tensor(self: FieldProtocol) -> torch.Tensor:
        # Scalar field
        if self.rank == 0:
            value: torch.Tensor = getattr(self, self.component_labels[0])
            return value
        else:
            # Returns tensor of vector field, ordered by geometric coords
            return self._stack_tensor_field(self.component_labels, self.rank)

    # Make dataclass instance
    Field = make_dataclass(
        "Field",
        components,
        namespace={
            "_validate_input": _validate_input,
            "_stack_tensor_field": _stack_tensor_field,
            "__post_init__": __post_init__,
            "tensor": tensor,
        },
    )

    return Field
