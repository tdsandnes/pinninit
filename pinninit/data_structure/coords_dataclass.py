from __future__ import annotations

from dataclasses import field, make_dataclass
from typing import Any, Protocol

import torch

from ..differential_equations.operators import Operators
from ..geometry.coordinate_system import CoordSystem
from .data_structure_utils import convert_to_tensor


class CoordsProtocol(Protocol):
    # Attributes
    # Input fields
    coord_labels: tuple[str, ...]
    coord_system: CoordSystem
    device: torch.device

    # Derived fields
    n_points: int
    geometric_labels: tuple[str, ...]
    internal_labels: tuple[str, ...]
    label_to_index: dict[str, int]
    internal_label_to_index: dict[str, int]
    internal_coord_dim: int
    internal_lb: torch.Tensor
    internal_ub: torch.Tensor
    tensor: torch.Tensor
    geometric: torch.Tensor
    internal: torch.Tensor
    operators: Operators

    # Methods
    def __post_init__(self) -> None: ...
    def _validate_input(self) -> None: ...


def _coords_post_init(self: CoordsProtocol) -> None:
    # Sort labels and store dict for coord label ordering
    self.coord_labels = tuple(sorted(self.coord_labels))
    self.label_to_index = {label: i for i, label in enumerate(self.coord_labels)}

    # Convert inputs to tensors if not already
    for label in self.coord_labels:
        value = getattr(self, label)
        value_tensor = convert_to_tensor(value, self.device, requires_grad=True)
        setattr(self, label, value_tensor)

    # Geometric coordinates (sorted based on coord system metadata)
    self.geometric_labels = self.coord_system.axes_labels_selected

    # Validate input
    self._validate_input()

    # Set tensors
    self.tensor = torch.stack(
        [getattr(self, label) for label in self.coord_labels], dim=1
    )
    self.geometric = torch.stack(
        [getattr(self, label) for label in self.geometric_labels], dim=1
    )

    # Get internal geometric coords
    internal_geometric_coords_dict = self.coord_system.map_coords_to_internal_coords(
        self.geometric
    )

    # Replace geometric coords with internal geometric coords
    internal_coords_dict = {
        label: getattr(self, label)
        for label in self.coord_labels
        if label not in self.geometric_labels
    }
    internal_coords_dict.update(
        internal_geometric_coords_dict
    )  # Add new geometric coordinates
    # sort alphabetically and stack into tensor
    internal_labels = sorted(internal_coords_dict)
    self.internal = torch.stack(
        [internal_coords_dict[label] for label in internal_labels], dim=1
    )

    self.internal_lb = torch.min(self.internal, dim=0).values
    self.internal_ub = torch.max(self.internal, dim=0).values

    self.internal_labels = tuple(internal_labels)
    self.internal_label_to_index = {label: i for i, label in enumerate(internal_labels)}
    for label in internal_geometric_coords_dict:
        setattr(self, label, internal_geometric_coords_dict[label])
    self.internal_coord_dim = len(self.internal_labels)

    # Set operators
    self.operators = Operators(self)


def _coords_validate_input(self: Any) -> None:
    # Validate coord components
    lengths = []
    for label in self.coord_labels:
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
    if not set(self.geometric_labels).issubset(set(self.coord_labels)):
        raise ValueError(
            "Geometric coordinate labels are not a subset of " "coordinate labels"
        )


def make_coords_dataclass(
    coord_labels: set[str], device: torch.device
) -> type[CoordsProtocol]:
    components: list[tuple[str, Any, Any | None]] = []

    # Each coordinate
    for label in coord_labels:
        components.append((label, Any, None))

    # Input fields
    components.extend(
        [
            (
                "coord_labels",
                tuple[str, ...],
                field(default=tuple(sorted(coord_labels))),
            ),
            ("coord_system", CoordSystem, None),
            ("device", torch.device, field(default=device)),
            # Derived fields
            ("n_points", int, field(init=False)),
            ("geometric_labels", tuple[str, ...], field(init=False)),
            ("internal_labels", tuple[str, ...], field(init=False)),
            ("label_to_index", dict[str, int], field(init=False)),
            ("internal_label_to_index", dict[str, int], field(init=False)),
            ("internal_coord_dim", int, field(init=False)),
            ("internal_lb", torch.Tensor, field(init=False)),
            ("internal_ub", torch.Tensor, field(init=False)),
            ("tensor", torch.Tensor, field(init=False)),
            ("geometric", torch.Tensor, field(init=False)),
            ("internal", torch.Tensor, field(init=False)),
            ("operators", Operators, field(init=False)),
        ]
    )

    # Make dataclass instance
    Coords = make_dataclass(
        "Coords",
        components,
        namespace={
            "__post_init__": _coords_post_init,
            "_validate_input": _coords_validate_input,
        },
    )

    return Coords
