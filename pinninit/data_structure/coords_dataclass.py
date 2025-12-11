from __future__ import annotations

from dataclasses import field, make_dataclass
from typing import TYPE_CHECKING, Protocol

import torch

from ..geometry.coordinate_system import CoordSystem
from .data_structure_utils import convert_to_tensor

if TYPE_CHECKING:
    from typing import Any


class CoordsProtocol(Protocol):
    # Attributes
    geometric_labels: tuple[str, ...]
    coord_system: CoordSystem
    coord_labels: tuple[str, ...]
    label_to_index: dict[str, int]
    n_points: int
    tensor: torch.Tensor
    geometric: torch.Tensor
    device: torch.device

    # Methods
    def __post_init__(self) -> None: ...
    def _validate_input(self) -> None: ...


def _coords_post_init(self: CoordsProtocol) -> None:
    # Sort labels and tore dict for coord label ordering
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
    coord_labels: tuple[str, ...], device: torch.device
) -> type[CoordsProtocol]:
    components: list[tuple[str, Any, Any | None]] = [
        # Input fields
        ("coord_system", CoordSystem, None),
        ("coord_labels", tuple[str, ...], field(default=tuple(sorted(coord_labels)))),
        ("device", torch.device, field(default=device)),
        # Derived fields
        ("geometric_labels", tuple[str, ...], field(init=False)),
        ("label_to_index", dict[str, int], field(init=False)),
        ("n_points", int, field(init=False)),
        ("tensor", torch.Tensor, field(init=False)),
        ("geometric", torch.Tensor, field(init=False)),
    ]

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
