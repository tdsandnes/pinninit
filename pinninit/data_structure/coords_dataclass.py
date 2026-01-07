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
    tensor: torch.Tensor
    geometric: torch.Tensor
    internal: torch.Tensor
    geometric_labels: tuple[str, ...]
    internal_labels: tuple[str, ...]
    label_to_index: dict[str, int]
    internal_label_to_index: dict[str, int]
    internal_coord_dim: int
    internal_lb: torch.Tensor
    internal_ub: torch.Tensor
    full_tensor: torch.Tensor
    full_geometric: torch.Tensor
    full_internal: torch.Tensor
    operators: Operators
    is_batch: bool
    batch_indices: slice | torch.Tensor
    _batch_tensor: torch.Tensor
    _batch_geometric: torch.Tensor
    _batch_internal: torch.Tensor

    # Methods
    def __post_init__(self) -> None: ...
    def set_batch(self: CoordsProtocol, indices: slice | torch.Tensor) -> None: ...
    def clear_batch(self: CoordsProtocol) -> None: ...
    def get_coord(self: CoordsProtocol, label: str) -> torch.Tensor: ...
    def _set_tensors(self: CoordsProtocol, setting_batch: bool = False) -> None: ...
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
    self._set_tensors()

    # Set internal labels and related attributes
    internal_geometric_coords_dict = self.coord_system.map_coords_to_internal_coords(
        self.full_geometric
    )

    self.internal_lb = torch.min(self.full_internal, dim=0).values
    self.internal_ub = torch.max(self.full_internal, dim=0).values

    self.internal_labels = tuple(sorted(internal_geometric_coords_dict.keys()))
    self.internal_label_to_index = {
        label: i for i, label in enumerate(self.internal_labels)
    }

    for label in internal_geometric_coords_dict:
        setattr(self, label, internal_geometric_coords_dict[label])
    self.internal_coord_dim = len(self.internal_labels)

    # Batch state flag
    self.is_batch = False

    # Set operators
    self.operators = Operators(self)


def set_batch(self: CoordsProtocol, indices: slice | torch.Tensor) -> None:
    """Set batch view into coordinates."""
    self.batch_indices = indices

    # Store individual batch coordinates for derivatives
    for label in self.coord_labels:
        coord = getattr(self, label)
        setattr(self, f"_batch_{label}", coord[indices])

    # Set batch tensors
    self._set_tensors(setting_batch=True)

    self.is_batch = True
    self.operators.set_quantities()


def clear_batch(self: CoordsProtocol) -> None:
    # Delete batch attributes
    for attr in (
        "batch_indices",
        "_batch_tensor",
        "_batch_geometric",
        "_batch_internal",
    ):
        if hasattr(self, attr):
            delattr(self, attr)

    # Delete individual batch coordinates
    for label in self.coord_labels:
        batch_attr = f"_batch_{label}"
        if hasattr(self, batch_attr):
            delattr(self, batch_attr)

    self.is_batch = False
    self.operators.set_quantities()


def get_coord(self: CoordsProtocol, label: str) -> torch.Tensor:
    """Get a coordinate, respecting batch mode."""
    coord = getattr(self, f"_batch_{label}" if self.is_batch else label)
    assert isinstance(coord, torch.Tensor)
    return coord


def n_points(self: CoordsProtocol) -> int:
    """Return number of points, respecting batch mode."""
    return self._batch_tensor.shape[0] if self.is_batch else self.full_tensor.shape[0]


def tensor(self: CoordsProtocol) -> torch.Tensor:
    """Get coord tensor, respecting batch mode."""
    return self._batch_tensor if self.is_batch else self.full_tensor


def geometric(self: CoordsProtocol) -> torch.Tensor:
    """Get gemoetric coord tensor, respecting batch mode."""
    return self._batch_geometric if self.is_batch else self.full_geometric


def internal(self: CoordsProtocol) -> torch.Tensor:
    """Get internal coord tensor, respecting batch mode."""
    return self._batch_internal if self.is_batch else self.full_internal


def _set_tensors(self: CoordsProtocol, setting_batch: bool = False) -> None:
    """Set coordinate tensors with optional prefix for batch mode."""
    coord_prefix = "_batch_" if setting_batch else ""
    tensor_prefix = "_batch_" if setting_batch else "full_"

    # Stack full coordinate tensor
    tensor = torch.stack(
        [getattr(self, f"{coord_prefix}{label}") for label in self.coord_labels], dim=1
    )

    # Stack geometric coordinate tensor
    geometric = torch.stack(
        [getattr(self, f"{coord_prefix}{label}") for label in self.geometric_labels],
        dim=1,
    )

    # Get internal geometric coords
    internal_geometric_coords_dict = self.coord_system.map_coords_to_internal_coords(
        geometric
    )

    # Replace geometric coords with internal geometric coords
    internal_coords_dict = {
        label: getattr(self, f"{coord_prefix}{label}")
        for label in self.coord_labels
        if label not in self.geometric_labels
    }
    internal_coords_dict.update(internal_geometric_coords_dict)

    # Sort alphabetically and stack into tensor
    internal_labels = sorted(internal_coords_dict)
    internal = torch.stack(
        [internal_coords_dict[label] for label in internal_labels], dim=1
    )

    # Set tensors
    setattr(self, f"{tensor_prefix}tensor", tensor)
    setattr(self, f"{tensor_prefix}geometric", geometric)
    setattr(self, f"{tensor_prefix}internal", internal)


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
            ("geometric_labels", tuple[str, ...], field(init=False)),
            ("internal_labels", tuple[str, ...], field(init=False)),
            ("label_to_index", dict[str, int], field(init=False)),
            ("internal_label_to_index", dict[str, int], field(init=False)),
            ("internal_coord_dim", int, field(init=False)),
            ("internal_lb", torch.Tensor, field(init=False)),
            ("internal_ub", torch.Tensor, field(init=False)),
            ("full_tensor", torch.Tensor, field(init=False)),
            ("full_geometric", torch.Tensor, field(init=False)),
            ("full_internal", torch.Tensor, field(init=False)),
            ("operators", Operators, field(init=False)),
            ("is_batch", bool, field(init=False)),
            ("batch_indices", slice | torch.Tensor, field(init=False)),
            ("_batch_tensor", torch.Tensor, field(init=False)),
            ("_batch_geometric", torch.Tensor, field(init=False)),
            ("_batch_internal", torch.Tensor, field(init=False)),
        ]
    )

    # Make dataclass instance
    Coords = make_dataclass(
        "Coords",
        components,
        namespace={
            "__post_init__": _coords_post_init,
            "set_batch": set_batch,
            "clear_batch": clear_batch,
            "get_coord": get_coord,
            "n_points": property(n_points),
            "tensor": property(tensor),
            "geometric": property(geometric),
            "internal": property(internal),
            "_set_tensors": _set_tensors,
            "_validate_input": _coords_validate_input,
        },
    )

    return Coords
