from __future__ import annotations

from dataclasses import MISSING, field, make_dataclass
from typing import TYPE_CHECKING, Protocol, cast

import torch

from ..geometry.coordinate_system import CoordSystem
from ..geometry.domain import Domain

if TYPE_CHECKING:
    from typing import Any


class CoordsProtocol(Protocol):
    # Attributes
    # Input fields at class creation time
    coord_labels: tuple[str, ...]
    coord_system: CoordSystem
    device: torch.device

    # Input fields at instance creation time
    # *component tensors*

    # Derived fields
    n_points: int
    tensor: torch.Tensor
    geometric: torch.Tensor
    internal: torch.Tensor
    geometric_labels: tuple[str, ...]
    internal_labels: tuple[str, ...]
    parent: CoordsProtocol | None
    parent_idx: torch.Tensor | None

    @property
    def is_batch(self) -> bool: ...

    # Methods
    def __init__(self, **kwargs: Any) -> None: ...
    def __post_init__(self) -> None: ...
    def _init_from_scratch(self) -> None: ...
    def _init_from_parent(self) -> None: ...
    def get_coord(self, label: str) -> torch.Tensor: ...
    def get_batch(self, idx: torch.Tensor) -> CoordsProtocol: ...
    def validate_in_domain(self, domain: Domain) -> None: ...
    def _set_tensors(self) -> None: ...
    def _validate_inputs(self) -> None: ...


def _coords_post_init(self: CoordsProtocol) -> None:
    """__post_init__ executed on initialistion of each coord object."""
    # Validate
    self._validate_inputs()

    # Init object based on whether it is a batch or full set of points
    if self.is_batch:
        self._init_from_parent()
    else:
        self._init_from_scratch()

    # Set tensors
    self._set_tensors()


def _init_from_scratch(self: CoordsProtocol) -> None:
    """Set up coordinates from scratch."""
    if self.parent is not None or self.parent_idx is not None:
        raise ValueError("parent and parent_idx must be None when not batched")

    # Geometric coord labels
    geometric_labels = self.coord_system.axes_labels_selected
    if not set(geometric_labels).issubset(set(self.coord_labels)):
        raise ValueError(
            "Geometric coordinate labels are not a subset of coordinate labels"
        )

    # Internal coord labels
    non_geometric = [
        label for label in self.coord_labels if label not in geometric_labels
    ]
    internal_labels = tuple(
        sorted([*non_geometric, *self.coord_system.internal_geometric_labels])
    )

    # Set attributes
    # Note: attributes must be set like this rather than self.xxx because of frozen=True
    object.__setattr__(self, "geometric_labels", geometric_labels)
    object.__setattr__(self, "internal_labels", internal_labels)


def _init_from_parent(self: CoordsProtocol) -> None:
    """Fast initialisation inheriting static metadata from parent."""
    if self.parent is None or self.parent_idx is None:
        raise ValueError("parent and parent_idx must be given when batched")

    parent = self.parent
    object.__setattr__(self, "geometric_labels", parent.geometric_labels)
    object.__setattr__(self, "internal_labels", parent.internal_labels)


def get_coord(self: CoordsProtocol, label: str) -> torch.Tensor:
    """Get a coordinate"""
    if label not in self.coord_labels:
        raise KeyError(f"Coordinate label '{label}' not found in coords")
    coord: torch.Tensor = getattr(self, label)
    return coord


def get_batch(self: CoordsProtocol, idx: torch.Tensor) -> CoordsProtocol:
    """Get a batched set of coordinates. Returns a new coord object."""
    if idx.dtype not in (torch.long, torch.int64):
        raise TypeError("idx must be integer tensor")
    if idx.ndim != 1:
        raise ValueError("idx must be 1D")

    # Get dict of batched coords
    batch_coords_dict = {
        label: getattr(self, label)[idx].detach().requires_grad_(True)
        for label in self.coord_labels
    }

    # Make new batched coord object using parent's Coord class
    Coords: Any = self.__class__
    batch_instance = Coords(
        **batch_coords_dict,
        coord_system=self.coord_system,
        device=self.device,
        parent=self,
        parent_idx=idx,
    )

    return cast(CoordsProtocol, batch_instance)


def validate_in_domain(self: CoordsProtocol, domain: Domain) -> None:
    """Verify that all points are within a domain."""
    if domain.coord_system != self.coord_system:
        raise ValueError(
            f"Domain coordinate system {domain.coord_system} does not "
            f"match {self.coord_system}"
        )

    mask = domain.contains_geometric(self.geometric)
    if not torch.all(mask):
        raise ValueError("Geometric coordinates lie outside of domain.")


def _set_tensors(self: CoordsProtocol) -> None:
    """
    Store coords tensor, geometric coords tensor, and internal coords tensor.

    The order here is important since we want the computational graph to go like:
    `individual coords -> geometric -> internal`
    so that we can differentiate internal wrt both geometric and individual coords.
    """
    # First stack coord tensor
    coords_dict = {label: getattr(self, label) for label in self.coord_labels}
    tensor = torch.stack([coords_dict[label] for label in self.coord_labels], dim=1)

    # Stack geometric coord tensor
    geometric = torch.stack(
        [coords_dict[label] for label in self.geometric_labels], dim=1
    )
    # Slice back to dict so graph goes through geometric
    geometric_dict = {
        label: geometric[:, i] for i, label in enumerate(self.geometric_labels)
    }

    # Make internal dict, first with non-geometric coords
    internal_coords_dict = {
        label: coords_dict[label]
        for label in self.coord_labels
        if label not in self.geometric_labels
    }
    # Then add internal geometric coords
    internal_coords_dict.update(
        self.coord_system.map_geometric_to_internal_geometric(geometric_dict)
    )

    # Stack internal coord tensor
    internal = torch.stack(
        [internal_coords_dict[label] for label in self.internal_labels], dim=1
    )

    # Set tensors
    object.__setattr__(self, "tensor", tensor)
    object.__setattr__(self, "geometric", geometric)
    object.__setattr__(self, "internal", internal)
    object.__setattr__(self, "n_points", tensor.shape[0])


def _validate_inputs(self: CoordsProtocol) -> None:
    # Validate coord components
    lengths = []
    for label in self.coord_labels:
        coord = getattr(self, label)

        if coord.dim() != 1:
            raise ValueError(f"Coordinate {label} must be 1D, got shape {coord.shape}")

        # Move coordinate to correct device
        if coord.device != self.device:
            raise ValueError(
                f"Coordinate {label} on {coord.device}, expected {self.device}"
            )

        lengths.append(coord.size(0))

    if len(set(lengths)) > 1:
        raise ValueError(f"All coordinates must have the same length, got {lengths}")


def _validate_factory_inputs(
    coord_labels: set[str],
    coord_system: CoordSystem,
) -> None:
    # Validate coord_labels
    if not coord_labels:
        raise ValueError("coord_labels must not be empty")

    # Validate coordinate system
    if not set(coord_system.axes_labels_selected).issubset(coord_labels):
        raise ValueError(
            f"coord_system requires geometric axes {coord_system.axes_labels_selected} "
            f"but coord_labels only contains {coord_labels}"
        )


def make_coords_dataclass(
    coord_labels: set[str],
    coord_system: CoordSystem,
    device: torch.device,
) -> type[CoordsProtocol]:
    # Validate
    _validate_factory_inputs(coord_labels, coord_system)

    dataclass_fields: list[tuple[str, Any, Any | None]] = []

    # Each coordinate
    coord_labels_sorted = tuple(sorted(coord_labels))

    # Input fields at instance creation time
    for label in coord_labels_sorted:
        dataclass_fields.append((label, torch.Tensor, MISSING))

    # Input fields at class creation time
    dataclass_fields.extend(
        [
            (
                "coord_labels",
                tuple[str, ...],
                field(default=coord_labels_sorted),
            ),
            ("coord_system", CoordSystem, field(default=coord_system)),
            ("device", torch.device, field(default=device)),
            ("parent", CoordsProtocol | None, field(default=None)),
            ("parent_idx", torch.Tensor | None, field(default=None)),
        ]
    )

    # Derived fields
    dataclass_fields.extend(
        [
            ("n_points", int, field(init=False)),
            ("geometric_labels", tuple[str, ...], field(init=False)),
            ("internal_labels", tuple[str, ...], field(init=False)),
            ("tensor", torch.Tensor, field(init=False)),
            ("geometric", torch.Tensor, field(init=False)),
            ("internal", torch.Tensor, field(init=False)),
        ]
    )

    # Make dataclass instance
    Coords = make_dataclass(
        "Coords",
        dataclass_fields,
        namespace={
            "__post_init__": _coords_post_init,
            "_init_from_scratch": _init_from_scratch,
            "_init_from_parent": _init_from_parent,
            "get_coord": get_coord,
            "get_batch": get_batch,
            "validate_in_domain": validate_in_domain,
            "_set_tensors": _set_tensors,
            "_validate_inputs": _validate_inputs,
            "is_batch": property(lambda self: self.parent is not None),
        },
        frozen=True,
        slots=True,
    )

    return Coords
