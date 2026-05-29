from __future__ import annotations

from dataclasses import field, make_dataclass
from typing import Any, Protocol, cast

import torch

from ..geometry.coordinate_system import CoordSystem
from ..geometry.domain import Domain


class CoordsProtocol(Protocol):
    # Attributes
    # Input fields
    coord_labels: tuple[str, ...]
    coord_system: CoordSystem
    domain: Domain | None
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
    parent: CoordsProtocol | None
    parent_idx: torch.Tensor | None

    # Methods
    def __post_init__(self) -> None: ...
    def _init_from_parent(self) -> None: ...
    def _init_from_scratch(self) -> None: ...
    def get_coord(self, label: str) -> torch.Tensor: ...
    def get_batch(self, idx: torch.Tensor) -> CoordsProtocol: ...
    def _set_tensors(self) -> None: ...
    def _validate_domain(self) -> None: ...
    def _validate_input(self) -> None: ...

    @property
    def is_batch(self) -> bool: ...


def _coords_post_init(self: CoordsProtocol) -> None:
    # Set up based it is a batch or full dataset
    if self.is_batch:
        self._init_from_parent()
    else:
        self._init_from_scratch()

    # Set tensors
    self._set_tensors()
    self.n_points = self.tensor.shape[0]

    # From scratch setup that requires tensors
    if not self.is_batch:
        self._validate_domain()
        # Note that these bounds are of the full dataset and are inherited by batches
        self.internal_lb = torch.min(self.internal, dim=0).values.detach()
        self.internal_ub = torch.max(self.internal, dim=0).values.detach()


def _init_from_scratch(self: CoordsProtocol) -> None:
    """Set up coordinates from scratch."""
    self.coord_labels = tuple(sorted(self.coord_labels))
    self.label_to_index = {label: i for i, label in enumerate(self.coord_labels)}

    # Validate
    self._validate_input()

    # Geometric coord labels
    self.geometric_labels = self.coord_system.axes_labels_selected
    if not set(self.geometric_labels).issubset(set(self.coord_labels)):
        raise ValueError(
            "Geometric coordinate labels are not a subset of coordinate labels"
        )

    # Internal coord labels
    coords_dict = {label: getattr(self, label) for label in self.coord_labels}
    geometric = torch.stack([coords_dict[lbl] for lbl in self.geometric_labels], dim=1)
    internal_geometric_dict = self.coord_system.map_geometric_to_internal(geometric)
    non_geometric = [
        lbl for lbl in self.coord_labels if lbl not in self.geometric_labels
    ]

    self.internal_labels = tuple(
        sorted([*non_geometric, *internal_geometric_dict.keys()])
    )
    self.internal_label_to_index = {
        lbl: i for i, lbl in enumerate(self.internal_labels)
    }
    self.internal_coord_dim = len(self.internal_labels)

    if self.parent is not None:
        raise ValueError("parent should be None when coords not batched")
    if self.parent_idx is not None:
        raise ValueError("parent_idx should be None when coords not batched")


def _init_from_parent(self: CoordsProtocol) -> None:
    """Fast initialisation inheriting static metadata from parent."""
    assert self.parent is not None
    assert self.parent_idx is not None

    parent = self.parent
    self.coord_labels = parent.coord_labels
    self.label_to_index = parent.label_to_index
    self.geometric_labels = parent.geometric_labels
    self.internal_labels = parent.internal_labels
    self.internal_label_to_index = parent.internal_label_to_index
    self.internal_coord_dim = parent.internal_coord_dim
    self.domain = parent.domain

    # Inherit system bounds
    self.internal_lb = parent.internal_lb.detach()
    self.internal_ub = parent.internal_ub.detach()

    # Fast validation specific to the slice array size
    first_coord = getattr(self, self.coord_labels[0])
    if self.parent_idx.shape != (first_coord.shape[0],):
        raise ValueError("parent_idx must match number of points")


def get_coord(self: CoordsProtocol, label: str) -> torch.Tensor:
    """Get a coordinate"""
    coord: torch.Tensor = getattr(self, label)
    return coord


def _compute_tensors(
    coord_labels: tuple[str, ...],
    geometric_labels: tuple[str, ...],
    internal_labels: tuple[str, ...],
    coord_system: CoordSystem,
    coords_dict: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute tensor, geometric, and internal from a dict of coord tensors"""
    tensor = torch.stack([coords_dict[label] for label in coord_labels], dim=1)

    geometric = torch.stack([coords_dict[label] for label in geometric_labels], dim=1)

    internal_geometric_coords_dict = coord_system.map_geometric_to_internal(geometric)

    internal_coords_dict = {
        label: coords_dict[label]
        for label in coord_labels
        if label not in geometric_labels
    }
    internal_coords_dict.update(internal_geometric_coords_dict)

    internal = torch.stack(
        [internal_coords_dict[label] for label in internal_labels], dim=1
    )

    return tensor, geometric, internal


def _set_tensors(self: CoordsProtocol) -> None:
    coords_dict = {label: getattr(self, label) for label in self.coord_labels}

    self.tensor, self.geometric, self.internal = _compute_tensors(
        self.coord_labels,
        self.geometric_labels,
        self.internal_labels,
        self.coord_system,
        coords_dict,
    )


def get_batch(self: CoordsProtocol, idx: torch.Tensor) -> CoordsProtocol:
    if idx.dtype not in (torch.long, torch.int64):
        raise TypeError("idx must be integer tensor")
    if idx.ndim != 1:
        raise ValueError("idx must be 1D")

    batch_coords_dict = {
        label: getattr(self, label)[idx].detach().requires_grad_(True)
        for label in self.coord_labels
    }

    Coords: Any = self.__class__
    batch_instance = Coords(
        **batch_coords_dict,
        coord_system=self.coord_system,
        domain=self.domain,
        device=self.device,
        parent=self,
        parent_idx=idx,
    )

    return cast(CoordsProtocol, batch_instance)


def _validate_input(self: Any) -> None:
    # Validate coordinate system
    if not isinstance(self.coord_system, CoordSystem):
        raise TypeError(
            "coord_system must be a CoordSystem instance, "
            f"got {type(self.coord_system)}"
        )

    # Validate coord components
    lengths = []
    for label in self.coord_labels:
        if not hasattr(self, label):
            raise ValueError(f"Coordinate label '{label}' not found")

        coord = getattr(self, label)

        if not isinstance(coord, torch.Tensor):
            raise TypeError(f"Coordinate {label} is not a torch.Tensor")
        if coord.dim() != 1:
            raise ValueError(f"Coordinate {label} must be 1D, got shape {coord.shape}")

        # Move coordinate to correct device
        if coord.device != self.device:
            raise ValueError(
                f"Coordinate {label} on {coord.device}, expected {self.device}"
            )

        lengths.append(coord.size(0))

    if not lengths:
        raise ValueError("No coordinates provided")

    if len(set(lengths)) > 1:
        raise ValueError(f"All coordinates must have the same length, got {lengths}")


def _validate_domain(self: CoordsProtocol) -> None:
    """Isolate domain verification boundaries."""
    if self.domain is not None:
        mask = self.domain.contains_geometric(self.geometric)
        if not torch.all(mask):
            raise ValueError("Geometric coordinates lie outside of domain.")


def make_coords_dataclass(
    coord_labels: set[str] | list[str] | tuple[str, ...], device: torch.device
) -> type[CoordsProtocol]:
    components: list[tuple[str, Any, Any | None]] = []

    # Check for duplicate coordinate labels
    labels_list = list(coord_labels)
    if len(labels_list) != len(set(labels_list)):
        seen: set[str] = set()
        duplicates: set[str] = set()
        for item in labels_list:
            if item in seen:
                duplicates.add(item)
            else:
                seen.add(item)
        raise ValueError(f"Duplicate coordinate labels provided: {duplicates}")

    # Sort labels and store dict for coord label ordering
    coord_labels_sorted = tuple(sorted(labels_list))

    # Each coordinate
    for label in coord_labels_sorted:
        components.append((label, torch.Tensor, None))

    # Input fields
    components.extend(
        [
            (
                "coord_labels",
                tuple[str, ...],
                field(default=coord_labels_sorted),
            ),
            ("coord_system", CoordSystem, None),
            ("domain", Domain | None, None),
            ("device", torch.device, field(default=device)),
            ("parent", CoordsProtocol | None, field(default=None)),
            ("parent_idx", torch.Tensor | None, field(default=None)),
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
        ]
    )

    # Make dataclass instance
    Coords = make_dataclass(
        "Coords",
        components,
        namespace={
            "__post_init__": _coords_post_init,
            "_init_from_scratch": _init_from_scratch,
            "_init_from_parent": _init_from_parent,
            "get_coord": get_coord,
            "get_batch": get_batch,
            "_set_tensors": _set_tensors,
            "_validate_input": _validate_input,
            "_validate_domain": _validate_domain,
            "is_batch": property(lambda self: self.parent is not None),
        },
    )

    return Coords
