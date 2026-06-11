from __future__ import annotations

from dataclasses import MISSING, field, make_dataclass
from itertools import product
from typing import TYPE_CHECKING, Any, Protocol

import torch

if TYPE_CHECKING:
    from .coords_dataclass import CoordsProtocol


class FieldProtocol(Protocol):
    # Attributes
    # Input fields at class creation time
    rank: int
    coords: CoordsProtocol  # bound as class attribute in make_field_dataclass
    device: torch.device

    # Input fields at instance creation time
    label: str
    # *component tensors* optional at init, always tensors after __post_init__

    # Derived fields
    component_labels: tuple[str, ...]

    # Methods
    def __init__(self, **kwargs: Any) -> None: ...
    def __post_init__(self) -> None: ...
    def _root_indices_for_coords(
        self, coords: CoordsProtocol
    ) -> torch.Tensor | None: ...
    def get_component(
        self, label: str, coords: CoordsProtocol | None = None
    ) -> torch.Tensor: ...
    def component_tensor(self, coords: CoordsProtocol) -> torch.Tensor: ...
    def tensor(self, coords: CoordsProtocol) -> torch.Tensor: ...
    def update_from_component_dict(
        self,
        new_components: dict[str, torch.Tensor],
        coords: CoordsProtocol | None = None,
        detach_stale_values: bool = True,
        detach_new_values: bool = False,
    ) -> None: ...
    def update_from_component_tensor(
        self,
        new_tensor: torch.Tensor,
        coords: CoordsProtocol | None = None,
        detach_stale_values: bool = True,
        detach_new_values: bool = False,
    ) -> None: ...
    def update_from_tensor(
        self,
        new_tensor: torch.Tensor,
        coords: CoordsProtocol | None = None,
        detach_stale_values: bool = True,
        detach_new_values: bool = False,
    ) -> None: ...
    def _validate_inputs(self) -> None: ...
    @property
    def _component_tensor(self) -> torch.Tensor: ...
    @property
    def _tensor(self) -> torch.Tensor: ...


def _field_post_init(self: FieldProtocol) -> None:
    """__post_init__ executed on initialistion of each field object."""
    # Fill missing components with zeros.
    for label in self.component_labels:
        if getattr(self, label) is None:
            setattr(
                self,
                label,
                torch.zeros(self.coords.n_points, device=self.device),
            )

    # Validate
    self._validate_inputs()


# @property (defined as property below)
def _component_tensor(self: FieldProtocol) -> torch.Tensor:
    return torch.stack([getattr(self, label) for label in self.component_labels], dim=1)


# @property (defined as property below)
def _tensor(self: FieldProtocol) -> torch.Tensor:
    component_tensor = self._component_tensor
    if self.rank == 0:
        return component_tensor.squeeze(1)

    tensor_shape = (self.coords.n_points,) + (
        len(self.coords.geometric_labels),
    ) * self.rank
    return component_tensor.reshape(tensor_shape)


def _root_indices_for_coords(
    self: FieldProtocol,
    coords: CoordsProtocol,
) -> torch.Tensor | None:
    """
    Get indices of coords in this field's root coords.

    Returns None when coords is the root coordinate object itself.
    Returns a 1D integer tensor for batch or nested-batch coords.
    """
    if coords is self.coords:
        return None

    if not coords.is_batch:
        raise ValueError(
            "Evaluation/update coordinates must be the field's root coords "
            "or a batch derived from them. Got unrelated non-batch coords."
        )
    if coords.parent is None or coords.parent_idx is None:
        raise ValueError("Batch coords must have a parent and parent_idx set.")

    # Recursion through chain of parents
    parent_indices = self._root_indices_for_coords(coords.parent)

    if parent_indices is None:
        return coords.parent_idx

    return parent_indices[coords.parent_idx]


def get_component(
    self: FieldProtocol, label: str, coords: CoordsProtocol | None = None
) -> torch.Tensor:
    """Get a specific component of the field by label."""
    coords = coords or self.coords
    if label not in self.component_labels:
        raise KeyError(f"Component label '{label}' not found in field components")
    root_indices = self._root_indices_for_coords(coords)
    component: torch.Tensor = getattr(self, label)
    return component if root_indices is None else component[root_indices]


def component_tensor(
    self: FieldProtocol, coords: CoordsProtocol | None = None
) -> torch.Tensor:
    """
    Field values arranged in tensor of shape (n_points, n_components).
    Useful for e.g. component-wise comparisons.

    Note: scalar fields have shape (n_points, 1).
    """
    coords = coords or self.coords
    root_indices = self._root_indices_for_coords(coords)
    return (
        self._component_tensor
        if root_indices is None
        else self._component_tensor[root_indices]
    )


def tensor(self: FieldProtocol, coords: CoordsProtocol | None = None) -> torch.Tensor:
    """
    Field values arranged in tensor of shape (n_points, n_geometric, ..., n_geometric).
    Useful for e.g. linear algebra.

    Note: scalar fields have shape (n_points,).
    """
    coords = coords or self.coords

    # If full original coords
    if coords is self.coords:
        return self._tensor

    batch_component_tensor = self.component_tensor(coords)

    # Protect rank 0 flattening
    if self.rank == 0:
        return batch_component_tensor.squeeze(1)

    tensor_shape = (coords.n_points,) + (len(coords.geometric_labels),) * self.rank

    # Reshape component_tensor into structured tensor
    return batch_component_tensor.reshape(tensor_shape)


def update_from_component_dict(
    self: FieldProtocol,
    new_components: dict[str, torch.Tensor],
    coords: CoordsProtocol | None = None,
    detach_stale_values: bool = True,
    detach_new_values: bool = False,
) -> None:
    """Update field from dict of components in form
    dict[component label: tensor with shape (n_points,)]

    If coords is None, update the full field.
    If coords is a batch, update only those root rows.
    """
    if coords is None:
        coords = self.coords

    # Check that new labels contain all field components
    if set(new_components.keys()) != set(self.component_labels):
        raise KeyError(
            "New components must contain all field component labels when "
            "updating field"
        )

    # Check that all new components have correct shape
    for label, tensor in new_components.items():
        if tensor.shape != (coords.n_points,):
            raise ValueError(
                f"Component '{label}' has shape {tensor.shape}, "
                f"expected ({coords.n_points},)"
            )

    # Stack in canonical component order, then update from component tensor
    new_tensor = torch.stack(
        [new_components[label] for label in self.component_labels],
        dim=1,
    )
    self.update_from_component_tensor(
        new_tensor,
        coords=coords,
        detach_stale_values=detach_stale_values,
        detach_new_values=detach_new_values,
    )


def update_from_component_tensor(
    self: FieldProtocol,
    new_tensor: torch.Tensor,
    coords: CoordsProtocol | None = None,
    detach_stale_values: bool = True,
    detach_new_values: bool = False,
) -> None:
    """Update the field from a component tensor of shape (n_points, n_components).

    If coords is None, n_points means self.coords.n_points.
    If coords is a batch, n_points means coords.n_points and values are scattered
    into the corresponding rows of the root field.
    """
    if coords is None:
        coords = self.coords

    # Check shape of input
    expected_shape = (coords.n_points, len(self.component_labels))
    if new_tensor.shape != expected_shape:
        raise ValueError(
            f"Component tensor must have shape {expected_shape}, got {new_tensor.shape}"
        )

    if new_tensor.device != self.device:
        raise ValueError(
            f"Component tensor is on {new_tensor.device}, expected {self.device}"
        )

    if detach_new_values:
        new_tensor = new_tensor.detach()

    root_indices = self._root_indices_for_coords(coords)

    for i, label in enumerate(self.component_labels):
        new_component_data = new_tensor[:, i]
        if root_indices is None:
            # Full override
            setattr(self, label, new_component_data)
        else:
            # Batch update
            component_data = (
                getattr(self, label).detach()
                if detach_stale_values
                else getattr(self, label)
            )
            component_data[root_indices] = new_component_data
            setattr(self, label, component_data)


def update_from_tensor(
    self: FieldProtocol,
    new_tensor: torch.Tensor,
    coords: CoordsProtocol | None = None,
    detach_stale_values: bool = True,
    detach_new_values: bool = False,
) -> None:
    """Update the field from a structured tensor of shape
    (n_points, n_geometric, ..., n_geometric)

    If coords is None, update the full field.
    If coords is a batch, update only those root rows.
    """
    if coords is None:
        coords = self.coords

    # Check shape of input
    dim = len(coords.geometric_labels)
    expected_shape: tuple[int, ...]
    if self.rank == 0:
        expected_shape = (coords.n_points,)
    else:
        expected_shape = (coords.n_points,) + (dim,) * self.rank

    if new_tensor.shape != expected_shape:
        raise ValueError(
            f"Tensor must have shape {expected_shape}, got {new_tensor.shape}"
        )

    if new_tensor.device != self.device:
        raise ValueError(f"Tensor is on {new_tensor.device}, expected {self.device}")

    # Flatten the tensor along the component axes to match component_tensor shape
    component_tensor = new_tensor.reshape(coords.n_points, -1)

    self.update_from_component_tensor(
        component_tensor,
        coords=coords,
        detach_stale_values=detach_stale_values,
        detach_new_values=detach_new_values,
    )


def _validate_inputs(self: FieldProtocol) -> None:
    # field label must not clash with any coord label
    if self.label in self.coords.coord_labels:
        raise ValueError(f"label '{self.label}' clashes with a coordinate label")

    # field label must not clash with any of its own component labels
    component_labels = generate_tensor_component_labels(
        self.coords.geometric_labels, self.rank
    )
    if self.label in component_labels:
        raise ValueError(
            f"label '{self.label}' clashes with a generated component label"
        )

    # Check components have right shape
    for label in self.component_labels:
        component = getattr(self, label)

        if not isinstance(component, torch.Tensor):
            raise TypeError(f"Field component {label} must be a torch.Tensor")

        if component.device != self.device:
            raise ValueError(
                f"Field component {label} is on {component.device}, "
                f"expected {self.device}"
            )

        if component.shape != (self.coords.n_points,):
            raise ValueError(
                f"Field component {label} must have shape matching "
                f"coordinate points ({self.coords.n_points},), got "
                f"shape {component.shape}"
            )


def generate_tensor_component_labels(
    geometric_labels: tuple[str, ...], rank: int
) -> tuple[str, ...]:
    """Generate labels for each component of a rank-n tensor field."""
    if rank < 0:
        raise ValueError(f"rank must be >= 0, got {rank}")
    if rank == 0:
        return ("value",)
    return tuple(
        "_".join(geometric_labels[i] for i in idx)
        for idx in product(range(len(geometric_labels)), repeat=rank)
    )


def _build_dataclass_fields(
    component_labels: tuple[str, ...],
    rank: int,
    device: torch.device,
) -> list[tuple[str, Any, Any]]:
    return [
        ("label", str, MISSING),
        *(
            (label, torch.Tensor | None, field(default=None))
            for label in component_labels
        ),
        ("rank", int, field(default=rank)),
        ("device", torch.device, field(default=device)),
        ("component_labels", tuple[str, ...], field(default=component_labels)),
    ]


def make_field_dataclass(
    rank: int, coords: CoordsProtocol, device: torch.device
) -> type[FieldProtocol]:
    # Normalise device so it matches realised tensor devices ("cuda" -> "cuda:0")
    device = torch.empty(0, device=device).device

    # Get field components
    component_labels = generate_tensor_component_labels(coords.geometric_labels, rank)

    # Build dataclass fields list
    dataclass_fields = _build_dataclass_fields(component_labels, rank, device)

    # Dataclass namespace with methods and properties
    namespace = {
        "coords": coords,
        "__post_init__": _field_post_init,
        "_component_tensor": property(_component_tensor),
        "_tensor": property(_tensor),
        "_root_indices_for_coords": _root_indices_for_coords,
        "get_component": get_component,
        "component_tensor": component_tensor,
        "tensor": tensor,
        "update_from_component_dict": update_from_component_dict,
        "update_from_component_tensor": update_from_component_tensor,
        "update_from_tensor": update_from_tensor,
        "_validate_inputs": _validate_inputs,
    }

    # Make dataclass instance
    return make_dataclass(
        "Field", dataclass_fields, namespace=namespace, frozen=False, slots=True
    )
