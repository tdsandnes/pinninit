from __future__ import annotations

from dataclasses import field, make_dataclass
from itertools import product
from typing import TYPE_CHECKING, Protocol

import torch

if TYPE_CHECKING:
    from typing import Any

    from .coords_dataclass import CoordsProtocol


class FieldProtocol(Protocol):
    # Attributes
    # Input fields at class creation time
    rank: int
    coords: CoordsProtocol  # bound as class attribute in make_field_dataclass
    device: torch.device

    # Input fields at instance creation time
    label: str
    # *component tensors*

    # Derived fields
    _tensor: torch.Tensor
    _component_tensor: torch.Tensor
    component_labels: tuple[str, ...]

    # Methods
    def __init__(self, **kwargs: Any) -> None: ...
    def __post_init__(self) -> None: ...
    def _set_tensors(self) -> None: ...
    def _loop_parent_component_tensor(self, coords: CoordsProtocol) -> torch.Tensor: ...
    def get_component(self, label: str) -> torch.Tensor: ...
    def component_tensor(self, coords: CoordsProtocol) -> torch.Tensor: ...
    def tensor(self, coords: CoordsProtocol) -> torch.Tensor: ...
    def update_from_component_dict(
        self, new_components: dict[str, torch.Tensor]
    ) -> None: ...
    def update_from_component_tensor(self, new_tensor: torch.Tensor) -> None: ...
    def update_from_tensor(self, new_tensor: torch.Tensor) -> None: ...
    def _validate_inputs(self) -> None: ...


def _field_post_init(self: FieldProtocol) -> None:
    """__post_init__ executed on initialistion of each field object."""
    # Validate
    self._validate_inputs()

    # Set tensors
    self._set_tensors()


def _set_tensors(self: FieldProtocol) -> None:
    """Construct component tensor and tensor in shape that can be used for maths
    operations."""

    # Set component tensor
    component_tensor = torch.stack(
        [getattr(self, label) for label in self.component_labels],
        dim=1,
    )

    # Get tensor shape
    # Note: rank 0 tensor shape is (n_points,) while component_tensor is (n_points, 1).
    tensor_shape = (self.coords.n_points,) + (
        len(self.coords.geometric_labels),
    ) * self.rank

    if self.rank == 0:
        # Protect rank 0 flattening
        tensor = component_tensor.squeeze(1)
    else:
        # Reshape component_tensor into structured tensor
        tensor = component_tensor.reshape(tensor_shape)

    # Set attributes
    # Note: attributes must be set like this rather than self.xxx because of frozen=True
    object.__setattr__(self, "_component_tensor", component_tensor)
    object.__setattr__(self, "_tensor", tensor)


def _loop_parent_component_tensor(
    self: FieldProtocol, coords: CoordsProtocol
) -> torch.Tensor:
    """Recursively get component_tensor of parent field and slice.
    This works for nested batches, searching parents of parents."""

    # If full original coords
    if coords is self.coords:
        return self._component_tensor

    if not coords.is_batch or coords.parent is None:
        raise ValueError(
            "Evaluation coordinates must be a valid subset of field coordinates"
        )

    # Recursivly get parent component tensor until we reach the full dataset
    parent_component_tensor = self._loop_parent_component_tensor(coords.parent)

    # Slice the parent using the current sub-batch index mapping
    return parent_component_tensor[coords.parent_idx]


def get_component(self: FieldProtocol, label: str) -> torch.Tensor:
    """Get a specific component of the field by label."""
    if label not in self.component_labels:
        raise KeyError(f"Component label '{label}' not found in field components")
    component: torch.Tensor = getattr(self, label)
    return component


def component_tensor(self: FieldProtocol, coords: CoordsProtocol) -> torch.Tensor:
    """
    Field values arranged in tensor of shape (n_points, n_components).
    Useful for e.g. component-wise comparisons.

    Note: scalar fields have shape (n_points, 1).
    """
    # Loop through parent component tensors until we reach the full dataset,
    # then slice down to the current batch
    batch_component_tensor = self._loop_parent_component_tensor(coords)

    return batch_component_tensor


def tensor(self: FieldProtocol, coords: CoordsProtocol) -> torch.Tensor:
    """
    Field values arranged in tensor of shape (n_points, n_geometric, ..., n_geometric).
    Useful for e.g. linear algebra.

    Note: scalar fields have shape (n_points,).
    """

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
    self: FieldProtocol, new_components: dict[str, torch.Tensor]
) -> None:
    """Update field from dict of components in form
    dict[component label: tensor with shape (n_points,)]"""
    # Check that new labels contain all field components
    if not set(new_components.keys()) == set(self.component_labels):
        raise KeyError(
            "New components must contain all field component labels when "
            "updating field"
        )

    # Update components from dict
    for label, new_component in new_components.items():
        if new_component.shape != (self.coords.n_points,):
            raise ValueError(
                f"Component '{label}' must have shape ({self.coords.n_points},), "
                f"got {new_component.shape}"
            )
        object.__setattr__(self, label, new_component)

    # Set tensors
    self._set_tensors()


def update_from_component_tensor(self: FieldProtocol, new_tensor: torch.Tensor) -> None:
    """Update the field from a component tensor of shape (n_points, n_components)"""
    # Check shape of input
    expected_shape = (self.coords.n_points, len(self.component_labels))
    if new_tensor.shape != expected_shape:
        raise ValueError(
            f"Component tensor must have shape {expected_shape}, got {new_tensor.shape}"
        )

    # Update components from tensor
    for i, label in enumerate(self.component_labels):
        object.__setattr__(self, label, new_tensor[:, i])

    # Set tensors
    self._set_tensors()


def update_from_tensor(self: FieldProtocol, new_tensor: torch.Tensor) -> None:
    """Update the field from a structured tensor of shape
    (n_points, n_geometric, ..., n_geometric)"""
    # Check shape of input
    dim = len(self.coords.geometric_labels)
    expected_shape: tuple[int, ...]
    if self.rank == 0:
        expected_shape = (self.coords.n_points,)
    else:
        expected_shape = (self.coords.n_points,) + (dim,) * self.rank

    if new_tensor.shape != expected_shape:
        raise ValueError(
            f"Tensor must have shape {expected_shape}, got {new_tensor.shape}"
        )

    # Flatten the tensor along the component axes to match component_tensor shape
    component_tensor = new_tensor.reshape(self.coords.n_points, -1)

    self.update_from_component_tensor(component_tensor)


def _validate_inputs(self: FieldProtocol) -> None:
    # label must not clash with any coord label
    if self.label in self.coords.coord_labels:
        raise ValueError(f"label '{self.label}' clashes with a coordinate label")

    # label must not clash with any component label that would be generated
    component_labels = generate_tensor_component_labels(
        self.coords.geometric_labels, self.rank
    )
    if self.label in component_labels:
        raise ValueError(
            f"label '{self.label}' clashes with a generated component label"
        )

    # Check componets have right shape
    for label in self.component_labels:
        component = getattr(self, label)
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

    # Check rank
    if rank < 0:
        raise ValueError(f"rank must be >= 0, got {rank}")
    if rank == 0:
        return ("value",)

    # Construct lables
    indices = tuple(product(range(len(geometric_labels)), repeat=rank))
    index_to_name = {idx: "_".join(geometric_labels[i] for i in idx) for idx in indices}
    labels = tuple(index_to_name[idx] for idx in indices)
    return labels


def _validate_factory_inputs(
    rank: int,
    coords: CoordsProtocol,
) -> None:
    # Check rank
    if rank < 0:
        raise ValueError(f"rank must be >= 0, got {rank}")


def make_field_dataclass(
    rank: int, coords: CoordsProtocol, device: torch.device
) -> type[FieldProtocol]:
    # Validate
    _validate_factory_inputs(rank, coords)

    dataclass_fields: list[tuple[str, Any, Any | None]] = []

    # Each field component
    component_labels = generate_tensor_component_labels(coords.geometric_labels, rank)

    # Input fields at instance creation time
    for label in component_labels:
        dataclass_fields.append((label, torch.Tensor, None))
    dataclass_fields.extend(
        [
            ("label", str, None),
        ]
    )

    # Input fields at class creation time
    dataclass_fields.extend(
        [
            ("rank", int, field(default=rank)),
            ("device", torch.device, field(default=device)),
            ("component_labels", tuple[str, ...], field(default=component_labels)),
        ]
    )

    # Derived fields
    dataclass_fields.extend(
        [
            ("_tensor", torch.Tensor, field(init=False)),
            ("_component_tensor", torch.Tensor, field(init=False)),
        ]
    )

    # Make dataclass instance
    Field = make_dataclass(
        "Field",
        dataclass_fields,
        namespace={
            "coords": coords,
            "__post_init__": _field_post_init,
            "_set_tensors": _set_tensors,
            "_loop_parent_component_tensor": _loop_parent_component_tensor,
            "get_component": get_component,
            "component_tensor": component_tensor,
            "tensor": tensor,
            "update_from_component_dict": update_from_component_dict,
            "update_from_component_tensor": update_from_component_tensor,
            "update_from_tensor": update_from_tensor,
            "_validate_inputs": _validate_inputs,
        },
        frozen=True,
        slots=True,
    )

    return Field
