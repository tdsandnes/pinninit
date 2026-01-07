from __future__ import annotations

from dataclasses import field, make_dataclass
from typing import Any, Protocol

import torch

from ..checks.validate_data import validate_pytorch_tensor
from .coords_dataclass import CoordsProtocol
from .data_structure_utils import convert_to_tensor, generate_tensor_component_labels


class FieldProtocol(Protocol):
    # Attributes
    # Input fields
    rank: int
    coords: CoordsProtocol
    device: torch.device

    # Derived fields
    tensor: torch.Tensor
    component_tensor: torch.Tensor
    is_batch_only: bool
    full_tensor: torch.Tensor
    full_component_tensor: torch.Tensor
    component_labels: tuple[str, ...]
    index_to_component_label: dict[tuple[int, ...], str]
    batch_indices: slice | torch.Tensor

    # Methods
    def __post_init__(self) -> None: ...
    def _validate_input(self) -> None: ...
    def _set_tensors(self) -> None: ...
    def update_from_component_dict(self, new_components: dict[str, Any]) -> None: ...
    def update_from_component_tensor(self, new_tensor: torch.Tensor) -> None: ...
    def update_from_tensor(self, new_tensor: torch.Tensor) -> None: ...


def _field_post_init(self: FieldProtocol) -> None:
    # Generate component labels based on geometric coords and rank
    self.component_labels, self.index_to_component_label = (
        generate_tensor_component_labels(self.coords.geometric_labels, self.rank)
    )

    # Convert inputs to tensors if not already
    for label in self.component_labels:
        value = getattr(self, label)
        value_tensor = convert_to_tensor(value, self.device, requires_grad=True)
        setattr(self, label, value_tensor)

    # Validate input
    self._validate_input()

    # Set tensors
    self._set_tensors()


def tensor(self: FieldProtocol) -> torch.Tensor:
    """Get field tensor, respecting batch mode."""
    if self.coords.is_batch:
        if self.is_batch_only:
            # Field was created with batched coords, so full_tensor is already batched
            return self.full_tensor
        else:
            # Field was created with full coords, slice to current batch
            return self.full_tensor[self.coords.batch_indices]
    else:
        if self.is_batch_only:
            # Trying to access batch-only field when coords are not batched
            raise RuntimeError(
                "Cannot access batch-only field when coordinates are not in batch mode."
            )
        else:
            # Full field with full coords
            return self.full_tensor


def component_tensor(self: FieldProtocol) -> torch.Tensor:
    """Get component tensor, respecting batch mode."""
    if self.coords.is_batch:
        if self.is_batch_only:
            # Field was created with batched coords, so full_component_tensor is
            # already batched
            return self.full_component_tensor
        else:
            # Field was created with full coords, slice to current batch
            return self.full_component_tensor[self.coords.batch_indices]
    else:
        if self.is_batch_only:
            # Trying to access batch-only field when coords are not batched
            raise RuntimeError(
                "Cannot access batch-only field when coordinates are not in batch mode."
            )
        else:
            # Full field with full coords
            return self.full_component_tensor


def _validate_input(self: Any) -> None:
    # Don't allow negative rank
    if self.rank < 0:
        raise ValueError(
            f"Tensor rank must be greater than or equal to 0," f"got {self.rank}"
        )

    # Validate components
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


def _set_tensors(self: FieldProtocol) -> None:
    """Construct component tensor and tensor in shape that can be used for maths
    operations."""

    # Is this field sampled at batched coords
    self.is_batch_only = self.coords.is_batch

    # Set component tensor
    self.full_component_tensor = torch.stack(
        [getattr(self, label) for label in self.component_labels],
        dim=1,
    )

    # Get tensor shape
    n_points = self.coords.n_points
    n_geometric = len(self.coords.geometric_labels)
    tensor_shape = (n_points,) + (n_geometric,) * self.rank

    # Reshape component_tensor into structured tensor
    self.full_tensor = self.full_component_tensor.reshape(tensor_shape)


def update_from_component_dict(
    self: FieldProtocol, new_components: dict[str, Any]
) -> None:
    # Check that new labels are subset of field components
    if not set(new_components.keys()).issubset(set(self.component_labels)):
        raise KeyError(
            "New components must be subset of field component labels when "
            "updating field"
        )

    # Update components
    for label in new_components:
        value_tensor = convert_to_tensor(
            new_components[label], self.device, requires_grad=True
        )
        setattr(self, label, value_tensor)

    # Validate input
    self._validate_input()

    # Set tensors
    self._set_tensors()


def update_from_component_tensor(self: FieldProtocol, new_tensor: torch.Tensor) -> None:
    """Update the field from a component tensor of shape (n_points, n_components)"""

    # Validate input tensor
    validate_pytorch_tensor(
        new_tensor, (self.coords.n_points, len(self.component_labels))
    )

    # Update components from tensor
    for i, label in enumerate(self.component_labels):
        setattr(self, label, new_tensor[:, i])

    # Validate input
    self._validate_input()

    # Set tensors
    self._set_tensors()


def update_from_tensor(self: FieldProtocol, new_tensor: torch.Tensor) -> None:
    """
    Update the field from a structured tensor of shape
    (n_points, n_geometric, ..., n_geometric)"""

    # Validate input tensor
    n_points = self.coords.n_points
    n_geometric = len(self.coords.geometric_labels)
    expected_shape = (n_points,) + (n_geometric,) * self.rank
    validate_pytorch_tensor(new_tensor, expected_shape)

    # Flatten the tensor along the component axes to match component_tensor shape
    component_tensor = new_tensor.reshape(n_points, -1)

    self.update_from_component_tensor(component_tensor)


def make_field_dataclass(
    rank: int, coords: CoordsProtocol, device: torch.device
) -> type[FieldProtocol]:
    components: list[tuple[str, Any, Any | None]] = []

    # Each field component
    component_labels, _ = generate_tensor_component_labels(
        coords.geometric_labels, rank
    )
    for label in component_labels:
        components.append((label, Any, None))

    # Input fields
    components.extend(
        [
            ("rank", int, field(default=rank)),
            ("device", torch.device, field(default=device)),
            # Derived fields
            ("component_labels", tuple[str, ...], field(init=False)),
            ("index_to_component_label", dict[tuple[int, ...], str], field(init=False)),
            ("is_batch_only", bool, field(init=False)),
            ("full_tensor", torch.Tensor, field(init=False)),
            ("full_component_tensor", torch.Tensor, field(init=False)),
            ("batch_indices", slice | torch.Tensor, field(init=False)),
        ]
    )

    # Make dataclass instance
    Field = make_dataclass(
        "Field",
        components,
        namespace={
            "coords": coords,
            "__post_init__": _field_post_init,
            "tensor": property(tensor),
            "component_tensor": property(component_tensor),
            "_validate_input": _validate_input,
            "_set_tensors": _set_tensors,
            "update_from_component_dict": update_from_component_dict,
            "update_from_component_tensor": update_from_component_tensor,
            "update_from_tensor": update_from_tensor,
        },
    )

    return Field
