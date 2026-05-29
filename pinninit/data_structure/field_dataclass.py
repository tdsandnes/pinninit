from __future__ import annotations

from dataclasses import field, make_dataclass
from itertools import product
from typing import Any, Protocol

import torch

from .coords_dataclass import CoordsProtocol


class FieldProtocol(Protocol):
    # Attributes
    # Input fields
    rank: int
    coords: CoordsProtocol  # bound as class attribute in make_field_dataclass
    device: torch.device

    # Derived fields
    _tensor: torch.Tensor
    _component_tensor: torch.Tensor
    component_labels: tuple[str, ...]

    # Methods
    def __post_init__(self) -> None: ...
    def _validate_input(self) -> None: ...
    def _set_tensors(self) -> None: ...
    def _loop_parent_component_tensor(self, coords: CoordsProtocol) -> torch.Tensor: ...
    def component_tensor(self, coords: CoordsProtocol) -> torch.Tensor: ...
    def tensor(self, coords: CoordsProtocol) -> torch.Tensor: ...
    def update_from_component_dict(
        self, new_components: dict[str, torch.Tensor]
    ) -> None: ...
    def update_from_component_tensor(self, new_tensor: torch.Tensor) -> None: ...
    def update_from_tensor(self, new_tensor: torch.Tensor) -> None: ...


def _field_post_init(self: FieldProtocol) -> None:
    # Validate input
    self._validate_input()

    # Set tensors
    self._set_tensors()


def _set_tensors(self: FieldProtocol) -> None:
    """Construct component tensor and tensor in shape that can be used for maths
    operations."""

    # Set component tensor
    self._component_tensor = torch.stack(
        [getattr(self, label) for label in self.component_labels],
        dim=1,
    )

    # Get tensor shape
    # Note: rank 0 tensor shape is (n_points,) while component_tensor is (n_points, 1).
    tensor_shape = (self.coords.n_points,) + (
        len(self.coords.geometric_labels),
    ) * self.rank

    # Reshape component_tensor into structured tensor
    self._tensor = self._component_tensor.reshape(tensor_shape)


def _loop_parent_component_tensor(
    self: FieldProtocol, coords: CoordsProtocol
) -> torch.Tensor:
    # If full original coords
    if coords is self.coords:
        return self._component_tensor

    if not coords.is_batch or coords.parent is None:
        raise ValueError(
            "Evaluation coordinates must be a valid subset of field coordinates"
        )

    # Recursivly get parent component tensor until we reach the full dataset
    parent_tensor = self._loop_parent_component_tensor(coords.parent)

    # Slice the parent using the current sub-batch index mapping
    return parent_tensor[coords.parent_idx]


def component_tensor(self: FieldProtocol, coords: CoordsProtocol) -> torch.Tensor:
    # Loop through parent component tensors until we reach the full dataset,
    # then slice down to the current batch
    batch_component_tensor = self._loop_parent_component_tensor(coords)

    # Detach and set requires_grad
    return batch_component_tensor.detach().requires_grad_(True)


def tensor(self: FieldProtocol, coords: CoordsProtocol) -> torch.Tensor:
    # If full original coords
    if coords is self.coords:
        return self._tensor

    batch_component_tensor = self.component_tensor(coords)

    tensor_shape = (coords.n_points,) + (len(coords.geometric_labels),) * self.rank

    # Reshape component_tensor into structured tensor
    return batch_component_tensor.reshape(tensor_shape)


def update_from_component_dict(
    self: FieldProtocol, new_components: dict[str, torch.Tensor]
) -> None:
    # Check that new labels are subset of field components
    if not set(new_components.keys()).issubset(set(self.component_labels)):
        raise KeyError(
            "New components must be subset of field component labels when "
            "updating field"
        )

    # Update components
    for label in new_components:
        setattr(self, label, new_components[label])

    # Validate input
    self._validate_input()

    # Set tensors
    self._set_tensors()


def update_from_component_tensor(self: FieldProtocol, new_tensor: torch.Tensor) -> None:
    """Update the field from a component tensor of shape (n_points, n_components)"""

    # Validate input tensor
    if not isinstance(new_tensor, torch.Tensor):
        raise TypeError("Component tensor given is not a torch.Tensor")

    if new_tensor.shape != self._component_tensor.shape:
        raise ValueError(
            f"Expected component tensor of shape {self._component_tensor.shape} "
            f"got {new_tensor.shape}"
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

    # Flatten the tensor along the component axes to match component_tensor shape
    component_tensor = new_tensor.reshape(self.coords.n_points, -1)

    self.update_from_component_tensor(component_tensor)


def _validate_input(self: Any) -> None:
    # Validate components
    for label in self.component_labels:
        if not hasattr(self, label):
            raise ValueError(f"Coordinate label '{label}' not found")

        component = getattr(self, label)

        if not isinstance(component, torch.Tensor):
            raise TypeError(f"Field component {label} is not a torch.Tensor")

        if component.shape != (self.coords.n_points,):
            raise ValueError(
                f"Field component {label} must have shape matching "
                f"coordinate points ({self.coords.n_points},), got "
                f"shape {component.shape}"
            )


def generate_tensor_component_labels(
    geometric_labels: tuple[str, ...], rank: int
) -> tuple[str, ...]:
    """
    Generate labels for each component of a rank-n tensor field.

    Parameters
    ----------
    geometric_labels : list[str]
        List of geometric labels.
    rank : int
        Rank of tensor

    Returns
    -------
    tuple[str]
        Tuple for tensor components.
    """
    if rank < 0:
        raise ValueError("rank must be >= 0")
    if rank == 0:
        return ("value",)

    indices = tuple(product(range(len(geometric_labels)), repeat=rank))

    index_to_name = {idx: "_".join(geometric_labels[i] for i in idx) for idx in indices}
    labels = tuple(index_to_name[idx] for idx in indices)
    return labels


def make_field_dataclass(
    rank: int, coords: CoordsProtocol, device: torch.device
) -> type[FieldProtocol]:
    components: list[tuple[str, Any, Any | None]] = []

    # Each field component
    component_labels = generate_tensor_component_labels(coords.geometric_labels, rank)
    for label in component_labels:
        components.append((label, torch.Tensor, None))

    # Input fields
    components.extend(
        [
            ("rank", int, field(default=rank)),
            ("device", torch.device, field(default=device)),
            ("component_labels", tuple[str, ...], field(default=component_labels)),
            # Derived fields
            ("_tensor", torch.Tensor, field(default=None, init=False)),
            ("_component_tensor", torch.Tensor, field(default=None, init=False)),
        ]
    )

    # Make dataclass instance
    Field = make_dataclass(
        "Field",
        components,
        namespace={
            "coords": coords,
            "__post_init__": _field_post_init,
            "_validate_input": _validate_input,
            "_set_tensors": _set_tensors,
            "_loop_parent_component_tensor": _loop_parent_component_tensor,
            "component_tensor": component_tensor,
            "tensor": tensor,
            "update_from_component_dict": update_from_component_dict,
            "update_from_component_tensor": update_from_component_tensor,
            "update_from_tensor": update_from_tensor,
        },
    )

    return Field
