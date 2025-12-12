from __future__ import annotations

from dataclasses import field, make_dataclass
from typing import Any, Protocol

import torch

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
    component_labels: tuple[str, ...]

    # Methods
    def __post_init__(self) -> None: ...
    def _validate_input(self) -> None: ...
    def _stack_tensor_field(
        self, labels: tuple[str, ...], rank: int
    ) -> torch.Tensor: ...
    def update(self, new_components: dict[str, Any]) -> None: ...


def _field_post_init(self: FieldProtocol) -> None:
    # Generate component labels based on geometric coords and rank
    self.component_labels = generate_tensor_component_labels(
        self.coords.geometric_labels, self.rank
    )

    # Convert inputs to tensors if not already
    for label in self.component_labels:
        value = getattr(self, label)
        value_tensor = convert_to_tensor(value, self.device, requires_grad=True)
        setattr(self, label, value_tensor)

    # Validate input
    self._validate_input()

    # Set tensor
    if self.rank == 0:
        # Scalar field
        self.tensor = getattr(self, self.component_labels[0])
    else:
        # Vector/tensor field, ordered by geometric coords
        self.tensor = self._stack_tensor_field(self.component_labels, self.rank)


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
            len_str = len(coord)
            sub_labels = [
                label[len_str:] for label in labels if label.startswith(coord)
            ]
            # recursively stack along remaining coords
            grouped.append(self._stack_tensor_field(tuple(sub_labels), rank - 1))
        return torch.stack(grouped, dim=1)


def update(self: FieldProtocol, new_components: dict[str, Any]) -> None:
    # Check that new labels are subset of field components
    if not set(new_components.keys()).issubset(set(self.component_labels)):
        raise KeyError(
            "New components must be subset of field component labels when "
            "updating field"
        )

    # Update labels
    for label in new_components:
        value_tensor = convert_to_tensor(
            new_components[label], self.device, requires_grad=True
        )
        setattr(self, label, value_tensor)

    # Validate input
    self._validate_input()

    # Set tensor
    if self.rank == 0:
        # Scalar field
        self.tensor = getattr(self, self.component_labels[0])
    else:
        # Vector/tensor field, ordered by geometric coords
        self.tensor = self._stack_tensor_field(self.component_labels, self.rank)


def make_field_dataclass(
    rank: int, coords: CoordsProtocol, device: torch.device
) -> type[FieldProtocol]:
    components: list[tuple[str, Any, Any | None]] = []

    # Each field component
    component_labels = generate_tensor_component_labels(coords.geometric_labels, rank)
    for label in component_labels:
        components.append((label, Any, None))

    # Input fields
    components.extend(
        [
            ("rank", int, field(default=rank)),
            ("device", torch.device, field(default=device)),
            # Derived fields
            ("component_labels", tuple[str, ...], field(init=False)),
            ("tensor", torch.Tensor, field(init=False)),
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
            "_stack_tensor_field": _stack_tensor_field,
            "update": update,
        },
    )

    return Field
