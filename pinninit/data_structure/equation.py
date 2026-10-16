from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from graphlib import TopologicalSorter
from typing import TYPE_CHECKING

import torch

from .field import tensor_shape

if TYPE_CHECKING:
    from collections.abc import Callable

    from ..differential_equations.operators import Operators
    from .coords import Coords
    from .data_container import DataContainer


@dataclass(frozen=True, slots=True)
class Equation:
    """
    An equation computing one or more fields from coords, fields and consts.

    The function's argument names are the labels of its inputs, plus an
    'operators' argument for taking derivatives. It returns one tensor per
    output, in the order of output_label_to_rank.
    """

    label: str
    function: Callable[..., torch.Tensor | tuple[torch.Tensor, ...]]
    output_label_to_rank: dict[str, int]
    input_labels: tuple[str, ...] = field(init=False)

    def __post_init__(self) -> None:
        signature = inspect.signature(self.function)
        if "operators" not in signature.parameters:
            raise ValueError(
                f"Equation '{self.label}' function must accept 'operators' argument."
            )

        input_labels = tuple(
            name for name in signature.parameters if name != "operators"
        )
        object.__setattr__(self, "input_labels", input_labels)

        for label, rank in self.output_label_to_rank.items():
            if rank < 0:
                raise ValueError(f"Output rank for '{label}' must be >= 0.")

    @property
    def output_labels(self) -> tuple[str, ...]:
        return tuple(self.output_label_to_rank)

    def evaluate(
        self,
        data_container: DataContainer,
        eval_coords: Coords,
        operators: Operators,
    ) -> dict[str, torch.Tensor]:
        inputs = self._gather_inputs(data_container, eval_coords)
        outputs = self.function(**inputs, operators=operators)
        if not isinstance(outputs, tuple):
            outputs = (outputs,)

        if len(outputs) != len(self.output_labels):
            raise ValueError(
                f"Equation {self.label} returned {len(outputs)} outputs, "
                f"expected {len(self.output_labels)}."
            )

        for (label, rank), output in zip(
            self.output_label_to_rank.items(), outputs, strict=True
        ):
            if not isinstance(output, torch.Tensor):
                raise TypeError(
                    f"Output '{label}' from equation '{self.label}' "
                    f"must be a torch.Tensor."
                )

            expected_shape = tensor_shape(eval_coords, rank)
            if output.shape != expected_shape:
                raise ValueError(
                    f"Output '{label}' from equation '{self.label}' has shape "
                    f"{output.shape}, expected {expected_shape} for rank {rank}."
                )

        return dict(zip(self.output_labels, outputs, strict=True))

    def _gather_inputs(
        self, data_container: DataContainer, eval_coords: Coords
    ) -> dict[str, torch.Tensor]:
        inputs: dict[str, torch.Tensor] = {}
        for label in self.input_labels:
            if label in data_container.coord_labels:
                inputs[label] = eval_coords.get_coord(label)
            elif label in data_container.field_labels:
                inputs[label] = data_container.get_field_tensor(label, eval_coords)
            elif label in data_container.consts:
                inputs[label] = data_container.get_const(label)
            else:
                raise KeyError(f"Label '{label}' not found in data container.")
        return inputs


def sort_equations(equations: list[Equation]) -> list[Equation]:
    """Order equations so each one comes after the equations it depends on."""
    equations_by_label = {equation.label: equation for equation in equations}
    if len(equations_by_label) != len(equations):
        raise ValueError("Equation labels must be unique.")

    producers: dict[str, Equation] = {}
    for equation in equations:
        for output_label in equation.output_labels:
            if output_label in producers:
                raise ValueError(
                    f"Field '{output_label}' is output by more than one equation."
                )
            producers[output_label] = equation

    dependencies = {
        equation.label: {
            producers[label].label
            for label in equation.input_labels
            if label in producers
        }
        for equation in equations
    }

    order = TopologicalSorter(dependencies).static_order()
    return [equations_by_label[label] for label in order]
