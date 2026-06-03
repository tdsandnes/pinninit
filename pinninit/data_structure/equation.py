from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Callable

    from ..differential_equations.operators import Operators
    from .coords_dataclass import CoordsProtocol
    from .data_container import DataContainer


@dataclass(frozen=True, slots=True)
class Equation:
    label: str
    function: Callable[..., torch.Tensor | tuple[torch.Tensor, ...]]
    output_label_to_rank: dict[str, int]  # Sorted in order outputted by function
    input_labels: tuple[str, ...] = field(init=False)

    def __post_init__(self) -> None:
        # Compute input labels from function signature and store as property
        signature = inspect.signature(self.function)

        # Check that function accepts 'operators'
        if "operators" not in signature.parameters:
            raise ValueError(
                f"Equation '{self.label}' function must accept 'operators' argument."
            )

        # Exclude 'operators' from input labels
        input_labels = tuple(
            name for name in signature.parameters if name != "operators"
        )
        object.__setattr__(self, "input_labels", input_labels)

        # Check valid ranks
        for label, rank in self.output_label_to_rank.items():
            if rank < 0:
                raise ValueError(f"Output rank for '{label}' must be >= 0.")

    @property
    def output_labels(self) -> tuple[str, ...]:
        return tuple(self.output_label_to_rank.keys())

    def evaluate(
        self,
        data_container: DataContainer,
        eval_coords: CoordsProtocol,
        operators: Operators,
    ) -> dict[str, torch.Tensor]:
        # Gather inputs from data container
        inputs = self._gather_inputs(data_container, eval_coords)

        # Evaluate function
        outputs = self.function(**inputs, operators=operators)

        # Make tuple even if single output
        if not isinstance(outputs, tuple):
            outputs = (outputs,)

        # Validate number of outputs
        if len(outputs) != len(self.output_labels):
            raise ValueError(
                f"Equation {self.label} returned {len(outputs)} outputs, "
                f"expected {len(self.output_labels)}."
            )

        output_field_dict: dict[str, torch.Tensor] = {}
        for (label, rank), output in zip(
            self.output_label_to_rank.items(), outputs, strict=True
        ):
            # Check output type
            if not isinstance(output, torch.Tensor):
                raise TypeError(
                    f"Output '{label}' from equation '{self.label}' "
                    f"must be a torch.Tensor."
                )

            # Check output shape
            expected_shape = (eval_coords.n_points,) + (
                len(eval_coords.geometric_labels),
            ) * rank
            if output.shape != expected_shape:
                raise ValueError(
                    f"Output '{label}' from equation '{self.label}' has shape "
                    f"{output.shape}, expected {expected_shape} for rank {rank}."
                )

            # Store output
            output_field_dict[label] = output

        return output_field_dict

    def _gather_inputs(
        self, data_container: DataContainer, eval_coords: CoordsProtocol
    ) -> dict[str, torch.Tensor]:
        value: torch.Tensor | None = None
        inputs: dict[str, torch.Tensor] = {}
        for label in self.input_labels:
            matches = 0

            # Check coords
            if label in data_container.coord_labels:
                value = eval_coords.get_coord(label)
                matches += 1

            # Check fields
            if label in data_container.field_labels:
                if data_container.fields_active_coords[label] is None:
                    raise ValueError(f"Field '{label}' has not been initialised.")
                value = data_container.fields[label].tensor(eval_coords)
                matches += 1

            # Check consts
            if label in data_container.consts:
                if data_container.consts_active_coords[label] is None:
                    raise ValueError(f"Constant '{label}' has not been initialised.")
                value = data_container.consts[label]
                matches += 1

            # Check only one label match was found
            if matches == 0:
                raise KeyError(f"Label '{label}' not found in data container.")
            elif matches > 1:
                raise ValueError(f"Duplicate labels '{label}' found in data container.")

            # Make sure value has been set in data container
            if value is None:
                raise ValueError(f"Unexpected '{label}' has value None.")

            inputs[label] = value

        return inputs
