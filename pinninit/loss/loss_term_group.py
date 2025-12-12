from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ..data_structure.data_container import CoordDataContainer
from .loss_term import LossTerm

if TYPE_CHECKING:
    from typing import Any

    from ..data_structure.coords_dataclass import CoordsProtocol
    from ..nn.neural_network import NeuralNetwork


class LossTermCoordGroup:
    """Grouping of loss terms based on the coords that they're evaluated at"""

    def __init__(
        self,
        coords: CoordsProtocol,
        input_field_labels: set[str],
        output_field_labels: set[str],
        input_fields: dict[str, Any] | None = None,
    ):
        self.coords = coords
        self.device = coords.device

        ### Temp make this an input and in a form coming straight from pinns_params
        equations_dict = {
            "div_B": {"input_args": "B", "output_args": {"div_B": 0}},
            "B_from_V": {"input_args": "V", "output_args": {"B": 1}},
        }
        ### Temp
        self.input_field_labels = input_field_labels
        self.output_field_labels = output_field_labels
        input_field_labels_and_ranks = dict.fromkeys(input_field_labels, 1)
        output_field_labels_and_ranks = dict.fromkeys(output_field_labels, 0)
        self.input_data = CoordDataContainer(
            coords, input_field_labels_and_ranks, (), equations_dict
        )
        self.output_data = CoordDataContainer(
            coords, output_field_labels_and_ranks, (), equations_dict
        )

        # Load fields and consts into container
        if input_fields is not None:
            for field, value in input_fields.items():
                self.input_data.update_field(field, value)

        ###for const, value in input_consts.items():
        ### self.input_data.update_const(const, value)

        # Init as empty group
        self.loss_terms: list[LossTerm] = []

    def add(self, loss_term: LossTerm) -> None:
        """Add a specific loss term to the group."""
        self.loss_terms.append(loss_term)

    def remove(self, loss_term: LossTerm) -> None:
        """Remove a specific loss term from the group."""
        self.loss_terms.remove(loss_term)

    def apply_model(
        self,
        model: NeuralNetwork,
    ) -> None:
        """Applies NN model and evaluates all equations needed for loss term group"""
        # Get model output
        output_fields = model(self.coords.internal)

        ### Temp ####
        label = "V"
        output_fields_dict = {"value": output_fields.squeeze(1)}

        # Add output fields
        self.output_data.update_field(label, output_fields_dict)

        # Evaluate equations
        self.input_data.evaluate_equations()
        self.output_data.evaluate_equations()

    def compute_loss(self) -> torch.Tensor:
        loss = torch.tensor(0.0, device=self.device)
        for loss_term in self.loss_terms:
            loss += loss_term.compute_loss(self.input_data, self.output_data)

        return loss
