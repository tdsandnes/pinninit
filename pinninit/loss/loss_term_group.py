from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ..data_structure.data_container import CoordDataContainer
from ..io.input_parameters import EquationsParams, FieldsParams
from ..io.io_utils import get_enum_label_from_value
from .loss_function import LossFunctionLabels
from .loss_term import LOSS_TERM_REGISTRY, LossTerm

if TYPE_CHECKING:
    from typing import Any

    from ..data_structure.coords_dataclass import CoordsProtocol
    from ..nn.neural_network import NeuralNetwork


class LossTermCoordGroup:
    """Grouping of loss terms"""

    def __init__(
        self,
        coords: CoordsProtocol,
        available_fields_params: FieldsParams,
        available_equations_params: EquationsParams,
        input_fields: dict[str, Any] | None = None,
        input_consts: dict[str, Any] | None = None,
    ) -> None:
        # Set coords and corresponding data containers
        self.coords = coords
        self.device = coords.device
        self.dim_geometry = coords.coord_system.dim_geometry
        self.input_data = CoordDataContainer(
            self.coords, available_fields_params, available_equations_params
        )
        self.output_data = CoordDataContainer(
            self.coords, available_fields_params, available_equations_params
        )

        # Load fields into containers
        if input_fields is not None:
            for field_label, value in input_fields.items():
                self.input_data.set_field_from_component_dict(field_label, value)

        # Load consts into containers
        if input_consts is not None:
            for const_label, value in input_consts.items():
                self.input_data.set_const(const_label, value)
                ### Maybe also add consts to output_data, but come back to this when
                ### looking at learnable consts

        # Init as empty group
        self.loss_terms: list[LossTerm] = []

    def add(self, loss_term: LossTerm) -> None:
        """Add a specific loss term to the group."""
        self.loss_terms.append(loss_term)

    def init_terms_and_add(
        self,
        loss_terms: list[dict[str, Any]],
        fields_params: FieldsParams,
    ) -> None:
        """Init multiple loss terms and add to group"""
        # Field label to rank dict
        fields = (
            fields_params.input_fields
            + fields_params.output_fields
            + fields_params.derivable_fields
        )
        field_label_to_rank_dict = {f["label"]: f["rank"] for f in fields}

        for term in loss_terms:
            field_label = term["field_label"]
            loss_type = term["loss_type"]
            rank = field_label_to_rank_dict[field_label]
            components = 1 if rank == 0 else self.dim_geometry**rank

            ### Eventually tidy this up with aliases similar to activation function
            loss_functions = (
                term["loss_function"] if components > 1 else [term["loss_function"]]
            )
            weight_methods = (
                term["weight_method"] if components > 1 else [term["weight_method"]]
            )
            weight_values = (
                term["weight_value"] if components > 1 else [term["weight_value"]]
            )

            for idx in range(components):
                loss_function = get_enum_label_from_value(
                    loss_functions[idx], LossFunctionLabels
                )
                weight_method = weight_methods[idx]
                weight_value = weight_values[idx]

                # Init loss term
                loss_term = LOSS_TERM_REGISTRY[loss_type](
                    field_label, idx, loss_function, weight_method, weight_value
                )

                # Add loss term to group
                self.add(loss_term)

    def remove(self, loss_term: LossTerm) -> None:
        """Remove a specific loss term from the group."""
        self.loss_terms.remove(loss_term)

    def apply_model(
        self,
        model: NeuralNetwork,
        output_fields_params: list[dict[str, Any]],
    ) -> None:
        """Applies NN model and evaluates all equations needed for loss term group"""

        # Get model output
        output_fields = model(self.coords.internal)

        # Add output fields
        idx = 0
        for field in output_fields_params:
            # Get field info
            field_label = field["label"]
            rank = field["rank"]

            # Extract field from model output and set in output data container
            idx_end = idx + (1 if rank == 0 else rank * self.dim_geometry)
            value = output_fields[:, idx:idx_end]

            # Set field in output data container
            self.output_data.set_field_from_component_tensor(field_label, value)

            # Update index for next field
            idx = idx_end

    def compute_loss(
        self,
    ) -> torch.Tensor:
        loss = torch.tensor(0.0, device=self.device)
        for loss_term in self.loss_terms:
            loss += loss_term.compute_loss(self.input_data, self.output_data)

        return loss

    def compute_rms_residual_per_point(self) -> torch.Tensor:
        """Compute RMS error per point for all loss terms in the group."""
        n_points = self.coords.n_points
        rms_error = torch.zeros(n_points, device=self.device)
        for loss_term in self.loss_terms:
            residual = loss_term.compute_residual(self.input_data, self.output_data)
            rms_error += residual**2
        rms_error = torch.sqrt(rms_error / len(self.loss_terms))
        return rms_error
