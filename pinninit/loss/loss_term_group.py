from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch.utils.data import DataLoader

from ..data_structure.batching import BatchingDataset
from ..data_structure.data_container import CoordDataContainer
from ..io.input_parameters import EquationsParams, FieldsParams
from ..io.io_utils import get_enum_label_from_value
from .loss_function import LossFunctionLabels
from .loss_term import LOSS_TERM_REGISTRY, LossTerm

if TYPE_CHECKING:
    from typing import Any

    from ..data_structure.coords_dataclass import CoordsProtocol
    from ..nn.neural_network import NeuralNetwork


class LossTermGroup:
    """Grouping of loss terms"""

    def __init__(
        self,
        train_coords: CoordsProtocol,
        val_coords: CoordsProtocol,
        dim_geometry: int,
        available_fields_params: FieldsParams,
        available_equations_params: EquationsParams,
        device: torch.device,
        train_input_fields: dict[str, Any] | None = None,
        val_input_fields: dict[str, Any] | None = None,
        input_consts: dict[str, Any] | None = None,
        batch_size: int | None = None,
        shuffle_train: bool = True,
        num_workers: int = 0,
    ) -> None:
        self.train_coords = train_coords
        self.val_coords = val_coords
        self.dim_geometry = dim_geometry
        self.available_fields_params = available_fields_params
        self.available_equations_params = available_equations_params
        self.device = device
        self.batch_size = batch_size
        self.shuffle_train = shuffle_train
        self.num_workers = num_workers

        # Check that all coords have same dimensions
        assert (
            train_coords.coord_system.dim_geometry == dim_geometry
        ), "Training coordinate system dimensions must match dim of loss term group."
        assert (
            val_coords.coord_system.dim_geometry == dim_geometry
        ), "Validation coordinate system dimensions must match dim of loss term group."

        # Init data containers
        self.train_input_data, self.train_output_data = self._init_data_containers(
            self.train_coords, train_input_fields, input_consts
        )
        self.val_input_data, self.val_output_data = self._init_data_containers(
            self.val_coords, val_input_fields, input_consts
        )

        # Init data loaders
        self._init_data_loaders()

        # Init loss terms as empty group
        self.loss_terms: list[LossTerm] = []

    def _init_data_containers(
        self,
        coords: CoordsProtocol,
        input_fields: dict[str, Any] | None = None,
        input_consts: dict[str, Any] | None = None,
    ) -> tuple[CoordDataContainer, CoordDataContainer]:
        """Init input and output data containers for a set of coords"""

        input_data = CoordDataContainer(
            coords, self.available_fields_params, self.available_equations_params
        )
        output_data = CoordDataContainer(
            coords, self.available_fields_params, self.available_equations_params
        )

        # Load fields into containers
        if input_fields is not None:
            for field_label, value in input_fields.items():
                input_data.set_field_from_component_dict(field_label, value)

        # Load consts into containers
        if input_consts is not None:
            for const_label, value in input_consts.items():
                input_data.set_const(const_label, value)
                ### Maybe also add consts to output_data, but come back to this when
                ### looking at learnable consts

        return input_data, output_data

    def _init_data_loaders(self) -> None:
        self.train_loader: DataLoader[int] | None = None
        self.val_loader: DataLoader[int] | None = None

        if self.batch_size is not None:
            train_dataset = BatchingDataset(self.train_coords.n_points)
            self.train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=self.shuffle_train,
                num_workers=self.num_workers,
                collate_fn=torch.utils.data.default_collate,
                pin_memory=torch.cuda.is_available(),
            )

            val_dataset = BatchingDataset(self.val_coords.n_points)
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                collate_fn=torch.utils.data.default_collate,
                pin_memory=torch.cuda.is_available(),
            )

    def add(self, loss_term: LossTerm) -> None:
        """Add a specific loss term to the group."""
        self.loss_terms.append(loss_term)

    def init_loss_terms_and_add(
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

    def _apply_model_at_coords(
        self,
        model: NeuralNetwork,
        coords: CoordsProtocol,
        output_data: CoordDataContainer,
    ) -> None:
        """Applies NN model stores output in output data container"""

        # Get model output
        output_fields = model(coords.internal)

        # Add output fields
        idx = 0
        for field_label, rank in model.output_field_label_to_rank.items():
            # Extract field from model output and set in output data container
            idx_end = idx + (1 if rank == 0 else rank * self.dim_geometry)
            value = output_fields[:, idx:idx_end]

            # Set field in output data container
            output_data.set_field_from_component_tensor(field_label, value)

            # Update index for next field
            idx = idx_end

    def _compute_loss(
        self,
        input_data: CoordDataContainer,
        output_data: CoordDataContainer,
    ) -> torch.Tensor:
        # Add contributions from all loss terms
        loss = torch.tensor(0.0, device=self.device)
        for loss_term in self.loss_terms:
            loss += loss_term.compute_loss(input_data, output_data)

        return loss

    def _compute_loss_batched(
        self,
        model: NeuralNetwork,
        input_data: CoordDataContainer,
        output_data: CoordDataContainer,
        dataloader: DataLoader[int] | None,
    ) -> torch.Tensor:
        """Compute loss over batches using a DataLoader"""
        if dataloader is not None and self.batch_size is not None:
            total_loss = torch.tensor(0.0, device=self.device)
            total_points: int = 0

            # Loop through batches
            for batch_indices in dataloader:
                batch_coords = input_data.coords[batch_indices]

                # Slice input and output data containers
                input_batch = input_data.slice_with_coords(batch_indices, batch_coords)
                output_batch = output_data.slice_with_coords(
                    batch_indices, batch_coords
                )

                # Apply model
                self._apply_model_at_coords(model, batch_coords, output_batch)

                # Compute loss
                batch_loss = self._compute_loss(input_batch, output_batch)

                # Add loss to total loss and update counter
                batch_size = batch_indices.numel()
                total_loss += batch_loss * batch_size
                total_points += batch_size

            return total_loss / total_points

        else:
            # Full-batch
            coords = input_data.coords

            self._apply_model_at_coords(model, coords, output_data)
            return self._compute_loss(input_data, output_data)

    def compute_loss_train(self, model: NeuralNetwork) -> torch.Tensor:
        return self._compute_loss_batched(
            model, self.train_input_data, self.train_output_data, self.train_loader
        )

    def compute_loss_val(self, model: NeuralNetwork) -> torch.Tensor:
        return self._compute_loss_batched(
            model, self.val_input_data, self.val_output_data, self.val_loader
        )
