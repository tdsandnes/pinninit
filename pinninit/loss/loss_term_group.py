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
        dim_geometry: int,
        available_fields_params: FieldsParams,
        available_equations_params: EquationsParams,
        device: torch.device,
        train_input_fields: dict[str, Any] | None = None,
        val_coords: CoordsProtocol | None = None,
        val_input_fields: dict[str, Any] | None = None,
        input_consts: dict[str, Any] | None = None,
        weight_method_label: str | None = None,
        weight_group_label: str | None = None,
        initial_weight: torch.Tensor | None = None,
        batch_size_train: int | None = None,
        batch_size_val: int | None = None,
        shuffle_train: bool = True,
        shuffle_val: bool = False,
        num_workers_train: int = 0,
        num_workers_val: int = 0,
        staggering: int = 0,
    ) -> None:
        self.train_coords = train_coords
        self.val_coords = val_coords
        self.dim_geometry = dim_geometry
        self.available_fields_params = available_fields_params
        self.available_equations_params = available_equations_params
        self.device = device

        # Check that all coords have same dimensions
        assert (
            train_coords.coord_system.dim_geometry == dim_geometry
        ), "Training coord system dimensions must match dim of loss term group."

        # Init data containers
        self.train_input_data, self.train_output_data = self._init_data_containers(
            self.train_coords, train_input_fields, input_consts
        )

        self.val_input_data: CoordDataContainer | None = None
        self.val_output_data: CoordDataContainer | None = None
        if val_coords is not None:
            # Check that all coords have same dimensions
            assert (
                val_coords.coord_system.dim_geometry == dim_geometry
            ), "Validation coord system dimensions must match dim of loss term group."

            self.val_input_data, self.val_output_data = self._init_data_containers(
                val_coords, val_input_fields, input_consts
            )

        # Init weight
        self.weight_method_label = weight_method_label
        self.weight_group_label = weight_group_label
        self.initial_weight = initial_weight
        self.weight = (
            initial_weight
            if initial_weight is not None
            else torch.tensor(1.0, device=self.device)
        )

        # Init batching
        self._default_batch_size_train = batch_size_train
        self._default_batch_size_val = batch_size_val
        self._default_shuffle_train = shuffle_train
        self._default_shuffle_val = shuffle_val
        self._default_num_workers_train = num_workers_train
        self._default_num_workers_val = num_workers_val
        self.batch_size_train = self._default_batch_size_train
        self.batch_size_val = self._default_batch_size_val
        self.shuffle_train = self._default_shuffle_train
        self.shuffle_val = self._default_shuffle_val
        self.num_workers_train = self._default_num_workers_train
        self.num_workers_val = self._default_num_workers_val
        self._set_train_val_data_loaders()
        self.is_default_batching = True

        # Init staggering
        self.staggering = staggering

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

    def _set_train_val_data_loaders(self) -> None:
        n_points_train = self.train_coords.full_tensor.shape[0]
        self.train_loader = self._set_data_loader(
            n_points_train,
            self.batch_size_train,
            self.shuffle_train,
            self.num_workers_train,
        )
        if self.val_coords is not None:
            n_points_val = self.val_coords.full_tensor.shape[0]
            self.val_loader = self._set_data_loader(
                n_points_val,
                self.batch_size_val,
                self.shuffle_val,
                self.num_workers_val,
            )

    def _set_data_loader(
        self,
        n_points: int,
        batch_size: int | None = None,
        shuffle: bool = True,
        num_workers: int = 0,
    ) -> DataLoader[int] | None:
        if batch_size is not None:
            dataset = BatchingDataset(n_points)
            return DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                collate_fn=torch.utils.data.default_collate,
                pin_memory=torch.cuda.is_available(),
            )
        else:
            return None

    def set_batching(
        self,
        batch_size_train: int | None,
        batch_size_val: int | None,
        shuffle_train: bool = True,
        shuffle_val: bool = False,
        num_workers_train: int = 0,
        num_workers_val: int = 0,
    ) -> None:
        # Early return if already has these exact settings
        if (
            batch_size_train == self.batch_size_train
            and batch_size_val == self.batch_size_val
            and shuffle_train == self.shuffle_train
            and shuffle_val == self.shuffle_val
            and num_workers_train == self.num_workers_train
            and num_workers_val == self.num_workers_val
        ):
            return

        if (
            batch_size_train == self._default_batch_size_train
            and batch_size_val == self._default_batch_size_val
            and shuffle_train == self._default_shuffle_train
            and shuffle_val == self._default_shuffle_val
            and num_workers_train == self._default_num_workers_train
            and num_workers_val == self._default_num_workers_val
        ):
            # If values given are defaults, then reset to default
            self.reset_to_default_batching()
        else:
            # Else update to new values
            self.batch_size_train = batch_size_train
            self.batch_size_val = batch_size_val
            self.shuffle_train = shuffle_train
            self.shuffle_val = shuffle_val
            self.num_workers_train = num_workers_train
            self.num_workers_val = num_workers_val
            self._set_train_val_data_loaders()
            self.is_default_batching = False

            # Print message
            print(
                f"Updated batching of {self.__class__.__name__} (id={id(self)}): "
                f"batch_size_train={self.batch_size_train}, "
                f"batch_size_val={self.batch_size_val}, "
                f"shuffle_train={self.shuffle_train}, shuffle_val={self.shuffle_val}, "
                f"num_workers_train={self.num_workers_train}, "
                f"num_workers_val={self.num_workers_val}"
            )

    def reset_to_default_batching(self) -> None:
        # Only reset if not already default
        if not self.is_default_batching:
            self.batch_size_train = self._default_batch_size_train
            self.batch_size_val = self._default_batch_size_val
            self.shuffle_train = self._default_shuffle_train
            self.shuffle_val = self._default_shuffle_val
            self.num_workers_train = self._default_num_workers_train
            self.num_workers_val = self._default_num_workers_val
            self._set_train_val_data_loaders()
            self.is_default_batching = True

            # Print message
            print(
                f"Reset batching of {self.__class__.__name__} (id={id(self)}): "
                f"batch_size_train={self.batch_size_train}, "
                f"batch_size_val={self.batch_size_val}, "
                f"shuffle_train={self.shuffle_train}, shuffle_val={self.shuffle_val}, "
                f"num_workers_train={self.num_workers_train}, "
                f"num_workers_val={self.num_workers_val}"
            )

    def add(self, loss_term: LossTerm) -> None:
        """Add a specific loss term to the group."""
        self.loss_terms.append(loss_term)

    def init_loss_term_and_add(
        self,
        loss_term: dict[str, Any],
        fields_params: FieldsParams,
    ) -> None:
        field_label = loss_term["field_label"]
        loss_type = loss_term["loss_type"]

        # Dict of field to rank
        field_to_rank = {
            f["label"]: f["rank"]
            for f in (
                fields_params.input_fields
                + fields_params.output_fields
                + fields_params.derivable_fields
            )
        }
        rank = field_to_rank[field_label]
        n_components = 1 if rank == 0 else self.dim_geometry**rank

        # Register field for input_data and output_data if it's not already there
        input_field_labels = {f["label"] for f in fields_params.input_fields}
        output_field_labels = {f["label"] for f in fields_params.output_fields}
        if field_label not in input_field_labels:
            self.train_input_data.register_required_field(field_label)
            if self.val_input_data is not None:
                self.val_input_data.register_required_field(field_label)
        if field_label not in output_field_labels:
            self.train_output_data.register_required_field(field_label)
            if self.val_output_data is not None:
                self.val_output_data.register_required_field(field_label)

        # Standardise inputs to lists if they aren't already
        loss_functions = loss_term["loss_function"]
        if not isinstance(loss_functions, list):
            loss_functions = [loss_functions] * n_components

        for component_idx in range(n_components):
            loss_function_label = get_enum_label_from_value(
                loss_functions[component_idx], LossFunctionLabels
            )

            self.add(
                LOSS_TERM_REGISTRY[loss_type](
                    field_label=field_label,
                    component_idx=component_idx,
                    loss_function_label=loss_function_label,
                )
            )

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
            idx_end = idx + (1 if rank == 0 else self.dim_geometry**rank)
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
        losses = [
            term.compute_loss(input_data, output_data) for term in self.loss_terms
        ]
        return torch.stack(losses).sum()

    def _compute_loss_batched(
        self,
        model: NeuralNetwork,
        input_data: CoordDataContainer,
        output_data: CoordDataContainer,
        dataloader: DataLoader[int] | None,
        detach: bool = False,
    ) -> torch.Tensor:
        """Compute loss over batches using a DataLoader"""
        # If detach=True, disables parameter gradients.
        if detach:
            model.requires_grad_(False)

        if dataloader is not None:
            total_loss = torch.tensor(0.0, device=self.device)
            total_points: int = 0

            # Loop through batches
            for batch_indices in dataloader:
                # Set batch on input/output containers
                input_data.set_batch(batch_indices)
                output_data.set_batch(batch_indices)

                # Apply model to coords.batch_internal (already batched)
                self._apply_model_at_coords(model, input_data.coords, output_data)

                # Compute loss on batch
                batch_loss = self._compute_loss(input_data, output_data)

                if detach:
                    total_loss += batch_loss.detach() * len(batch_indices)
                else:
                    total_loss += batch_loss * len(batch_indices)

                total_points += len(batch_indices)

            # Clear batch after loop
            input_data.clear_batch()
            output_data.clear_batch()
            final_loss = total_loss / total_points

        else:
            # Full-batch case
            self._apply_model_at_coords(model, input_data.coords, output_data)
            final_loss = self._compute_loss(input_data, output_data)
            if detach:
                final_loss = final_loss.detach()

        # Re-enable parameter grads if we turned them off
        if detach:
            model.requires_grad_(True)

        return final_loss

    def compute_unweighted_loss_train(self, model: NeuralNetwork) -> torch.Tensor:
        return self._compute_loss_batched(
            model,
            self.train_input_data,
            self.train_output_data,
            self.train_loader,
            detach=False,
        )

    def compute_weighted_loss_train(self, model: NeuralNetwork) -> torch.Tensor:
        return self.weight * self.compute_unweighted_loss_train(model)

    def compute_unweighted_loss_val(self, model: NeuralNetwork) -> torch.Tensor:
        if self.val_coords is None:
            raise ValueError("No validation coordinates provided to loss term group.")
        if self.val_input_data is None or self.val_output_data is None:
            raise ValueError(
                "Validation input and output data containers required to compute loss."
            )

        return self._compute_loss_batched(
            model,
            self.val_input_data,
            self.val_output_data,
            self.val_loader,
            detach=True,
        )

    def compute_weighted_loss_val(self, model: NeuralNetwork) -> torch.Tensor:
        return self.weight * self.compute_unweighted_loss_val(model)

    def compute_unweighted_loss_at_coords(
        self,
        model: NeuralNetwork,
        coords: CoordsProtocol,
        input_fields: dict[str, Any] | None = None,
        batch_size: int | None = None,
        shuffle: bool = True,
        num_workers: int = 0,
    ) -> torch.Tensor:
        # Set data containers
        input_data, output_data = self._init_data_containers(coords, input_fields)

        # Set up data loader
        n_points_test = coords.full_tensor.shape[0]
        test_loader = self._set_data_loader(
            n_points_test, batch_size, shuffle, num_workers
        )

        # Return unweighted loss
        return self._compute_loss_batched(
            model, input_data, output_data, test_loader, detach=True
        )

    def compute_weighted_loss_at_coords(
        self,
        model: NeuralNetwork,
        coords: CoordsProtocol,
        input_fields: dict[str, Any] | None = None,
        batch_size: int | None = None,
        shuffle: bool = True,
        num_workers: int = 0,
    ) -> torch.Tensor:
        return self.weight * self.compute_unweighted_loss_at_coords(
            model, coords, input_fields, batch_size, shuffle, num_workers
        )

    def _compute_residuals(
        self,
        input_data: CoordDataContainer,
        output_data: CoordDataContainer,
    ) -> dict[str, torch.Tensor]:
        """Compute residuals for the current batch stored in data containers."""

        residuals: dict[str, torch.Tensor] = {}

        for loss_term in self.loss_terms:
            res = loss_term.compute_residual(input_data, output_data)

            field = output_data.get_field(loss_term.field_label)
            component_label = field.component_labels[loss_term.component_idx]

            residuals[component_label] = res

        return residuals

    def compute_residuals_at_coords(
        self,
        model: NeuralNetwork,
        coords: CoordsProtocol,
        input_fields: dict[str, Any] | None = None,
        batch_size: int | None = None,
        shuffle: bool = True,
        num_workers: int = 0,
    ) -> dict[str, torch.Tensor]:
        # Disable weight gradients for analysis
        model.requires_grad_(False)

        input_data, output_data = self._init_data_containers(coords, input_fields)
        n_points = coords.full_tensor.shape[0]
        loader = self._set_data_loader(n_points, batch_size, shuffle, num_workers)

        if loader is not None:
            all_residuals: dict[str, list[torch.Tensor]] = {}

            for batch_indices in loader:
                input_data.set_batch(batch_indices)
                output_data.set_batch(batch_indices)
                self._apply_model_at_coords(model, input_data.coords, output_data)

                batch_residuals = self._compute_residuals(input_data, output_data)

                for label, res in batch_residuals.items():
                    if label not in all_residuals:
                        all_residuals[label] = []
                    all_residuals[label].append(res.detach())

            input_data.clear_batch()
            output_data.clear_batch()

            # Re-enable weight gradients
            model.requires_grad_(True)
            return {
                label: torch.cat(tensors) for label, tensors in all_residuals.items()
            }
        else:
            self._apply_model_at_coords(model, coords, output_data)
            residuals = self._compute_residuals(input_data, output_data)

            # detach tensors before returning
            detached_res = {label: val.detach() for label, val in residuals.items()}

            input_data.clear_batch()
            output_data.clear_batch()

            # Cleanup for full-batch case
            model.requires_grad_(True)
            return detached_res
