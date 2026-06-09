from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ..config.config import (
    ConstConfig,
    CoordSystemConfig,
    DatasetConfig,
    FieldsConfig,
    LossTermConfig,
)
from ..data_structure.batching import make_dataloader
from ..data_structure.equation import Equation
from ..io.load_dataset_from_hdf5 import load_coords_from_hdf5, load_fields_from_hdf5
from ..loss.evaluation_context import EvaluationContext
from ..loss.loss_function import LOSS_FUNCTION_REGISTRY
from .data_container_builder import (
    build_input_data_container,
    build_output_data_container,
    populate_consts,
)

if TYPE_CHECKING:
    from ..data_structure.coords_dataclass import CoordsProtocol
    from ..data_structure.data_container import DataContainer
    from ..geometry.domain import Domain
    from ..loss.loss_function import LossFunctionLabels


def build_evaluation_contexts(
    loss_term_config: LossTermConfig,
    dataset_config: DatasetConfig,
    Coords: type[CoordsProtocol],
    coord_system_config: CoordSystemConfig,
    fields_config: FieldsConfig,
    const_configs: list[ConstConfig],
    equations: list[Equation],
    domain: Domain,
    batch_size: int | None = None,
    shuffle: bool = True,
    num_workers: int = 0,
    device: torch.device | None = None,
) -> tuple[EvaluationContext, EvaluationContext]:
    """
    Build train and validation EvaluationContexts for a single LossTermConfig.

    This function:
      1. Loads train and val coordinates and input fields from HDF5
      2. Builds input and output DataContainers for each role
      3. Wires up the shared loss function
      4. Optionally wraps the train context in a DataLoader for mini-batching

    Parameters
    ----------
    loss_term_config:
        Config for this loss term (field_label, loss_function, etc.).
    dataset_config:
        Config for the HDF5 dataset to load.
    Coords:
        The Coords class for this set of coordinates.
    coord_system_config:
        The CoordSystemConfig corresponding to Coords (needed for axis labels).
    fields_config:
        Fields config — determines which fields are input/output/derivable.
    const_configs:
        Constant configs — values are populated into the output DataContainer.
    equations:
        Already-built Equation objects for derived field computation.
    domain:
        Domain object for this coordinate set.
    batch_size:
        If provided, wraps the training context's coords in a DataLoader.
    device:
        PyTorch device (inferred from Coords if None).

    Returns
    -------
    tuple[EvaluationContext, EvaluationContext]
        (train_eval_context, val_eval_context)
    """
    coord_labels = Coords.coord_labels

    # Determine which fields to load from file (only input fields are in the HDF5)
    input_field_label_to_rank = fields_config.input_field_label_to_rank

    # Load train role
    train_coords = load_coords_from_hdf5(
        dataset_config=dataset_config,
        role=dataset_config.train_label,
        coord_labels=coord_labels,
        Coords=Coords,
    )
    train_input_fields = load_fields_from_hdf5(
        dataset_config=dataset_config,
        role=dataset_config.train_label,
        coords=train_coords,
        field_label_to_rank=input_field_label_to_rank,
    )

    # Load val role
    val_coords = load_coords_from_hdf5(
        dataset_config=dataset_config,
        role=dataset_config.val_label,
        coord_labels=coord_labels,
        Coords=Coords,
    )
    val_input_fields = load_fields_from_hdf5(
        dataset_config=dataset_config,
        role=dataset_config.val_label,
        coords=val_coords,
        field_label_to_rank=input_field_label_to_rank,
    )

    # Build train contexts
    train_eval_context = _build_one_eval_context(
        coords=train_coords,
        loaded_input_fields=train_input_fields,
        fields_config=fields_config,
        const_configs=const_configs,
        equations=equations,
        loss_fn_label=loss_term_config.loss_function,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )

    # Build val contexts
    val_eval_context = _build_one_eval_context(
        coords=val_coords,
        loaded_input_fields=val_input_fields,
        fields_config=fields_config,
        const_configs=const_configs,
        equations=equations,
        loss_fn_label=LossFunctionLabels.MSE,  # Validation loss fn is always MSE
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )

    return train_eval_context, val_eval_context


def _build_one_eval_context(
    coords: CoordsProtocol,
    loaded_input_fields: dict[str, dict[str, torch.Tensor]],
    fields_config: FieldsConfig,
    const_configs: list[ConstConfig],
    equations: list[Equation],
    loss_fn_label: LossFunctionLabels,
    batch_size: int | None,
    shuffle: bool = True,
    num_workers: int = 0,
) -> EvaluationContext:
    """Build a single EvaluationContext for one data role."""
    # Get loss function
    loss_fn = LOSS_FUNCTION_REGISTRY[loss_fn_label]

    # Input DataContainer holds the ground-truth / boundary data loaded from file
    # Only built when there are input fields to compare against
    input_dc: DataContainer | None = None
    if fields_config.input_field_label_to_rank:
        input_dc = build_input_data_container(
            coords=coords,
            fields_config=fields_config,
            const_configs=const_configs,
        )
        # Populate input container with tensors loaded from file
        for field_label, component_dict in loaded_input_fields.items():
            input_dc.set_field_from_component_dict(
                label=field_label,
                component_dict=component_dict,
                coords=coords,
            )

    # Output DataContainer holds the model predictions
    output_dc = build_output_data_container(
        coords=coords,
        fields_config=fields_config,
        const_configs=const_configs,
        equations=equations,
    )
    populate_consts(output_dc, const_configs, coords)

    # If batch_size = None, use full batch
    batch_size = batch_size or coords.n_points

    # Make dataloader
    dataloader = make_dataloader(
        coords=coords,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )

    return EvaluationContext(
        output_data=output_dc,
        loss_function=loss_fn,
        dataloader=dataloader,
        input_data=input_dc,
    )
