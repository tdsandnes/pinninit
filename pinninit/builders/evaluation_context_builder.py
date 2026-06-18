from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ..config.config import (
    ConstConfig,
    DatasetConfig,
    FieldsConfig,
    LossTermConfig,
)
from ..data_structure.batching import make_dataloader
from ..data_structure.coords_dataclass import get_coords_class_default
from ..data_structure.equation import Equation
from ..data_structure.field_dataclass import make_field_dataclass
from ..differential_equations.operators import Operators
from ..io.load_dataset_from_hdf5 import load_coords_from_hdf5, load_fields_from_hdf5
from ..loss.evaluation_context import EvaluationContext
from ..loss.loss_function import LOSS_FUNCTION_REGISTRY, LossFunctionLabels
from ..loss.loss_term import LossTermLabels
from .data_container_builder import (
    build_input_data_container,
    build_output_data_container,
    populate_consts,
)

if TYPE_CHECKING:
    from ..data_structure.coords_dataclass import CoordsProtocol
    from ..data_structure.data_container import DataContainer
    from ..data_structure.field_dataclass import FieldProtocol

# Loss function used for validation loss
_VAL_LOSS_FN = LossFunctionLabels.MSE


def build_evaluation_contexts(
    loss_term_config: LossTermConfig,
    dataset_config: DatasetConfig | None,
    Coords: type[CoordsProtocol],
    fields_config: FieldsConfig,
    const_configs: list[ConstConfig],
    equations: list[Equation],
    point_weight_equation: Equation | None = None,
    n_batches: int = 1,
    shuffle: bool = True,
    num_workers: int = 0,
) -> tuple[EvaluationContext, EvaluationContext]:
    """
    Build train and validation EvaluationContexts for a single LossTermConfig.

    This function:
      1. Loads train and val coordinates and input fields from HDF5
      2. Builds input and output DataContainers for each role
      3. Wires up the shared loss function
      4. Wraps the train context in a DataLoader for mini-batching

    Parameters
    ----------
    loss_term_config:
        Config for this loss term (field_label, loss_function, etc.).
    dataset_config:
        Config for the HDF5 dataset to load.
    Coords:
        The Coords class for this set of coordinates.
    fields_config:
        Fields config — determines which fields are input/output/derivable.
    const_configs:
        Constant configs — values are populated into the output DataContainer.
    equations:
        Already-built Equation objects for derived field computation.
    point_weight_equation:
        Optional Equation evaluated on the training coords to produce a
        per-point multiplier on the point weights. Applied to the training
        context only.
    n_batches:
        Number of batches, used to sync all loss terms.
    shuffle:
        Whether to shuffle the DataLoader.
    num_workers:
        Number of DataLoader worker processes.

    Returns
    -------
    tuple[EvaluationContext, EvaluationContext]
        (train_eval_context, val_eval_context)
    """
    coord_labels = get_coords_class_default(Coords, "coord_labels")

    # Determine which fields to load from file (only input fields are in the HDF5)
    input_field_label_to_rank = fields_config.input_field_label_to_rank

    if dataset_config is None:
        raise ValueError(
            f"LossTermConfig '{loss_term_config.label}' has no dataset_label. "
            "Currently physics losses must still reference a dataset for coords."
        )

    # Load train role
    train_coords = load_coords_from_hdf5(
        dataset_config=dataset_config,
        role=dataset_config.train_label,
        coord_labels=coord_labels,
        Coords=Coords,
    )
    # Physics losses use collocation points only; no input fields to load
    is_physics = loss_term_config.loss_type == LossTermLabels.PHYSICS
    train_input_fields = (
        {}
        if is_physics
        else load_fields_from_hdf5(
            dataset_config=dataset_config,
            role=dataset_config.train_label,
            coords=train_coords,
            field_label_to_rank=input_field_label_to_rank,
        )
    )

    # Load val role
    val_coords = load_coords_from_hdf5(
        dataset_config=dataset_config,
        role=dataset_config.val_label,
        coord_labels=coord_labels,
        Coords=Coords,
    )
    val_input_fields = (
        {}
        if is_physics
        else load_fields_from_hdf5(
            dataset_config=dataset_config,
            role=dataset_config.val_label,
            coords=val_coords,
            field_label_to_rank=input_field_label_to_rank,
        )
    )

    # Build train contexts
    train_eval_context = _build_one_eval_context(
        coords=train_coords,
        loaded_input_fields=train_input_fields,
        fields_config=fields_config,
        const_configs=const_configs,
        equations=equations,
        loss_fn_label=loss_term_config.loss_function,
        point_weight_equation=point_weight_equation,
        n_batches=n_batches,
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
        loss_fn_label=_VAL_LOSS_FN,  # Validation loss fn is always MSE
        point_weight_equation=None,  # Validation loss fn is not weighted
        n_batches=n_batches,
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
    point_weight_equation: Equation | None = None,
    n_batches: int = 1,
    shuffle: bool = True,
    num_workers: int = 0,
) -> EvaluationContext:
    """Build a single EvaluationContext for one data role."""
    # Get loss function
    loss_fn = LOSS_FUNCTION_REGISTRY[loss_fn_label]

    # Input DataContainer holds the ground-truth / boundary data loaded from file
    # Only built when there are input fields to compare against
    input_dc: DataContainer | None = None
    if loaded_input_fields:
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

    # Make dataloader
    dataloader = make_dataloader(
        coords=coords,
        n_batches=n_batches,
        shuffle=shuffle,
        num_workers=num_workers,
    )

    # Every loss term carries a per-point weight field
    Field = make_field_dataclass(rank=0, coords=coords, device=coords.device)
    point_weights = Field(
        label="point_weights",
        value=torch.ones(coords.n_points, device=coords.device),
    )

    # Optional coord-based multiplier on the point weights
    point_weight_modifier = None
    if point_weight_equation is not None:
        point_weight_modifier = _build_point_weight_modifier(
            point_weight_equation, output_dc, coords
        )

    return EvaluationContext(
        output_data=output_dc,
        loss_function=loss_fn,
        dataloader=dataloader,
        input_data=input_dc,
        raw_point_weights=point_weights,
        point_weight_modifier=point_weight_modifier,
    )


def _build_point_weight_modifier(
    equation: Equation,
    output_dc: DataContainer,
    coords: CoordsProtocol,
) -> FieldProtocol:
    if len(equation.output_labels) != 1:
        raise ValueError(
            f"Point-weight equation '{equation.label}' must have exactly one "
            f"output, got {equation.output_labels}."
        )

    operators = Operators(coords)
    result = equation.evaluate(output_dc, coords, operators)
    modifier_tensor = result[equation.output_labels[0]].detach()

    Field = make_field_dataclass(rank=0, coords=coords, device=coords.device)
    return Field(label="point_weight_modifier", value=modifier_tensor)
