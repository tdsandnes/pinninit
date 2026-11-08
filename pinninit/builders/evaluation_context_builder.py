from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ..config.config import (
    ConstConfig,
    DatasetConfig,
    FieldsConfig,
    LossTermConfig,
)
from ..data_structure.batching import Batches
from ..data_structure.equation import Equation
from ..data_structure.field import make_field_dataclass
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
    from ..data_structure.coords import Coords
    from ..data_structure.data_container import DataContainer
    from ..data_structure.field import Field

# Loss function used for validation loss
_VAL_LOSS_FN = LossFunctionLabels.MSE


def build_evaluation_contexts(
    loss_term_config: LossTermConfig,
    dataset_config: DatasetConfig | None,
    coords_class: type[Coords],
    fields_config: FieldsConfig,
    const_configs: list[ConstConfig],
    equations: list[Equation],
    point_weight_equation: Equation | None = None,
    n_batches: int = 1,
    shuffle: bool = True,
) -> tuple[EvaluationContext, EvaluationContext]:
    """The training and validation EvaluationContexts of a loss term."""
    # A data loss loads the field it compares to (only input fields are in the
    # HDF5), so a dataset doesn't need to hold the other input fields
    input_field_label_to_rank = {
        label: rank
        for label, rank in fields_config.input_field_label_to_rank.items()
        if label == loss_term_config.field_label
    }

    if dataset_config is None:
        raise ValueError(
            f"LossTermConfig '{loss_term_config.label}' has no dataset_label. "
            "Currently physics losses must still reference a dataset for coords."
        )

    # Load train role
    train_coords = load_coords_from_hdf5(
        dataset_config=dataset_config,
        role=dataset_config.train_label,
        coords_class=coords_class,
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
        coords_class=coords_class,
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
    )

    return train_eval_context, val_eval_context


def _build_one_eval_context(
    coords: Coords,
    loaded_input_fields: dict[str, dict[str, torch.Tensor]],
    fields_config: FieldsConfig,
    const_configs: list[ConstConfig],
    equations: list[Equation],
    loss_fn_label: LossFunctionLabels,
    point_weight_equation: Equation | None = None,
    n_batches: int = 1,
    shuffle: bool = True,
) -> EvaluationContext:
    """The EvaluationContext of one set of coords."""
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
    populate_consts(output_dc, const_configs)

    batches = Batches(coords, n_batches, shuffle)

    # Optional coord-based multiplier on the point weights
    point_weight_modifier = None
    if point_weight_equation is not None:
        point_weight_modifier = _build_point_weight_modifier(
            point_weight_equation, output_dc, coords
        )

    return EvaluationContext(
        output_data=output_dc,
        loss_function=loss_fn,
        batches=batches,
        input_data=input_dc,
        point_weight_modifier=point_weight_modifier,
    )


def _build_point_weight_modifier(
    equation: Equation,
    output_dc: DataContainer,
    coords: Coords,
) -> Field:
    """The fixed point-weight multiplier that equation gives at coords."""
    if len(equation.output_labels) != 1:
        raise ValueError(
            f"Point-weight equation '{equation.label}' must have exactly one "
            f"output, got {equation.output_labels}."
        )

    operators = Operators(coords)
    result = equation.evaluate(output_dc, coords, operators)
    modifier_tensor = result[equation.output_labels[0]].detach()

    ScalarField = make_field_dataclass(0, type(coords))
    return ScalarField(
        label="point_weight_modifier", coords=coords, value=modifier_tensor
    )
