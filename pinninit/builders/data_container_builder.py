from __future__ import annotations

import torch

from ..config.config import ConstConfig, FieldsConfig
from ..data_structure.coords import Coords
from ..data_structure.data_container import DataContainer
from ..data_structure.equation import Equation


def build_input_data_container(
    coords: Coords,
    fields_config: FieldsConfig,
    const_configs: list[ConstConfig],
) -> DataContainer:
    """The DataContainer of the data at coords."""
    return DataContainer(
        coords=coords,
        primary_field_label_to_rank=fields_config.input_field_label_to_rank,
        derived_field_label_to_rank={},
        const_labels={cfg.label for cfg in const_configs} if const_configs else None,
        equations=None,
    )


def build_output_data_container(
    coords: Coords,
    fields_config: FieldsConfig,
    const_configs: list[ConstConfig],
    equations: list[Equation],
) -> DataContainer:
    """The DataContainer of the model's predictions, and the fields
    derived from them, at coords."""
    return DataContainer(
        coords=coords,
        primary_field_label_to_rank=fields_config.output_field_label_to_rank,
        derived_field_label_to_rank=fields_config.derivable_field_label_to_rank,
        const_labels={cfg.label for cfg in const_configs} if const_configs else None,
        equations=equations if equations else None,
    )


def populate_consts(
    data_container: DataContainer,
    const_configs: list[ConstConfig],
) -> None:
    """Set the values of the constants in data_container that aren't
    learnable."""

    for cfg in const_configs:
        if not cfg.learnable:
            value = torch.tensor(cfg.value, device=data_container.device)
            data_container.set_const(cfg.label, value)
        # Learnable constants are set by the PINN builder, which registers them
        # as nn.Parameters on the model
