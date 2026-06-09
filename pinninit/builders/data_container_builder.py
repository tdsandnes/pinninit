from __future__ import annotations

import torch

from ..config.config import ConstConfig, FieldsConfig
from ..data_structure.coords_dataclass import CoordsProtocol
from ..data_structure.data_container import DataContainer
from ..data_structure.equation import Equation


def build_input_data_container(
    coords: CoordsProtocol,
    fields_config: FieldsConfig,
    const_configs: list[ConstConfig],
) -> DataContainer:
    return DataContainer(
        coords=coords,
        primary_field_label_to_rank=fields_config.input_field_label_to_rank,
        derived_field_label_to_rank={},
        const_labels={cfg.label for cfg in const_configs} if const_configs else None,
        equations=None,
    )


def build_output_data_container(
    coords: CoordsProtocol,
    fields_config: FieldsConfig,
    const_configs: list[ConstConfig],
    equations: list[Equation],
) -> DataContainer:
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
    coords: CoordsProtocol,
) -> None:
    """
    Set constant values on a DataContainer from ConstConfig.

    Parameters
    ----------
    data_container:
        Container whose consts will be populated.
    const_configs:
        Configs with label and value for each constant.
    coords:
        The active coords to register for each constant.
    """

    for cfg in const_configs:
        if not cfg.learnable:
            value = torch.tensor(cfg.value, device=coords.device)
            data_container.set_const(cfg.label, value, coords=coords)
        # Learnable constants are handled separately by the NN/trainer builder
        # since they need to be registered as nn.Parameters.
