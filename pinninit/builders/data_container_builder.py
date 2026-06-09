from __future__ import annotations

import torch

from ..config.config import ConstConfig, FieldsConfig
from ..data_structure.coords_dataclass import CoordsProtocol
from ..data_structure.data_container import DataContainer
from ..data_structure.equation import Equation


def build_data_container(
    coords: CoordsProtocol,
    fields_config: FieldsConfig,
    const_configs: list[ConstConfig],
    equations: list[Equation],
    *,
    use_input_fields: bool,
) -> DataContainer:
    """
    Build a DataContainer bound to the given coords.

    Parameters
    ----------
    coords:
        The CoordsProtocol instance these fields live on.
    fields_config:
        Describes which fields exist and their ranks.
    const_configs:
        Describes scalar constants.
    equations:
        Already-built Equation objects for derived fields.
    use_input_fields:
        If True, primary fields are the *input* fields (for input DataContainers
        that hold ground-truth / boundary data loaded from file).
        If False, primary fields are the *output* fields (for the model's
        prediction DataContainer).

    Returns
    -------
    DataContainer
    """
    if use_input_fields:
        primary_field_label_to_rank = fields_config.input_field_label_to_rank
    else:
        primary_field_label_to_rank = fields_config.output_field_label_to_rank

    const_labels = {cfg.label for cfg in const_configs} if const_configs else None

    return DataContainer(
        coords=coords,
        primary_field_label_to_rank=primary_field_label_to_rank,
        derived_field_label_to_rank=fields_config.derivable_field_label_to_rank,
        const_labels=const_labels,
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


# def populate_fields_from_hdf5 ?
