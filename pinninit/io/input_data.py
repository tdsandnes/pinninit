from __future__ import annotations

from typing import TYPE_CHECKING

import h5py
import numpy as np
import torch

from .data_sampler import DataSetSamplerLabels

if TYPE_CHECKING:
    pass


def load_dataset(
    file_path: str,
    train_sampling_strategy: DataSetSamplerLabels,
    requested_dataset_type: str,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    # Dicts of all data contained in data file
    all_input_coords_dict = {}
    all_input_fields_dict = {}
    with h5py.File(file_path, "r") as f:
        # Check that required groups exist
        if "Metadata" not in f:
            raise KeyError(f"File {file_path} must contain 'Metadata' group")

        if "Coords" not in f:
            raise KeyError(f"File {file_path} must contain 'Coords' group")

        if "Fields" not in f:
            raise KeyError(f"File {file_path} must contain 'Fields' group")

        # Read Metadata
        num_points = f["Metadata/num_points"][()]
        dataset_type = f["Metadata/dataset_type"][()]
        # Convert to str
        dataset_type = (
            dataset_type.decode("utf-8")
            if isinstance(dataset_type, bytes)
            else str(dataset_type)
        )

        if dataset_type != requested_dataset_type:
            raise ValueError(
                f"Data set type much be {requested_dataset_type}, not {dataset_type}"
            )

        # Read Coords
        coords_group = f["Coords"]
        for coord in coords_group:
            all_input_coords_dict[coord] = coords_group[coord][()]

        # Check that all coordinate arrays have length matching num_points
        for name, array in all_input_coords_dict.items():
            if len(array) != num_points:
                raise ValueError(
                    f"Coordinate '{name}' length {len(array)} does not match "
                    f"num_points {num_points}"
                )

        # Read Fields
        fields_group = f["Fields"]
        for field in fields_group:
            item = fields_group[field]
            # If scalar field
            if isinstance(item, h5py.Dataset):
                all_input_fields_dict[field] = item[()]

            # If vector field
            elif isinstance(item, h5py.Group):
                field_dict = {}
                for component in item:
                    if component in all_input_coords_dict:
                        field_dict[component] = item[component][()]

                all_input_fields_dict[field] = field_dict
            else:
                raise TypeError(f"Unknown HDF5 object type for fields/{field}")

        # Check that all field arrays have length matching num_points
    #   for name, array in all_input_fields_dict.items():
    #       if len(array) != num_points:
    #           raise ValueError(
    #               f"Field '{name}' length {len(array)} does not match num_points "
    #               f"{num_points}"
    #           )

    # Get mask for sampling
    #### Temp comment this
    ###sample_mask = DATA_SET_SAMPLER_REGISTRY[train_sampling_strategy](
    ###    num_points,
    ###    pinn_params,
    ###)
    sample_mask = np.ones(num_points, dtype=bool)
    ####^temp

    # Sample data
    sampled_input_coords_dict = {}
    sampled_input_fields_dict = {}
    for coord, array in all_input_coords_dict.items():
        sampled_input_coords_dict[coord] = array[sample_mask]

    for field, item in all_input_fields_dict.items():
        if isinstance(item, np.ndarray):
            sampled_input_fields_dict[field] = item[sample_mask]
        elif isinstance(item, dict):
            sampled_input_fields_dict[field] = {}
            for component, array in item.items():
                sampled_input_fields_dict[field][component] = array[sample_mask]
    """
    ### Add these checks later
    # coords included in input file
    coord_params = pinn_params.coordinates_params
    param_coord_labels = [coord["label"] for coord in coord_params.coordinates]

    # Set up dict of coords specified in yaml file
    input_coords = {}
    for coord_label, input_array in sampled_input_coords_dict.items():
        if coord_label in param_coord_labels:
            coord_values = torch.tensor(input_array)
            coord_values = coord_values.float().to(device)
            coord_values.requires_grad_(True)
            input_coords[coord_label] = coord_values
        else:
            raise ValueError(
                "Data file must contain data for all coordinates specified in the "
                f"parameter file. Missing coordinate: {coord_label}"
            )

    # input fields included in input file
    fields_params = pinn_params.fields_params
    param_field_labels = [field["label"] for field in fields_params.input_fields]

    # Set up fields in dict
    input_fields = {}
    for field_label, input_array in sampled_input_fields_dict.items():
        if field_label in param_field_labels:
            field_values = torch.tensor(input_array)
            field_values = field_values.float().to(device)
            field_values.requires_grad_(True)
            input_fields[field_label] = field_values
        else:
            raise ValueError(
                "Data file must contain data for all input fields specified in the "
                f"parameter file. Missing input field: {field_label}"
            )
    """

    return (
        sampled_input_coords_dict,
        sampled_input_fields_dict,
    )
