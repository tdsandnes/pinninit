from __future__ import annotations

from typing import TYPE_CHECKING

import h5py
import torch

from .collocation_sampler import COLLOCATION_SAMPLER_REGISTRY

if TYPE_CHECKING:
    from ..io.input_parameters import PINNParameters


### This function should be done in a much nicer way
def load_collocation(
    lb: dict[str, torch.Tensor],
    ub: dict[str, torch.Tensor],
    pinn_params: PINNParameters,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    # Extract DataParams
    from_file = pinn_params.collocation_params.from_file

    if from_file:
        file_path = pinn_params.collocation_params.file_path

        ### This and similar bit in dataset.py should use same func

        # Dicts of all data contained in data file
        all_col_coords_dict = {}
        with h5py.File(file_path, "r") as f:
            # Check that required groups exist
            if "Metadata" not in f:
                raise KeyError(f"File {file_path} must contain 'Metadata' group")

            if "Coords" not in f:
                raise KeyError(f"File {file_path} must contain 'Coords' group")

            # Read Metadata
            num_points = f["Metadata/num_points"][()]
            dataset_type = f["Metadata/dataset_type"][()]
            # Convert to str
            dataset_type = (
                dataset_type.decode("utf-8")
                if isinstance(dataset_type, bytes)
                else str(dataset_type)
            )

            if dataset_type != "collocation":
                raise ValueError(
                    f"Data set type much be 'collocation', not {dataset_type}"
                )

            # Read Coords
            coords_group = f["Coords"]
            for coord in coords_group:
                all_col_coords_dict[coord] = coords_group[coord][()]

            # Check that all coordinate arrays have length matching num_points
            for name, array in all_col_coords_dict.items():
                if len(array) != num_points:
                    raise ValueError(
                        f"Coordinate '{name}' length {len(array)} does not match "
                        f"num_points {num_points}"
                    )

        # coords included in input file
        coord_params = pinn_params.coordinates_params
        param_coord_labels = [coord["label"] for coord in coord_params.coordinates]

        # Set up dict of coords specified in yaml file
        col_coords = {}
        for coord_label, input_array in all_col_coords_dict.items():
            if coord_label in param_coord_labels:
                coord_values = torch.tensor(input_array)
                coord_values = coord_values.float().to(device)
                coord_values.requires_grad_(True)
                col_coords[coord_label] = coord_values
            else:
                raise ValueError(
                    "Colloacation file must contain positions for all coordinates "
                    "specified in the parameter file. Missing coordinate:"
                    f" {coord_label}"
                )

        return col_coords
    else:
        collocation_strategy = pinn_params.collocation_params.collocation_strategy

        sampled_coords = COLLOCATION_SAMPLER_REGISTRY[collocation_strategy](
            lb,
            ub,
            pinn_params,
            device,
        )

        # Redefine to clean history of tensor
        collocation_coords = {}
        for coord_label, sampled_tensor in sampled_coords.items():
            clean_tensor = sampled_tensor.detach().float().to(device)
            clean_tensor.requires_grad_(True)
            collocation_coords[coord_label] = clean_tensor

    return collocation_coords
