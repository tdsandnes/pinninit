from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import scipy
import torch

from .dataset_sampler import DATA_SET_SAMPLER_REGISTRY

if TYPE_CHECKING:
    from ..io.input_parameters import PINNParameters


def load_dataset(
    pinn_params: PINNParameters, device: torch.device
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    # Extract DataParams
    file_path = pinn_params.data_params.file_path
    train_sampling_strategy = pinn_params.data_params.train_sampling_strategy

    ### THIS IS ALL TEMP. SHOULD BE LOADING STANDARDISED HDF5 FILE
    ### Should compare HDF5 headings with param labels in param file

    # Load and reformat the data.
    data = scipy.io.loadmat(file_path)

    U_star = data["U_star"]  # velocity coordinates (N x 2 x T)
    t_star = data["t"]  # temporal coordinates  (T x 1)
    X_star = data["X_star"]  # Spatial coordinates (N x 2)

    N = X_star.shape[0]
    T = t_star.shape[0]

    # Rearrange Data
    XX = np.tile(X_star[:, 0:1], (1, T))  # N x T
    YY = np.tile(X_star[:, 1:2], (1, T))  # N x T
    TT = np.tile(t_star, (1, N)).T  # N x T

    UU = U_star[:, 0, :]  # N x T
    VV = U_star[:, 1, :]  # N x T

    # Flatten data
    t = TT.flatten()
    x = XX.flatten()
    y = YY.flatten()
    u = UU.flatten()
    v = VV.flatten()

    # Get mask for sampling
    data_length = len(t)
    mask = DATA_SET_SAMPLER_REGISTRY[train_sampling_strategy](
        data_length,
        pinn_params,
    )

    # Sample data
    t_sampled = t[mask]
    x_sampled = x[mask]
    y_sampled = y[mask]
    u_sampled = u[mask]
    v_sampled = v[mask]

    ### TEMPORARY to give an idea of what it will look like from hdf5
    input_coord_dict = {"t": t_sampled, "x": x_sampled, "y": y_sampled}
    input_field_dict = {"u": u_sampled, "v": v_sampled}

    # Set up coords in correct order
    input_coords = []
    index_to_coord_dict = pinn_params.index_to_coord_dict
    num_input_coords = len(index_to_coord_dict)
    for i in range(num_input_coords):
        if i in index_to_coord_dict:
            axis_label = index_to_coord_dict[i]
            coord_values = torch.tensor(input_coord_dict[axis_label])
            coord_values = coord_values.float().to(device)
            coord_values.requires_grad_(True)
            input_coords.append(coord_values)
        else:
            raise ValueError(
                "index_to_coord_dict must contain all integer keys from 0 to "
                f"{num_input_coords} without gaps. Missing index: {i}"
            )

    # Set up fields in correct order
    input_fields = []
    index_to_input_field_dict = pinn_params.index_to_input_field_dict
    num_input_fields = len(index_to_input_field_dict)
    for i in range(num_input_fields):
        if i in index_to_input_field_dict:
            axis_label = index_to_input_field_dict[i]
            field_values = torch.tensor(input_field_dict[axis_label])
            field_values = field_values.float().to(device)
            field_values.requires_grad_(True)
            input_fields.append(field_values)
        else:
            raise ValueError(
                "index_to_input_field_dict must contain all integer keys from 0 to "
                f"{num_input_fields} without gaps. Missing index: {i}"
            )

    ### Adding this here as a placeholder for future
    input_consts = [torch.tensor([]).to(device)]

    return input_coords, input_fields, input_consts
