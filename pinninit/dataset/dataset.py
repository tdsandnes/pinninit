from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import scipy
import torch

from ..utils import execute_from_registry
from .dataset_sampler import DATA_SET_SAMPLER_REGISTRY

if TYPE_CHECKING:
    from ..io.input_parameters import PINNParameters


def load_dataset(
    pinn_params: PINNParameters,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

    # Flatten and stack
    coords_and_fields_array = np.hstack(
        [
            TT.flatten()[:, None],  # t
            XX.flatten()[:, None],  # x
            YY.flatten()[:, None],  # y
            UU.flatten()[:, None],  # u
            VV.flatten()[:, None],  # v
        ]
    )  # shape (N*T, 3+2)

    # Sample arrays
    sampled_array = execute_from_registry(
        train_sampling_strategy,
        DATA_SET_SAMPLER_REGISTRY,
        coords_and_fields_array,
        pinn_params,
    )

    ### Hardcoded 3 here must come from spatial coords + 1
    coords_sampled_array = sampled_array[:, :3]
    fields_sampled_array = sampled_array[:, 3:]

    input_coords = torch.tensor(coords_sampled_array, requires_grad=True).float()
    input_fields = torch.tensor(fields_sampled_array, requires_grad=True).float()
    ### Adding this here as a placeholder for overhall of this func
    input_consts = torch.tensor([])

    return input_coords, input_fields, input_consts
