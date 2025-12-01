from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ..utils import execute_from_registry
from .collocation_sampler import COLLOCATION_SAMPLER_REGISTRY

if TYPE_CHECKING:
    from ..io.input_parameters import PINNParameters


def load_collocation(
    dim_nongeometry: int,
    dim_geometry: int,
    lb: torch.Tensor,
    ub: torch.Tensor,
    pinn_params: PINNParameters,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Extract DataParams
    ###from_file = pinn_params.collocation_params.from_file

    ### if from_file:
    ###     file_path = pinn_params.collocation_params.file_path

    ###  NOT IMPLEMENTED
    ###     return ....###
    ###else:
    collocation_strategy = pinn_params.collocation_params.collocation_strategy

    collocation_coords_nongeometric, collocation_coords_geometric = (
        execute_from_registry(
            collocation_strategy,
            COLLOCATION_SAMPLER_REGISTRY,
            dim_nongeometry,
            dim_geometry,
            lb,
            ub,
            pinn_params,
            device,
        )
    )

    collocation_coords_nongeometric.requires_grad_(True)
    collocation_coords_geometric.requires_grad_(True)

    return collocation_coords_nongeometric, collocation_coords_geometric
