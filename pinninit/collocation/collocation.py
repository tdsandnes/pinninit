from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ..utils import execute_from_registry
from .collocation_sampler import COLLOCATION_SAMPLER_REGISTRY

if TYPE_CHECKING:
    from ..geometry.coordinate_system import CoordSystem
    from ..io.input_parameters import PINNParameters


def load_collocation(
    coord_system: CoordSystem,
    lb: torch.Tensor,
    ub: torch.Tensor,
    pinn_params: PINNParameters,
) -> torch.Tensor:
    # Extract DataParams
    ###from_file = pinn_params.collocation_params.from_file

    ### if from_file:
    ###     file_path = pinn_params.collocation_params.file_path

    ###  NOT IMPLEMENTED
    ###     return ....###
    ###else:
    collocation_strategy = pinn_params.collocation_params.collocation_strategy

    collocation_coords = execute_from_registry(
        collocation_strategy,
        COLLOCATION_SAMPLER_REGISTRY,
        coord_system,
        lb,
        ub,
        pinn_params,
    )

    collocation_coords_tensor: torch.Tensor = torch.tensor(
        collocation_coords, requires_grad=True
    ).float()

    return collocation_coords_tensor
