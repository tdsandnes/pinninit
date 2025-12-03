from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .collocation_sampler import COLLOCATION_SAMPLER_REGISTRY

if TYPE_CHECKING:
    from ..io.input_parameters import PINNParameters


def load_collocation(
    lb: list[torch.Tensor],
    ub: list[torch.Tensor],
    pinn_params: PINNParameters,
    device: torch.device,
) -> list[torch.Tensor]:
    # Extract DataParams
    ###from_file = pinn_params.collocation_params.from_file

    ### if from_file:
    ###     file_path = pinn_params.collocation_params.file_path

    ###  NOT IMPLEMENTED
    ###     return ....###
    ###else:
    collocation_strategy = pinn_params.collocation_params.collocation_strategy

    sampled_coords = COLLOCATION_SAMPLER_REGISTRY[collocation_strategy](
        lb,
        ub,
        pinn_params,
        device,
    )

    # Redefine to clean history of tensor
    collocation_coords = []
    for coord in sampled_coords:
        value = coord.detach().float().to(device)
        value.requires_grad_(True)
        collocation_coords.append(value)

    return collocation_coords
