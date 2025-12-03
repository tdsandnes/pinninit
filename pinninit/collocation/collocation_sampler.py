from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Callable

    from ..io.input_parameters import PINNParameters


class CollocationSamplerLabels(str, Enum):
    """Enumeration of collocation point sampler method labels."""

    RANDOM = "random"


def random_domain_sampler(
    lb: list[torch.Tensor],
    ub: list[torch.Tensor],
    pinn_params: PINNParameters,
    device: torch.device,
) -> list[torch.Tensor]:
    """
    Randomly sample coordinate domain.

    Parameters:
    ----------
    lb : list[torch.Tensor]
        Lower bounds for each coordinate
    ub :  list[torch.Tensor]
        Upper bounds for each coordinate
    pinn_params : PINNParameters
        PINN parameters.

    Returns:
    ----------
    torch.Tensor
        Sampled data. Shape (n_points, coord_dim)
    """
    # Extract params
    num_samples = pinn_params.collocation_params.num_collocation_points
    seed = pinn_params.collocation_params.collocation_sampling_seed
    dim = len(pinn_params.coordinates_params.coordinates)

    # Set seed
    torch.manual_seed(seed)

    # Sample uniformly in [lb, ub] for each dimension
    # torch.rand(num_samples, coord_dim) gives values in [0, 1]
    collocation_coords = []
    for i in range(dim):
        collocation_coords.append(
            lb[i] + (ub[i] - lb[i]) * torch.rand(num_samples).to(device)
        )

    ### Currently this is uniform in coordinate space

    return collocation_coords


# Mapping from data set sampler method labels to python functions
COLLOCATION_SAMPLER_REGISTRY: dict[
    CollocationSamplerLabels,
    Callable[
        [list[torch.Tensor], list[torch.Tensor], PINNParameters, torch.device],
        list[torch.Tensor],
    ],
] = {
    CollocationSamplerLabels.RANDOM: random_domain_sampler,
}
