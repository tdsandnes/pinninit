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
    dim_nongeometry: int,
    dim_geometry: int,
    lb: torch.Tensor,
    ub: torch.Tensor,
    pinn_params: PINNParameters,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Randomly sample coordinate domain.

    Parameters:
    ----------
    coord_dim: int
        Number of coordinates.
    lb : torch.Tensor
        Lower bounds for each coordinate
    ub : torch.Tensor
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

    # Set seed
    torch.manual_seed(seed)

    # Sample uniformly in [lb, ub] for each dimension
    # torch.rand(num_samples, coord_dim) gives values in [0, 1]
    collocation_coords_nongeometric = lb[:dim_nongeometry] + (
        ub[:dim_nongeometry] - lb[:dim_nongeometry]
    ) * torch.rand(num_samples, dim_nongeometry).to(device)
    collocation_coords_geometric = lb[
        dim_nongeometry : dim_nongeometry + dim_geometry
    ] + (
        ub[dim_nongeometry : dim_nongeometry + dim_geometry]
        - lb[dim_nongeometry : dim_nongeometry + dim_geometry]
    ) * torch.rand(num_samples, dim_geometry).to(device)

    ### Currently this is uniform in coordinate space

    return collocation_coords_nongeometric, collocation_coords_geometric


# Mapping from data set sampler method labels to python functions
COLLOCATION_SAMPLER_REGISTRY: dict[
    CollocationSamplerLabels,
    Callable[
        [int, int, torch.Tensor, torch.Tensor, PINNParameters, torch.device],
        tuple[torch.Tensor, torch.Tensor],
    ],
] = {
    CollocationSamplerLabels.RANDOM: random_domain_sampler,
}
