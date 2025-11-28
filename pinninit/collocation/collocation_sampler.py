from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Callable

    from ..geometry.coordinate_system import CoordSystem
    from ..io.input_parameters import PINNParameters


class CollocationSamplerLabels(str, Enum):
    """Enumeration of collocation point sampler method labels."""

    RANDOM = "random"


def random_domain_sampler(
    coord_system: CoordSystem,
    lb: torch.Tensor,
    ub: torch.Tensor,
    pinn_params: PINNParameters,
    device: torch.device,
) -> torch.Tensor:
    """
    Randomly sample coordinate domain.

    Parameters:
    ----------
    coord_system : CoordSystem
        CoordSystem object
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
    coord_dim = coord_system.coord_dim

    # Set seed
    torch.manual_seed(seed)

    # Sample uniformly in [lb, ub] for each dimension
    # torch.rand(num_samples, coord_dim) gives values in [0, 1]
    collocation_coords = lb + (ub - lb) * torch.rand(num_samples, coord_dim).to(device)

    ### Currently this is uniform in coordinate space

    return collocation_coords


# Mapping from data set sampler method labels to python functions
COLLOCATION_SAMPLER_REGISTRY: dict[
    CollocationSamplerLabels,
    Callable[
        [CoordSystem, torch.Tensor, torch.Tensor, PINNParameters, torch.device],
        torch.Tensor,
    ],
] = {
    CollocationSamplerLabels.RANDOM: random_domain_sampler,
}
