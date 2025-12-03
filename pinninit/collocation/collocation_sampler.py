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
    lb: dict[str, torch.Tensor],
    ub: dict[str, torch.Tensor],
    pinn_params: PINNParameters,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """
    Randomly sample coordinate domain.

    Parameters:
    ----------
    lb : dict[str, torch.Tensor]
        Lower bounds for each coordinate
    ub : dict[str, torch.Tensor]
        Upper bounds for each coordinate
    pinn_params : PINNParameters
        PINN parameters.

    Returns:
    ----------
    dict[str, torch.Tensor]
        Sampled data {axis_label: n_points}
    """
    # Extract params
    num_samples = pinn_params.collocation_params.num_collocation_points
    seed = pinn_params.collocation_params.collocation_sampling_seed

    # Set seed
    torch.manual_seed(seed)

    # Sample uniformly in [lb, ub] for each dimension
    # torch.rand(num_samples) gives values in [0, 1]
    collocation_coords = {}
    for coord_label in lb:
        collocation_coords[coord_label] = lb[coord_label] + (
            ub[coord_label] - lb[coord_label]
        ) * torch.rand(num_samples).to(device)

    ### Currently this is uniform in coordinate space

    return collocation_coords


# Mapping from data set sampler method labels to python functions
COLLOCATION_SAMPLER_REGISTRY: dict[
    CollocationSamplerLabels,
    Callable[
        [
            dict[str, torch.Tensor],
            dict[str, torch.Tensor],
            PINNParameters,
            torch.device,
        ],
        dict[str, torch.Tensor],
    ],
] = {
    CollocationSamplerLabels.RANDOM: random_domain_sampler,
}
