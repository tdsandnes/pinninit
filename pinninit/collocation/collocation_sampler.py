from enum import Enum

import torch
from coordinate_system import CoordSystem
from input_parameters import PINNParameters


class DataSetSamplerLabels(str, Enum):
    """Enumeration of data set sampler method labels."""

    RANDOM = "random"


def random_domain_sampler(
    coord_system: CoordSystem,
    lb: torch.Tensor,
    ub: torch.Tensor,
    pinn_params: PINNParameters,
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
    num_samples = pinn_params.collocation_params.num_train_samples
    seed = pinn_params.collocation_params.train_sampling_seed
    coord_dim = coord_system.coord_dim

    # Set seed
    torch.manual_seed(seed)

    # Sample uniformly in [lb, ub] for each dimension
    # torch.rand(num_samples, coord_dim) gives values in [0, 1]
    collocation_coords = lb + (ub - lb) * torch.rand(num_samples, coord_dim)

    ### Currently this is uniform in coordinate space

    return collocation_coords


# Mapping from data set sampler method labels to python functions
DATA_SET_SAMPLER_REGISTRY = {
    DataSetSamplerLabels.RANDOM: random_domain_sampler,
}
