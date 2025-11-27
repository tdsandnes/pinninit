from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..io.input_parameters import PINNParameters


class DataSetSamplerLabels(str, Enum):
    """Enumeration of data set sampler method labels."""

    RANDOM = "random"


def random_array_sampler(array: np.ndarray, pinn_params: PINNParameters) -> np.ndarray:
    """
    Randomly sample NumPy array.

    Parameters:
    ----------
    array : ndarray
        array to sample. Shape (n_points, :)
    pinn_params : PINNParameters
        PINN parameters.

    Returns:
    ----------
    ndarray
        Sampled data.
    """
    # Extract DataParams
    num_samples = pinn_params.data_params.num_train_samples
    seed = pinn_params.data_params.train_sampling_seed

    # Use seed to select num_train_samples points
    rng = np.random.default_rng(seed)
    mask = rng.choice(array.shape[0], num_samples, replace=False)
    return array[mask, :]


# Mapping from data set sampler method labels to python functions
DATA_SET_SAMPLER_REGISTRY = {
    DataSetSamplerLabels.RANDOM: random_array_sampler,
}
