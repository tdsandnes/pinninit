from enum import Enum

import numpy as np
from input_parameters import DataParams


class DataSetSamplerLabels(str, Enum):
    """Enumeration of data set sampler method labels."""

    RANDOM = "random"


def random_sampler(array: np.ndarray, data_params: DataParams) -> np.ndarray:
    """
    Randomly sample NumPy array.

    Parameters:
    ----------
    array : ndarray
        array to sample. Shape (n_points, :)
    num_samples : int
        The number of samples to take.
    seed : int
        Random seed.

    Returns:
    ----------
    ndarray
        Sampled data.
    """
    # Extract DataParams
    num_samples = data_params.num_train_samples
    seed = data_params.train_sampling_seed

    # Use seed to select num_train_samples points
    rng = np.random.default_rng(seed)
    mask = rng.choice(array.shape[0], num_samples, replace=False)
    return array[mask, :]


# Mapping from data set sampler method labels to python functions
DATA_SET_SAMPLER_REGISTRY = {
    DataSetSamplerLabels.RANDOM: random_sampler,
}
