from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

    from ..io.input_parameters import PINNParameters


class DataSetSamplerLabels(str, Enum):
    """Enumeration of data set sampler method labels."""

    NONE = "none"
    RANDOM = "random"


def mask_no_sampler(length: int, pinn_params: PINNParameters) -> np.ndarray:
    """Returns mask with all True"""
    return np.ones(length, dtype=bool)


def mask_random_array_sampler(length: int, pinn_params: PINNParameters) -> np.ndarray:
    """
    Mask for randomly sampling 1D array of given length.

    Parameters:
    ----------
    length:
        Mask length
    pinn_params : PINNParameters
        PINN parameters.

    Returns:
    ----------
    ndarray
        Mask.
    """
    # Extract DataParams
    num_samples = pinn_params.data_params.num_train_samples
    seed = pinn_params.data_params.train_sampling_seed

    # Use seed to select num_train_samples points
    rng = np.random.default_rng(seed)
    mask = rng.choice(length, num_samples, replace=False)
    return mask


# Mapping from data set sampler method labels to python functions
DATA_SET_SAMPLER_REGISTRY: dict[
    DataSetSamplerLabels, Callable[[int, PINNParameters], np.ndarray]
] = {
    DataSetSamplerLabels.NONE: mask_no_sampler,
    DataSetSamplerLabels.RANDOM: mask_random_array_sampler,
}
