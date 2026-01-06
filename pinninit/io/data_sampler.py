from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable


class DataSetSamplerLabels(str, Enum):
    """Enumeration of data set sampler method labels."""

    NONE = "none"
    RANDOM = "random"


def mask_no_sampler(length: int, num_samples: int, seed: int) -> np.ndarray:
    """Returns mask with all True"""
    return np.ones(length, dtype=bool)


def mask_random_array_sampler(length: int, num_samples: int, seed: int) -> np.ndarray:
    """
    Mask for randomly sampling 1D array of given length.

    Parameters:
    ----------
    length : int
        Mask length
    num_samples : int
        Number of samples.
    seed : int
        Random seed

    Returns:
    ----------
    ndarray
        Mask.
    """
    # Use seed to select num_train_samples points
    rng = np.random.default_rng(seed)
    mask = rng.choice(length, num_samples, replace=False)
    return mask


# Mapping from data set sampler method labels to python functions
DATA_SET_SAMPLER_REGISTRY: dict[
    DataSetSamplerLabels, Callable[[int, int, int], np.ndarray]
] = {
    DataSetSamplerLabels.NONE: mask_no_sampler,
    DataSetSamplerLabels.RANDOM: mask_random_array_sampler,
}
