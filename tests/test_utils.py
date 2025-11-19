import numpy as np
import pytest
from utils import validate_coords_array


def test_validate_coords_array_correct() -> None:
    # Should not raise
    validate_coords_array(np.array([1.0, 2.0, 3.0]), 3)
    validate_coords_array(np.array([1.0, 2.0]), 2)
    validate_coords_array(np.array([1.0]), 1)


def test_validate_coords_array_wrong_type() -> None:
    coords = [1.0, 2.0, 3.0]  # list, not ndarray
    with pytest.raises(TypeError):
        validate_coords_array(coords, 3)


def test_validate_coords_array_wrong_shape() -> None:
    coords = np.array([[1.0, 2.0, 3.0]])  # shape (1,3)
    with pytest.raises(ValueError):
        validate_coords_array(coords, 3)


def test_validate_coords_array_wrong_dim() -> None:
    coords = np.array([1.0, 2.0])  # shape (2,), dim = 3
    with pytest.raises(ValueError):
        validate_coords_array(coords, 3)
