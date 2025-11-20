import numpy as np
import pytest
from validate_data import validate_numpy_array


def test_validate_numpy_array_correct() -> None:
    # Should not raise
    validate_numpy_array(np.array([]), (0,))  # empty array of shape (0,)
    validate_numpy_array(np.array([1.0, 2.0, 3.0]), (3,))  # 1D array of shape (3,)
    validate_numpy_array(np.array([1.0, 2.0]), (2,))  # 1D array of shape (2,)
    validate_numpy_array(np.array([1.0]), (1,))  # 1D array of shape (1,)
    arr_2d = np.array([[1.0, 2.0], [3.0, 4.0]])  # 2D array of shape (2,2)
    validate_numpy_array(arr_2d, (2, 2))


def test_validate_numpy_array_wrong_type() -> None:
    arr = [1.0, 2.0, 3.0]  # list, not NumPy array
    with pytest.raises(TypeError):
        validate_numpy_array(arr, (3,))


def test_validate_numpy_array_wrong_shape() -> None:
    arr = np.array([[1.0, 2.0, 3.0]])  # 2D array of shape (1,3)
    with pytest.raises(ValueError):
        validate_numpy_array(arr, (3,))

    arr = np.array([1.0, 2.0, 3.0, 4.0])  # correct number of elements but wrong shape
    with pytest.raises(ValueError):
        validate_numpy_array(arr, (2, 2))

    arr = np.array([])  # empty array
    with pytest.raises(ValueError):
        validate_numpy_array(arr, (1,))
