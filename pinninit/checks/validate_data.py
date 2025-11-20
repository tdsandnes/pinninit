import numpy as np


def validate_numpy_array(arr: np.ndarray, shape: tuple[int, ...]) -> None:
    """Validate that the input is a NumPy array with the expected shape.

    Parameters
    ----------
    arr : ndarray
        array to validate.
    shape : tuple
        Expected shape of the array.

    Raises
    ------
    TypeError
        If arr is not a NumPy array.
    ValueError
        If arr does not have expected shape.
    """

    # Check that arr is a NumPy array
    if not isinstance(arr, np.ndarray):
        raise TypeError(
            "[validate_numpy_array] Expected input to be a NumPy array, "
            f"got {type(arr).__name__}"
        )

    # Check that arr has correct shape
    if arr.shape != shape:
        raise ValueError(
            f"[validate_numpy_array] Expected input to have shape {shape}, "
            f"got {arr.shape}"
        )
