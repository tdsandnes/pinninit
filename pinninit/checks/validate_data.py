import numpy as np


def validate_coords_array(coords: np.ndarray, dim: int) -> None:
    """Validate that coords is a NumPy array and has the correct shape.

    Parameters
    ----------
    coords : ndarray
        Coordinates to validate.
    dim : int
        Expected dimension of the coordinates.

    Raises
    ------
    TypeError
        If coords is not a NumPy array.
    ValueError
        If coords is not of shape (dim,).
    """

    # Check type
    if not isinstance(coords, np.ndarray):
        raise TypeError(
            "[validate_coords_array] Expected coords to be a NumPy array, "
            f"got {type(coords).__name__}"
        )

    # Check that coords have correct shape
    if coords.shape != (dim,):
        raise ValueError(
            f"[validate_coords_array] Expected shape ({dim},), got {coords.shape}"
        )
