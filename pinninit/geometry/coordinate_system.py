import numpy as np
from utils import validate_coords_array


class CoordinateSystem:
    """Base class for coordinate systems."""

    def __init__(self, dim: int):
        self.dim = dim


class CartesianCoords(CoordinateSystem):
    """Cartesian coordinate system: x, y, z."""

    def __init__(self, dim: int):
        # Raise error if dimensions greater than 3
        if dim > 3:
            raise ValueError(
                f"Cartesian coordinates only implemented for dim<=3, got dim={dim}"
            )
        super().__init__(dim)

    def metric_tensor(self, _coords: np.ndarray | None = None) -> np.ndarray:
        """Return the metric tensor for Cartesian coordinates.

        Parameters
        ----------
        _coords : ndarray , optional
            Coordinates (not used in Cartesian coordinate system).

        Returns
        -------
        g : ndarray
            Metric tensor.

        """
        g = np.eye(self.dim)
        return g


class CylindricalCoords(CoordinateSystem):
    """Cylindrical polar coordinate system: r, theta, z."""

    def __init__(self, dim: int):
        # Raise error if dimensions not 2 or 3
        if dim not in (2, 3):
            raise ValueError(
                "Cylindrical polar coordinates only implemented for dim=2 or 3, "
                f"got dim={dim}"
            )
        super().__init__(dim)

    def metric_tensor(self, coords: np.ndarray) -> np.ndarray:
        """Return the metric tensor for cylindrical polar coordinates.

        Parameters
        ----------
        coords : ndarray
            Coordinates (r, theta [rad], z).

        Returns
        -------
        g : ndarray
            Metric tensor.

        """
        # Validate coords array type and shape
        validate_coords_array(coords, self.dim)

        g = np.eye(self.dim)
        g[1, 1] = coords[0] ** 2  # theta component
        return g


class SphericalCoords(CoordinateSystem):
    """Spherical polar coordinate system: r, theta, phi."""

    def __init__(self, dim: int):
        # Raise error if dimensions not 3
        if dim != 3:
            raise ValueError(
                f"Spherical polar coordinates only implemented for dim=3, got dim={dim}"
            )
        super().__init__(dim)

    def metric_tensor(self, coords: np.ndarray) -> np.ndarray:
        """Return the metric tensor for spherical polar coordinates.

        Parameters
        ----------
        coords : ndarray
            Coordinates (r, theta [rad], phi [rad]).

        Returns
        -------
        g : ndarray
            Metric tensor.

        """
        # Validate coords array type and shape
        validate_coords_array(coords, self.dim)

        g = np.eye(self.dim)
        g[1, 1] = coords[0] ** 2  # theta component
        g[2, 2] = coords[0] ** 2 * np.sin(coords[1]) ** 2  # phi component
        return g
