import numpy as np
from validate_data import validate_numpy_array


class CoordinateSystem:
    """Base class for coordinate systems."""

    def __init__(self, dim: int):
        # Common dimension checks for all coordinate systems
        if not isinstance(dim, int):
            raise TypeError(f"Expected dim to be an integer, got {type(dim).__name__}")
        if dim < 1:
            raise ValueError(f"dim must be >= 1, got {dim}")

        self.dim = dim


class CartesianCoords(CoordinateSystem):
    """Cartesian coordinate system: x, y, z."""

    def __init__(self, dim: int):
        super().__init__(dim)
        self.validate_dimensions()

    def metric_tensor(self, _coords: np.ndarray | None = None) -> np.ndarray:
        """
        Return the metric tensor for Cartesian coordinates.

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

    def validate_dimensions(self) -> None:
        """
        Validate that the Cartesian coordinate system supports the given number of
        dimensions.

        Raises
        ------
        ValueError
            If coordinate system is incompatible with the given number of dimensions.
        """
        # Raise error if dimensions greater than 3
        if self.dim > 3:
            raise ValueError(
                f"Cartesian coordinates only implemented for dim<=3, got dim={self.dim}"
            )

    def validate_coordinates(self, coords: np.ndarray) -> None:
        """
        Validate that the provided coordinates are compatible with Cartesian
        coordinates.

        Parameters
        ----------
        coords : ndarray
            Coordinates to validate.

        Raises
        ------
        ValueError
            If coordinates are incompatible with Cartesian coordinates.
        """
        # Validate coords array type and shape
        validate_numpy_array(coords, (self.dim,))


class CylindricalCoords(CoordinateSystem):
    """Cylindrical polar coordinate system: r, theta, z."""

    def __init__(self, dim: int):
        super().__init__(dim)
        self.validate_dimensions()

    def metric_tensor(self, coords: np.ndarray) -> np.ndarray:
        """
        Return the metric tensor for cylindrical polar coordinates.

        Parameters
        ----------
        coords : ndarray
            Coordinates (r, theta [rad], z).

        Returns
        -------
        g : ndarray
            Metric tensor.

        """
        # Validate coords array
        self.validate_coordinates(coords)

        g = np.eye(self.dim)
        g[1, 1] = coords[0] ** 2  # theta component
        return g

    def validate_dimensions(self) -> None:
        """
        Validate that the cylindrical polar coordinate system supports the given number
        of dimensions.

        Raises
        ------
        ValueError
            If coordinate system is incompatible with the given number of dimensions.
        """
        # Raise error if dimensions greater than 3
        if self.dim > 3:
            raise ValueError(
                "Cylindrical polar coordinates only implemented for dim<=3, "
                f"got dim={self.dim}"
            )

        # Suggest Cartesian for 1D
        if self.dim == 1:
            raise ValueError(
                "To enforce consistent coordinate system labelling, use Cartesian "
                "instead of cylindrical polar coordinates when using dim=1"
            )

    def validate_coordinates(self, coords: np.ndarray) -> None:
        """
        Validate that the provided coordinates are compatible with cylindrical polar
        coordinates.

        Parameters
        ----------
        coords : ndarray
            Coordinates to validate.

        Raises
        ------
        ValueError
            If coordinates are incompatible with cylindrical polar coordinates.
        """
        # Validate coords array type and shape
        validate_numpy_array(coords, (self.dim,))

        # Check that radius is non-negative
        if coords[0] < 0:
            raise ValueError(
                "In cylindrical polar coordinates, radius must be non-negative, "
                f"got r={coords[0]}"
            )

        # Check that theta is within [0, 2pi)
        if not (0.0 <= coords[1] < 2 * np.pi):
            raise ValueError(
                "In cylindrical polar coordinates, theta must be in [0, 2pi), "
                f"got theta={coords[1]}"
            )


class SphericalCoords(CoordinateSystem):
    """Spherical polar coordinate system: r, theta, phi."""

    def __init__(self, dim: int):
        super().__init__(dim)
        self.validate_dimensions()

    def metric_tensor(self, coords: np.ndarray) -> np.ndarray:
        """
        Return the metric tensor for spherical polar coordinates.

        Parameters
        ----------
        coords : ndarray
            Coordinates (r, theta [rad], phi [rad]).

        Returns
        -------
        g : ndarray
            Metric tensor.

        """
        # Validate coords array
        self.validate_coordinates(coords)

        g = np.eye(self.dim)
        g[1, 1] = coords[0] ** 2  # theta component
        g[2, 2] = coords[0] ** 2 * np.sin(coords[1]) ** 2  # phi component
        return g

    def validate_dimensions(self) -> None:
        """
        Validate that the spherical polar coordinate system supports the given number
        of dimensions.

        Raises
        ------
        ValueError
            If coordinate system is incompatible with the given number of dimensions.
        """
        # Raise error if dimensions greater than 3
        if self.dim > 3:
            raise ValueError(
                "Spherical polar coordinates only implemented for dim<=3, "
                f"got dim={self.dim}"
            )

        # Suggest Cartesian for 1D
        if self.dim == 1:
            raise ValueError(
                "To enforce consistent coordinate system labelling, use Cartesian "
                "instead of spherical polar coordinates when using dim=1"
            )

        # Suggest cylindrical polar for 2D
        if self.dim == 2:
            raise ValueError(
                "To enforce consistent coordinate system labelling, use cylindrical "
                "instead of spherical polar coordinates when using dim=2"
            )

    def validate_coordinates(self, coords: np.ndarray) -> None:
        """
        Validate that the provided coordinates are compatible with spherical polar
        coordinates.

        Parameters
        ----------
        coords : ndarray
            Coordinates to validate.

        Raises
        ------
        ValueError
            If coordinates are incompatible with spherical polar coordinates.
        """
        # Validate coords array type and shape
        validate_numpy_array(coords, (self.dim,))

        # Check that radius is non-negative
        if coords[0] < 0:
            raise ValueError(
                "In spherical polar coordinates, radius must be non-negative, "
                f"got r={coords[0]}"
            )

        # Check that theta is within [0, pi]
        if not (0.0 <= coords[1] <= np.pi):
            raise ValueError(
                "In spherical polar coordinates, theta must be in [0, pi], "
                f"got theta={coords[1]}"
            )

        # Check that phi is within [0, 2pi)
        if not (0.0 <= coords[2] < 2 * np.pi):
            raise ValueError(
                "In spherical polar coordinates, phi must be in [0, 2pi), "
                f"got phi={coords[2]}"
            )
