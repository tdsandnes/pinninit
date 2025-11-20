from dataclasses import dataclass
from enum import Enum

import numpy as np
from utils import check_number_within_bounds, interval_str
from validate_data import validate_numpy_array


class CoordSystemLabels(str, Enum):
    """Enumeration of coordinate system labels."""

    CARTESIAN = "cartesian"
    CYLINDRICAL = "cylindrical"
    SPHERICAL = "spherical"


@dataclass(frozen=True)
class CoordSystemMetadata:
    """Dataclass for coordinate system metadata."""

    name: str
    axes_labels: tuple[str, ...]
    axes_bounds: tuple[tuple[float | None, float | None], ...]
    axes_bounds_inclusive: tuple[tuple[bool | None, bool | None], ...]
    max_dim: int
    dim_redirections: dict[int, CoordSystemLabels] | None


class CoordSystem:
    """Base class for coordinate systems."""

    # Placeholder for subclass class attributes
    metadata: CoordSystemMetadata

    def __init__(self, dim: int):
        self.dim = dim

    def validate_dimensions(self) -> None:
        """
        Validate that the coordinate system supports the given number of dimensions.

        Raises
        ------
        TypeError
            If dim is not an integer.
        ValueError
            If coordinate system is incompatible with the given number of dimensions.
        """
        # Extract metadata
        name = self.metadata.name
        max_dim = self.metadata.max_dim
        dim_redirections = self.metadata.dim_redirections

        # Coordinate system parameters
        dim = self.dim

        # Check type of dim
        if not isinstance(dim, int):
            raise TypeError(f"Expected dim to be an integer, got {type(dim).__name__}")

        # Check value of dim
        if dim < 1:
            raise ValueError(f"dim must be >= 1, got {dim}")
        if dim > max_dim:
            raise ValueError(
                f"{name} only implemented for dim<={max_dim}, got dim={dim}"
            )

        # Error to redirect user to alternative coordinate system if necessary
        if dim_redirections and dim in dim_redirections:
            alt_coord_system_label = dim_redirections[dim]
            alt_coord_system_name = COORD_SYSTEM_REGISTRY[
                alt_coord_system_label
            ].metadata.name
            raise ValueError(
                f"To enforce consistent coordinate system labelling, use "
                f"{alt_coord_system_name} instead of {name} when using dim={dim}"
            )

    def validate_coordinates(self, coords: np.ndarray) -> None:
        """
        Validate that a set of provided coordinates are compatible with the coordinate
        system.

        Parameters
        ----------
        coords : ndarray
            Coordinates to validate.

        Raises
        ------
        ValueError
            If coordinates are incompatible with the coordinate system.
        """
        # Extract metadata
        axes_labels = self.metadata.axes_labels
        axes_bounds = self.metadata.axes_bounds
        axes_bounds_inclusive = self.metadata.axes_bounds_inclusive

        # Coordinate system parameters
        dim = self.dim

        # Validate coords array type and shape
        validate_numpy_array(coords, (dim,))

        # Check each coordinate against its bounds
        errors = []
        for i in range(dim):
            value = coords[i]
            axes_label = axes_labels[i]
            bounds = axes_bounds[i]
            bounds_inclusive = axes_bounds_inclusive[i]

            # Check if value is within bounds
            in_bounds = check_number_within_bounds(value, bounds, bounds_inclusive)
            if not in_bounds:
                bounds_str = interval_str(bounds, bounds_inclusive)
                errors.append(
                    f"Coordinate '{axes_label}'={value} is out of coordinate "
                    f"system bounds {bounds_str}"
                )

        if errors:
            # Join error messages with line breaks and raise an error
            errors_joined = "\n".join(errors)
            raise ValueError(f"{errors_joined}")


class CartesianCoords(CoordSystem):
    """Cartesian coordinate system: x, y, z."""

    # Metadata as class attributes
    metadata = CoordSystemMetadata(
        name="Cartesian Coordinates",
        axes_labels=("x", "y", "z"),
        axes_bounds=((None, None), (None, None), (None, None)),
        axes_bounds_inclusive=((None, None), (None, None), (None, None)),
        max_dim=3,
        dim_redirections=None,
    )

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


class CylindricalCoords(CoordSystem):
    """Cylindrical polar coordinate system: r, theta, z."""

    # Metadata as class attributes
    metadata = CoordSystemMetadata(
        name="Cylindrical Polar Coordinates",
        axes_labels=("r", "theta", "z"),
        axes_bounds=((0, None), (0, 2 * np.pi), (None, None)),
        axes_bounds_inclusive=((True, None), (True, False), (None, None)),
        max_dim=3,
        dim_redirections={1: CoordSystemLabels.CARTESIAN},
    )

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


class SphericalCoords(CoordSystem):
    """Spherical polar coordinate system: r, theta, phi."""

    # Metadata as class attributes
    metadata = CoordSystemMetadata(
        name="Spherical Polar Coordinates",
        axes_labels=("r", "theta", "phi"),
        axes_bounds=((0, None), (0, np.pi), (0, 2 * np.pi)),
        axes_bounds_inclusive=((True, None), (True, True), (True, False)),
        max_dim=3,
        dim_redirections={
            1: CoordSystemLabels.CARTESIAN,
            2: CoordSystemLabels.CYLINDRICAL,
        },
    )

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


# Mapping from coordinate system labels to classes
COORD_SYSTEM_REGISTRY = {
    CoordSystemLabels.CARTESIAN: CartesianCoords,
    CoordSystemLabels.CYLINDRICAL: CylindricalCoords,
    CoordSystemLabels.SPHERICAL: SphericalCoords,
}
