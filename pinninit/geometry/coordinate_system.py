from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

import numpy as np
import torch
from utils import interval_str
from validate_data import validate_pytorch_tensor


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


class CoordSystem(ABC):
    """Base class for coordinate systems."""

    # Placeholder for subclass class attributes
    metadata: CoordSystemMetadata

    def __init__(self, dim: int):
        self.dim = dim

        # Ensure coordinate system metadata exists and is of correct type
        metadata = getattr(self, "metadata", None)
        if metadata is None:
            raise NotImplementedError(
                f"{type(self).__name__} lacks a 'metadata' class attribute"
            )

        if not isinstance(metadata, CoordSystemMetadata):
            raise TypeError(
                "'metadata' must be of type CoordSystemMetadata, "
                f"got {type(metadata)}"
            )

        # Ensure coordinate system is valid
        self._validate_dimensions()
        self._validate_bounds()

    @abstractmethod
    def metric_tensor(self, n_points: int, coords: torch.Tensor) -> torch.Tensor:
        """
        Calculates the metric tensor(s) at a set of coords.

        Subclasses must implement this method.

        Parameters
        ----------
        n_points : int
            Number of points for which to generate metric tensors.
        coords : torch.Tensor
            Coordinates of the points. Shape: (n_points, dim).

        Returns
        -------
        torch.Tensor
            Metric tensors of shape (n_points, dim, dim).
        """
        raise NotImplementedError(
            "Subclasses of CoordSystem must implement " "metric_tensor method."
        )

    def grad(self, f: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """
        Compute the gradient of a function in a given coordinate system.

        Parameters
        ----------
        f : torch.Tensor
            Function to be taken gradient of. Shape: (n_points,)
        coords : torch.Tensor
            A set of points and their spatial coordinates. Shape: (n_points, dim)

        Returns
        -------
        torch.Tensor
            Gradient of the function: (n_points, dim)
        """
        # Coordinate system dimensions
        dim = self.dim

        # Validate input tensors
        n_points = f.shape[0]
        validate_pytorch_tensor(f, (n_points,))
        validate_pytorch_tensor(coords, (n_points, dim))

        # Reshape for derivative calculation
        f = f.unsqueeze(1)  # (n_points, 1)

        # Calculate derivatives using autograd
        derivative_f = torch.autograd.grad(
            f,
            coords,
            grad_outputs=torch.ones_like(f),
            create_graph=True,
            allow_unused=True,  # If not a function of a coord, gives 0 rather than None
        )[0]

        # Set derivatives to zero in None (Case where f is indep of all coords)
        if derivative_f is None:
            derivative_f = torch.zeros_like(coords)

        # Tensor of metric tensors for each point
        G = self.metric_tensor(n_points, coords)  # Shape: (n_points, dim, dim)
        validate_pytorch_tensor(G, (n_points, dim, dim))

        # Inverse metric tensor
        G_inv = torch.linalg.inv(G)  # Shape: (n_points, dim, dim)
        validate_pytorch_tensor(G_inv, (n_points, dim, dim))

        # Batch matrix multiplication:
        # (n_points, dim, dim) @ (n_points, dim, 1) -> (n_points, dim, 1)
        # Squeeze the result to get the final (n_points, dim) vector
        grad_f = torch.bmm(G_inv, derivative_f.unsqueeze(2)).squeeze(2)
        validate_pytorch_tensor(grad_f, (n_points, dim))

        return grad_f

    def _validate_dimensions(self) -> None:
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

        # Coordinate system dimensions
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

    def _validate_bounds(self) -> None:
        """
        Validate the bounds of axes in the coordinate system.

        Raises
        ------
        ValueError
            If axes bounds and their inclusivity are inconsistent.
        """
        # Coordinate system dimensions
        self._validate_dimensions()
        dim = self.dim

        # Extract metadata
        axes_bounds = np.array(self.metadata.axes_bounds)
        axes_bounds_inclusive = np.array(self.metadata.axes_bounds_inclusive)

        # Check that axes_bounds and axes_bounds_inclusive have correct shape
        if (np.shape(axes_bounds) != (dim, 2)) or (
            np.shape(axes_bounds_inclusive) != (dim, 2)
        ):
            raise ValueError(
                "axes_bounds and axes_bounds_inclusive must be of " "shape (dim, 2)."
            )

        # Separate lower and upper bounds and their inclusivity
        lower_bounds = axes_bounds[:, 0]
        upper_bounds = axes_bounds[:, 1]
        lower_bounds_inclusive = axes_bounds_inclusive[:, 0]
        upper_bounds_inclusive = axes_bounds_inclusive[:, 1]

        # Find which elements are None
        lower_bound_is_none = np.array([x is None for x in lower_bounds])
        upper_bound_is_none = np.array([x is None for x in upper_bounds])
        lower_bound_inclusive_is_none = np.array(
            [x is None for x in lower_bounds_inclusive]
        )
        upper_bound_inclusive_is_none = np.array(
            [x is None for x in upper_bounds_inclusive]
        )

        # Validate the lower bounds inclusivity are either True, False, or None
        if not np.all(
            np.logical_or(
                lower_bound_is_none, np.isin(lower_bounds_inclusive, [True, False])
            )
        ):
            raise ValueError("Lower bound inclusivity must be True, False, or None")

        # Validate that upper bounds inclusivity are either True, False, or None
        if not np.all(
            np.logical_or(
                upper_bound_is_none, np.isin(upper_bounds_inclusive, [True, False])
            )
        ):
            raise ValueError("Upper bound inclusivity must be True, False, or None")

        # Condition A: None only if both bounds and inclusives are None
        lower_a = np.logical_and(lower_bound_is_none, lower_bound_inclusive_is_none)
        upper_a = np.logical_and(upper_bound_is_none, upper_bound_inclusive_is_none)

        # Condition B: Both are NOT None (Bound is Value AND Inclusive is True/False)
        lower_b = np.logical_and(~lower_bound_is_none, ~lower_bound_inclusive_is_none)
        upper_b = np.logical_and(~upper_bound_is_none, ~upper_bound_inclusive_is_none)

        # Check if all elements satisfy (Condition A OR Condition B)
        lower_ab = np.logical_or(lower_a, lower_b)
        upper_ab = np.logical_or(upper_a, upper_b)

        # Both lower and upper bounds must be consistent, otherwise raise error
        if not np.all(np.logical_and(lower_ab, upper_ab)):
            raise ValueError("Inconsistent axes bounds.")

    def _validate_coordinates(self, coords: torch.Tensor) -> None:
        """
        Validate that a set of provided coordinates are compatible with the coordinate
        system.

        Parameters
        ----------
        coords : torch.Tensor
            Coordinates to validate. Shape: (n_points, dim)

        Raises
        ------
        ValueError
            If coordinates are incompatible with the coordinate system.
        """
        # Coordinate system dimensions
        self._validate_dimensions()
        dim = self.dim

        # Extract metadata
        axes_labels = self.metadata.axes_labels
        self._validate_bounds()
        axes_bounds = self.metadata.axes_bounds
        axes_bounds_inclusive = self.metadata.axes_bounds_inclusive

        # Validate coords tensor type and shape
        n_points = coords.shape[0]
        validate_pytorch_tensor(coords, (n_points, dim))

        # Separate lower and upper bounds and inclusivity
        lower_bounds = np.array(axes_bounds)[:, 0]
        upper_bounds = np.array(axes_bounds)[:, 1]
        lower_inclusive = np.array(axes_bounds_inclusive)[:, 0]
        upper_inclusive = np.array(axes_bounds_inclusive)[:, 1]

        # Convert bounds to tensors, replacing None with -inf / +inf. Reshape to
        # (1, dim) so that PyTorch will broadcast to (n_points, dim) in comparisons with
        # coords which has shape (n_points, dim).
        lower_bounds = torch.tensor(
            [lb if lb is not None else -torch.inf for lb in lower_bounds],
            dtype=coords.dtype,
        ).unsqueeze(0)
        upper_bounds = torch.tensor(
            [ub if ub is not None else torch.inf for ub in upper_bounds],
            dtype=coords.dtype,
        ).unsqueeze(0)
        lower_inclusive = torch.tensor(
            [li if li is not None else True for li in lower_inclusive], dtype=torch.bool
        ).unsqueeze(0)
        upper_inclusive = torch.tensor(
            [ui if ui is not None else True for ui in upper_inclusive], dtype=torch.bool
        ).unsqueeze(0)

        # Vectorized mask checks
        mask_in_bounds_lower = torch.where(
            lower_inclusive, coords >= lower_bounds, coords > lower_bounds
        )
        mask_in_bounds_upper = torch.where(
            upper_inclusive, coords <= upper_bounds, coords < upper_bounds
        )

        # Out-of-bounds mask
        mask_out_of_bounds = ~(mask_in_bounds_lower & mask_in_bounds_upper)

        # Initialise list to collect error messages
        errors = []

        # Iterate over dimensions to find and report out-of-bounds elements
        for idx_dim, label in enumerate(axes_labels):
            # Get n_points indices where this coordinate is out of bounds
            idx_points_errors = torch.nonzero(mask_out_of_bounds[:, idx_dim]).squeeze(1)
            if idx_points_errors.numel() == 0:
                continue  # no errors for this coordinate

            # String representation of bounds
            bounds_str = interval_str(
                np.array(axes_bounds)[idx_dim, :], np.array(axes_bounds)[idx_dim, :]
            )

            # Generate messages for this dimension
            for idx_points_error in idx_points_errors:
                value = coords[idx_points_error, idx_dim].item()
                errors.append(
                    f"Coordinate '{label}'={value} is out of bounds {bounds_str}"
                )

        # Raise error if any violations
        if errors:
            raise ValueError("\n".join(errors))


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

    def metric_tensor(
        self, n_points: int, _coords: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
         Return the metric tensor(s) for Cartesian coordinates.

        Parameters
         ----------
         n_points : int
             Number of points for which to generate metric tensors.
         _coords : torch.Tensor
             Coordinates of the points. Unused for Cartesian coordinates.

         Returns
         -------
         torch.Tensor
             Metric tensors of shape (n_points, dim, dim).
        """
        g = torch.eye(self.dim)
        G = g.unsqueeze(0).repeat(n_points, 1, 1)  # Shape: (n_points, dim, dim)
        return G


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

    def metric_tensor(self, n_points: int, coords: torch.Tensor) -> torch.Tensor:
        """
         Return the metric tensor(s) for cylindrical polar coordinates.

        Parameters
         ----------
         n_points : int
             Number of points for which to generate metric tensors.
         coords : torch.Tensor
             Coordinates of the points. Shape: (n_points, dim).

         Returns
         -------
         torch.Tensor
             Metric tensors of shape (n_points, dim, dim).
        """
        # Validate coords tensor
        self._validate_coordinates(coords)

        g = torch.eye(self.dim)
        G = g.unsqueeze(0).repeat(n_points, 1, 1)  # Shape: (n_points, dim, dim)
        G[:, 1, 1] = coords[:, 0] ** 2  # theta theta components
        return G


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

    def metric_tensor(self, n_points: int, coords: torch.Tensor) -> torch.Tensor:
        """
         Return the metric tensor(s) for spherical polar coordinates.

        Parameters
         ----------
         n_points : int
             Number of points for which to generate metric tensors.
         coords : torch.Tensor
             Coordinates of the points. Shape: (n_points, dim).

         Returns
         -------
         torch.Tensor
             Metric tensors of shape (n_points, dim, dim).
        """
        # Validate coords tensor
        self._validate_coordinates(coords)

        g = torch.eye(self.dim)
        G = g.unsqueeze(0).repeat(n_points, 1, 1)  # Shape: (n_points, dim, dim)
        G[:, 1, 1] = coords[:, 0] ** 2  # theta theta components
        G[:, 2, 2] = (
            coords[:, 0] ** 2 * torch.sin(coords[:, 1]) ** 2
        )  # phi phi components
        return G


# Mapping from coordinate system labels to classes
COORD_SYSTEM_REGISTRY = {
    CoordSystemLabels.CARTESIAN: CartesianCoords,
    CoordSystemLabels.CYLINDRICAL: CylindricalCoords,
    CoordSystemLabels.SPHERICAL: SphericalCoords,
}
