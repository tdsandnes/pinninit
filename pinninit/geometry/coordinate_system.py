from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
import torch

from ..checks.validate_data import validate_pytorch_tensor
from ..utils import interval_str, lookup_from_registry
from .geometry_utils import get_required_or_fixed_coord, select_coord_system_axes

if TYPE_CHECKING:
    from collections.abc import Callable


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
    required_or_fixed_axes: tuple[str, ...]
    required_fields: tuple[str, ...]
    required_consts: tuple[str, ...]
    axes_bounds: tuple[tuple[float | None, float | None], ...]
    axes_bounds_inclusive: tuple[tuple[bool | None, bool | None], ...]
    max_coord_dim: int
    coord_redirections: dict[tuple[int, tuple[str, ...]], CoordSystemLabels] | None


class CoordSystem(ABC):
    """Base class for coordinate systems."""

    # Placeholder for subclass class attributes
    metadata: CoordSystemMetadata

    def __init__(
        self,
        coord_dim: int,
        coord_axes: tuple[str],
        fixed_axes: tuple[str] | None = None,
        fixed_axes_values: tuple[float] | None = None,
    ):
        self.coord_dim = coord_dim

        # Ensure coordinate system metadata exists and is of correct type
        metadata = getattr(self, "metadata", None)
        if metadata is None:
            raise NotImplementedError(
                f"{type(self).__name__} lacks a 'metadata' class attribute"
            )

        if not isinstance(metadata, CoordSystemMetadata):
            raise TypeError(
                f"'metadata' must be of type CoordSystemMetadata, got {type(metadata)}"
            )

        (
            self.axes_labels_selected,
            self.axes_bounds_selected,
            self.axes_bounds_inclusive_selected,
            self.indices_selected,
        ) = select_coord_system_axes(
            self.metadata.axes_labels,
            self.metadata.axes_bounds,
            self.metadata.axes_bounds_inclusive,
            coord_axes,
        )
        self.fixed_axes = fixed_axes
        self.fixed_axes_values = fixed_axes_values

        # Ensure coordinate system is valid
        self._validate_dimensions()
        self._validate_selected_axes()
        self._validate_required_or_fixed_axes()
        self._validate_bounds()

    @abstractmethod
    def metric_tensor(
        self, n_points: int, coords: torch.Tensor, device: torch.device | None = None
    ) -> torch.Tensor:
        """Subclasses must implement this method to return the metric tensor(s)."""
        pass

    def grad(
        self, f: torch.Tensor, coords: torch.Tensor, device: torch.device | None = None
    ) -> torch.Tensor:
        """
        Compute the gradient of a function in a given coordinate system.

        Parameters
        ----------
        f : torch.Tensor
            Function to be taken gradient of. Shape: (n_points,)
        coords : torch.Tensor
            A set of points and their spatial coordinates. Shape: (n_points, coord_dim)

        Returns
        -------
        torch.Tensor
            Gradient of the function: (n_points, coord_dim)
        """
        # Coordinate system dimensions
        coord_dim = self.coord_dim

        # Validate input tensors
        n_points = f.shape[0]
        validate_pytorch_tensor(f, (n_points,))
        validate_pytorch_tensor(coords, (n_points, coord_dim))

        # Reshape for derivative calculation
        f = f.unsqueeze(1)  # (n_points, 1)

        # Calculate derivatives using autograd
        derivative_f = torch.autograd.grad(
            f,
            coords,
            grad_outputs=torch.ones_like(f),
            create_graph=True,
            retain_graph=True,
            allow_unused=True,  # If not a function of a coord, gives 0 rather than None
        )[0]

        # Set derivatives to zero in None (Case where f is indep of all coords)
        if derivative_f is None:
            derivative_f = torch.zeros_like(coords)

        # Tensor of metric tensors for each point
        G = self.metric_tensor(
            n_points, coords, device
        )  # Shape: (n_points, coord_dim, coord_dim)
        validate_pytorch_tensor(G, (n_points, coord_dim, coord_dim))

        # Inverse metric tensor
        G_inv = torch.linalg.inv(G)  # Shape: (n_points, coord_dim, coord_dim)
        validate_pytorch_tensor(G_inv, (n_points, coord_dim, coord_dim))

        # Batch matrix multiplication:
        # (n_points, coord_dim, coord_dim) @ (n_points, coord_dim, 1) ->
        # (n_points, coord_dim, 1)
        # Squeeze the result to get the final (n_points, coord_dim) vector
        grad_f = torch.bmm(G_inv, derivative_f.unsqueeze(2)).squeeze(2)
        validate_pytorch_tensor(grad_f, (n_points, coord_dim))

        return grad_f

    def _validate_dimensions(self) -> None:
        """
        Validate that the coordinate system supports the given number of dimensions.

        Raises
        ------
        TypeError
            If coord_dim is not an integer.
        ValueError
            If coordinate system is incompatible with the given number of dimensions.
        """
        # Extract metadata
        name = self.metadata.name
        max_coord_dim = self.metadata.max_coord_dim

        # Coordinate system dimensions
        coord_dim = self.coord_dim

        # Check type of coord_dim
        if not isinstance(coord_dim, int):
            raise TypeError(
                f"Expected coord_dim to be an integer, got {type(coord_dim).__name__}"
            )

        # Check value of coord_dim
        if coord_dim < 2:
            raise ValueError(f"coord_dim must be >= 2, got {coord_dim}")
        if coord_dim > max_coord_dim:
            raise ValueError(
                f"{name} only implemented for coord_dim<={max_coord_dim}, "
                f"got coord_dim={coord_dim}"
            )

    def _validate_selected_axes(self) -> None:
        """
        Validate that the selected axes are compatible coordinate system.

        In cases where two coord systems behave the same, we raise an error in one of
        them to enforce consistent coordinate system labelling i.e. (t, r) cylindrical
        polars redirects you to (t, x) Cartesian.

        Raises
        ------
        ValueError
            If selected axes are not compatible.
        """

        # Extract metadata
        name = self.metadata.name
        coord_redirections = self.metadata.coord_redirections

        # Coordinate system dimensions
        self._validate_dimensions()
        coord_dim = self.coord_dim
        axes_labels_selected = self.axes_labels_selected

        # Error to redirect user to alternative coordinate system if necessary
        if (
            coord_redirections
            and (coord_dim, axes_labels_selected) in coord_redirections
        ):
            alt_coord_system_label = coord_redirections[
                (coord_dim, axes_labels_selected)
            ]
            alt_coord_system_class = lookup_from_registry(
                alt_coord_system_label, COORD_SYSTEM_REGISTRY
            )

            if isinstance(alt_coord_system_class, type) and issubclass(
                alt_coord_system_class, CoordSystem
            ):
                alt_coord_system_name = alt_coord_system_class.metadata.name
            else:
                raise TypeError(
                    "Expected a CoordSystem from lookup_from_registry, "
                    "got a function"
                )

            raise ValueError(
                f"To enforce consistent coordinate system labelling, use "
                f"{alt_coord_system_name} instead of {name} when using "
                f"coord_dim={coord_dim} with axes {axes_labels_selected}"
            )

    def _validate_required_or_fixed_axes(self) -> None:
        """
        Validate axes that are required either a selected axis or a fixed value.

        Raises
        ------
        ValueError
            If required axes are not given.
        """

        # Extract metadata
        required_or_fixed_axes = self.metadata.required_or_fixed_axes

        # Check lengths of fixed_axes and fixed_axes_values
        if self.fixed_axes is not None and self.fixed_axes_values is not None:
            if len(self.fixed_axes) != len(self.fixed_axes_values):
                raise ValueError(
                    "fixed_axes and fixed_axes_values must be of the same length."
                )
        elif self.fixed_axes is not None or self.fixed_axes_values is not None:
            raise ValueError(
                "fixed_axes and fixed_axes_values must both be None if one is."
            )

        # Check that all required axes are given
        for axis_label in required_or_fixed_axes:
            if (axis_label not in self.axes_labels_selected) and (
                self.fixed_axes is None or axis_label not in self.fixed_axes
            ):
                raise ValueError(f"Missing required axis: {axis_label}")

    def _validate_bounds(self) -> None:
        """
        Validate the bounds of axes in the coordinate system.

        Raises
        ------
        ValueError
            If axes bounds and their inclusivity are inconsistent.
        """
        # Extract metadata
        max_coord_dim = self.metadata.max_coord_dim
        axes_bounds = np.array(self.metadata.axes_bounds)
        axes_bounds_inclusive = np.array(self.metadata.axes_bounds_inclusive)

        # Coordinate system dimensions
        self._validate_dimensions()

        # Check that axes_bounds and axes_bounds_inclusive have correct shape
        if (np.shape(axes_bounds) != (max_coord_dim, 2)) or (
            np.shape(axes_bounds_inclusive) != (max_coord_dim, 2)
        ):
            raise ValueError(
                f"axes_bounds and axes_bounds_inclusive must be of shape "
                f"({max_coord_dim}, 2), got {np.shape(axes_bounds)} and "
                f"{np.shape(axes_bounds_inclusive)}."
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
            Coordinates to validate. Shape: (n_points, coord_dim)

        Raises
        ------
        ValueError
            If coordinates are incompatible with the coordinate system.
        """
        # Coordinate system dimensions
        self._validate_dimensions()
        coord_dim = self.coord_dim

        # Extract metadata
        axes_labels = self.metadata.axes_labels
        self._validate_bounds()
        axes_bounds = self.metadata.axes_bounds
        axes_bounds_inclusive = self.metadata.axes_bounds_inclusive

        # Validate coords tensor type and shape
        n_points = coords.shape[0]
        validate_pytorch_tensor(coords, (n_points, coord_dim))

        # Separate lower and upper bounds and inclusivity
        lower_bounds = np.array(axes_bounds)[:, 0]
        upper_bounds = np.array(axes_bounds)[:, 1]
        lower_inclusive = np.array(axes_bounds_inclusive)[:, 0]
        upper_inclusive = np.array(axes_bounds_inclusive)[:, 1]

        # Convert bounds to tensors, replacing None with -inf / +inf. Reshape to
        # (1, coord_dim) so that PyTorch will broadcast to (n_points, coord_dim) in
        # comparisons with coords which has shape (n_points, coord_dim).
        lower_bounds_tensor = torch.tensor(
            [lb if lb is not None else -torch.inf for lb in lower_bounds],
            dtype=coords.dtype,
        ).unsqueeze(0)
        upper_bounds_tensor = torch.tensor(
            [ub if ub is not None else torch.inf for ub in upper_bounds],
            dtype=coords.dtype,
        ).unsqueeze(0)
        lower_inclusive_tensor = torch.tensor(
            [li if li is not None else True for li in lower_inclusive], dtype=torch.bool
        ).unsqueeze(0)
        upper_inclusive_tensor = torch.tensor(
            [ui if ui is not None else True for ui in upper_inclusive], dtype=torch.bool
        ).unsqueeze(0)

        # Vectorized mask checks
        mask_in_bounds_lower = torch.where(
            lower_inclusive_tensor,
            coords >= lower_bounds_tensor,
            coords > lower_bounds_tensor,
        )
        mask_in_bounds_upper = torch.where(
            upper_inclusive_tensor,
            coords <= upper_bounds_tensor,
            coords < upper_bounds_tensor,
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
        axes_labels=("t", "x", "y", "z"),
        required_or_fixed_axes=(),
        required_fields=(),
        required_consts=(),
        axes_bounds=((None, None), (None, None), (None, None), (None, None)),
        axes_bounds_inclusive=((None, None), (None, None), (None, None), (None, None)),
        max_coord_dim=4,
        coord_redirections=None,
    )

    def __init__(
        self,
        coord_dim: int,
        coord_axes: tuple[str],
        fixed_axes: tuple[str] | None = None,
        fixed_axes_values: tuple[float] | None = None,
    ):
        super().__init__(coord_dim, coord_axes, fixed_axes, fixed_axes_values)
        self._cartesian_validate_selected_axes()

    def metric_tensor(
        self,
        n_points: int,
        _coords: torch.Tensor | None = None,
        device: torch.device | None = None,
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
             Metric tensors of shape (n_points, coord_dim, coord_dim).
        """
        g = torch.eye(self.metadata.max_coord_dim).to(device)
        G = g.unsqueeze(0).repeat(
            n_points, 1, 1
        )  # Shape: (n_points, coord_dim, coord_dim)

        # Only use rows and columns of axes included in coords
        mask_keep = [
            i for i in range(self.metadata.max_coord_dim) if i in self.indices_selected
        ]
        G = G[:, mask_keep, :][:, :, mask_keep]

        # Validate output
        validate_pytorch_tensor(G, (n_points, self.coord_dim, self.coord_dim))

        return G

    def _cartesian_validate_selected_axes(self) -> None:
        """
        Validate that the selected axes are compatible coordinate system.

        In cases where two combinations of selected coords behave the same, we raise an
        error in one of them to enforce consistent coordinate system labelling i.e.
        (t, x) and (t, y).

        Raises
        ------
        ValueError
            If selected axes are not compatible.
        """

        # Extract metadata
        name = self.metadata.name
        axes_labels = self.metadata.axes_labels

        # Only valid combinations
        valid_axes = [axes_labels[: self.coord_dim]]

        if self.axes_labels_selected not in valid_axes:
            raise ValueError(
                f"To enforce consistent coordinate system labelling, use "
                f"axes {valid_axes} instead of {self.axes_labels_selected} "
                f"when using {name} with coord_dim={self.coord_dim}"
            )


class CylindricalCoords(CoordSystem):
    """Cylindrical polar coordinate system: r, theta, z."""

    # Metadata as class attributes
    metadata = CoordSystemMetadata(
        name="Cylindrical Polar Coordinates",
        axes_labels=("t", "r", "theta", "z"),
        required_or_fixed_axes=("r",),
        required_fields=(),
        required_consts=(),
        axes_bounds=((None, None), (0, None), (0, 2 * np.pi), (None, None)),
        axes_bounds_inclusive=((None, None), (True, None), (True, False), (None, None)),
        max_coord_dim=4,
        coord_redirections={
            (2, ("t", "r")): CoordSystemLabels.CARTESIAN,
            (2, ("t", "z")): CoordSystemLabels.CARTESIAN,
            (2, ("r", "z")): CoordSystemLabels.CARTESIAN,
            (3, ("t", "r", "z")): CoordSystemLabels.CARTESIAN,
        },
    )

    def __init__(
        self,
        coord_dim: int,
        coord_axes: tuple[str],
        fixed_axes: tuple[str] | None = None,
        fixed_axes_values: tuple[float] | None = None,
    ):
        super().__init__(coord_dim, coord_axes, fixed_axes, fixed_axes_values)

    def metric_tensor(
        self, n_points: int, coords: torch.Tensor, device: torch.device | None = None
    ) -> torch.Tensor:
        """
         Return the metric tensor(s) for cylindrical polar coordinates.

        Parameters
         ----------
         n_points : int
             Number of points for which to generate metric tensors.
         coords : torch.Tensor
             Coordinates of the points. Shape: (n_points, coord_dim).

         Returns
         -------
         torch.Tensor
             Metric tensors of shape (n_points, coord_dim, coord_dim).
        """
        # Validate coords tensor
        self._validate_coordinates(coords)

        # Get quantities needed in metric tensor
        r = get_required_or_fixed_coord(
            "r",
            coords,
            self.axes_labels_selected,
            self.fixed_axes,
            self.fixed_axes_values,
        )

        g = torch.eye(self.metadata.max_coord_dim).to(device)
        G = g.unsqueeze(0).repeat(
            n_points, 1, 1
        )  # Shape: (n_points, coord_dim, coord_dim)
        G[:, 2, 2] = r  # theta theta components

        # Only use rows and columns of axes included in coords
        mask_keep = [
            i for i in range(self.metadata.max_coord_dim) if i in self.indices_selected
        ]
        G = G[:, mask_keep, mask_keep]

        # Validate output
        validate_pytorch_tensor(G, (n_points, self.coord_dim, self.coord_dim))

        return G

    def _cylindrical_validate_selected_axes(self) -> None:
        """
        Validate that the selected axes are compatible coordinate system.

        In cases where two combinations of selected coords behave the same, we raise an
        error in one of them to enforce consistent coordinate system labelling i.e.
        (t, theta) and (r, theta).

        Raises
        ------
        ValueError
            If selected axes are not compatible.
        """

        # Extract metadata
        name = self.metadata.name
        axes_labels = self.metadata.axes_labels

        # valid combinations
        valid_axes = [axes_labels[: self.coord_dim]]
        if self.coord_dim == 2:
            valid_axes.append(("t", "theta"))

        if self.axes_labels_selected not in valid_axes:
            raise ValueError(
                f"To enforce consistent coordinate system labelling, use "
                f"axes {valid_axes} instead of {self.axes_labels_selected} "
                f"when using {name} with coord_dim={self.coord_dim}"
            )


class SphericalCoords(CoordSystem):
    """Spherical polar coordinate system: r, theta, phi."""

    # Metadata as class attributes
    metadata = CoordSystemMetadata(
        name="Spherical Polar Coordinates",
        axes_labels=("t", "r", "theta", "phi"),
        required_or_fixed_axes=("r", "theta"),
        required_fields=(),
        required_consts=(),
        axes_bounds=((None, None), (0, None), (0, np.pi), (0, 2 * np.pi)),
        axes_bounds_inclusive=((None, None), (True, None), (True, True), (True, False)),
        max_coord_dim=4,
        coord_redirections={
            (2, ("t", "r")): CoordSystemLabels.CARTESIAN,
            (3, ("t", "r", "phi")): CoordSystemLabels.CYLINDRICAL,
        },
    )

    def __init__(
        self,
        coord_dim: int,
        coord_axes: tuple[str],
        fixed_axes: tuple[str] | None = None,
        fixed_axes_values: tuple[float] | None = None,
    ):
        super().__init__(coord_dim, coord_axes, fixed_axes, fixed_axes_values)

    def metric_tensor(
        self, n_points: int, coords: torch.Tensor, device: torch.device | None = None
    ) -> torch.Tensor:
        """
         Return the metric tensor(s) for spherical polar coordinates.

        Parameters
         ----------
         n_points : int
             Number of points for which to generate metric tensors.
         coords : torch.Tensor
             Coordinates of the points. Shape: (n_points, coord_dim).

         Returns
         -------
         torch.Tensor
             Metric tensors of shape (n_points, coord_dim, coord_dim).
        """
        # Validate coords tensor
        self._validate_coordinates(coords)

        # Get quantities needed in metric tensor
        r = get_required_or_fixed_coord(
            "r",
            coords,
            self.axes_labels_selected,
            self.fixed_axes,
            self.fixed_axes_values,
        )
        theta = get_required_or_fixed_coord(
            "theta",
            coords,
            self.axes_labels_selected,
            self.fixed_axes,
            self.fixed_axes_values,
        )

        g = torch.eye(self.metadata.max_coord_dim).to(device)
        G = g.unsqueeze(0).repeat(
            n_points, 1, 1
        )  # Shape: (n_points, coord_dim, coord_dim)
        G[:, 2, 2] = r**2  # theta theta components
        G[:, 3, 3] = r**2 * torch.sin(theta) ** 2  # phi phi components

        # Only use rows and columns of axes included in coords
        mask_keep = [
            i for i in range(self.metadata.max_coord_dim) if i in self.indices_selected
        ]
        G = G[:, mask_keep, mask_keep]

        # Validate output
        validate_pytorch_tensor(G, (n_points, self.coord_dim, self.coord_dim))

        return G

    def _spherical_validate_selected_axes(self) -> None:
        """
        Validate that the selected axes are compatible coordinate system.

        In cases where two combinations of selected coords behave the same, we raise an
        error in one of them to enforce consistent coordinate system labelling i.e.
        (t, theta) and (r, theta).

        Raises
        ------
        ValueError
            If selected axes are not compatible.
        """

        # Extract metadata
        name = self.metadata.name
        axes_labels = self.metadata.axes_labels

        # valid combinations
        valid_axes = [axes_labels[: self.coord_dim]]
        if self.coord_dim == 2:
            valid_axes.append(("t", "theta"))
            valid_axes.append(("t", "phi"))
        elif self.coord_dim == 3:
            valid_axes.append(("t", "theta", "phi"))

        if self.axes_labels_selected not in valid_axes:
            raise ValueError(
                f"To enforce consistent coordinate system labelling, use "
                f"axes {valid_axes} instead of {self.axes_labels_selected} "
                f"when using {name} with coord_dim={self.coord_dim}"
            )


# Mapping from coordinate system labels to classes
COORD_SYSTEM_REGISTRY: dict[
    CoordSystemLabels, Callable[..., CoordSystem] | type[CoordSystem]
] = {
    CoordSystemLabels.CARTESIAN: CartesianCoords,
    CoordSystemLabels.CYLINDRICAL: CylindricalCoords,
    CoordSystemLabels.SPHERICAL: SphericalCoords,
}
