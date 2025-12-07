from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
import torch

from ..checks.validate_data import validate_pytorch_tensor
from .geometry_utils import select_coord_system_axes

if TYPE_CHECKING:
    pass


class CoordSystemLabels(str, Enum):
    """Enumeration of coordinate system labels."""

    CARTESIAN = "cartesian"
    CYLINDRICAL = "cylindrical"
    SPHERICAL = "spherical"


@dataclass(frozen=True)
class CoordSystemMetadata:
    """Dataclass for coordinate system metadata."""

    name: str
    is_orthogonal: bool
    axes_labels: tuple[str, ...]
    required_or_fixed_axes: tuple[str, ...]
    required_fields: tuple[str, ...]
    required_consts: tuple[str, ...]
    axes_bounds: tuple[tuple[float, float], ...]
    internal_coords: CoordSystemLabels
    max_dim_geometry: int
    coord_redirections: dict[tuple[int, tuple[str, ...]], CoordSystemLabels] | None


class CoordSystem(ABC):
    """Base class for coordinate systems."""

    # Placeholder for subclass class attributes
    metadata: CoordSystemMetadata

    def __init__(
        self,
        dim_geometry: int,
        coord_axes: tuple[str, ...],
        fixed_axes: tuple[str, ...] | None = None,
        fixed_axes_values: tuple[float, ...] | None = None,
    ):
        self.dim_geometry = dim_geometry

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
            self.indices_selected,
        ) = select_coord_system_axes(
            self.metadata.axes_labels,
            self.metadata.axes_bounds,
            coord_axes,
        )
        self.fixed_axes = fixed_axes
        self.fixed_axes_values = fixed_axes_values

        # Ensure coordinate system is valid
        self._validate_selected_dimensions()
        self._validate_selected_axes()
        self._validate_required_or_fixed_axes()
        self._validate_selected_bounds()

    @abstractmethod
    def metric_tensor(
        self,
        n_points: int,
        coords_geometry: list[torch.Tensor],
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Subclasses must implement this method to return the metric tensor(s)."""
        pass

    @abstractmethod
    def map_coord_dict_to_internal_coords(
        self, coords: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Subclasses must implement this method to map coords to internal coords
        used for a given coordinate system."""
        pass

    def contravariant_grad(
        self,
        f: torch.Tensor,
        coords_geometry: list[torch.Tensor],
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """
        Compute the contravarient gradient of a function in a given coordinate system.

        contravariant_grad(f) = g^{ij} * partial_j f

        Parameters
        ----------
        f : torch.Tensor
            Function to be taken gradient of. Shape: (n_points,)
        coords_geometry : list[torch.Tensor]
            A set of points and their geometric coordinates.
            List length: dim_geometry, tensor lengths: n_points

        Returns
        -------
        torch.Tensor
            Contravarient gradient of the function: (n_points, dim_geometry)
        """
        # Coordinate system dimensions
        dim_geometry = self.dim_geometry

        # Validate input tensors
        n_points = f.shape[0]
        validate_pytorch_tensor(f, (n_points,))
        self._validate_coordinates(coords_geometry)

        # Tensor of metric tensors for each point
        G = self.metric_tensor(
            n_points, coords_geometry, device
        )  # Shape: (n_points, dim_geometry, dim_geometry)
        validate_pytorch_tensor(G, (n_points, dim_geometry, dim_geometry))

        # Inverse metric tensor
        G_inv = torch.linalg.inv(G)  # Shape: (n_points, dim_geometry, dim_geometry)
        validate_pytorch_tensor(G_inv, (n_points, dim_geometry, dim_geometry))

        derivative_f: list[torch.Tensor] = []
        for coord in coords_geometry:
            # Calculate derivative wrt a coord using autograd
            derivative_f_coord = torch.autograd.grad(
                f,
                coord,
                grad_outputs=torch.ones_like(f),
                create_graph=True,
                retain_graph=True,
            )[0]

            # Set derivatives to zero in None (Case where f is indep of coord)
            if derivative_f is None:
                derivative_f_coord = torch.zeros_like(coord)

            # Append to list
            derivative_f.append(derivative_f_coord)

        derivative_f_tensor = torch.stack(derivative_f, dim=1)
        validate_pytorch_tensor(derivative_f_tensor, (n_points, dim_geometry))

        # Batch matrix multiplication:
        # (n_points, dim_geometry, dim_geometry) @ (n_points, dim_geometry, 1) ->
        # (n_points, dim_geometry, 1)
        # Squeeze the result to get the final (n_points, dim_geometry) vector
        contra_grad_f = torch.bmm(G_inv, derivative_f_tensor.unsqueeze(2)).squeeze(2)
        validate_pytorch_tensor(contra_grad_f, (n_points, dim_geometry))

        return contra_grad_f

    def get_scale_factors(
        self,
        coords_geometry: list[torch.Tensor],
        n_points: int,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """
        Compute coordinate scale factors in an orthogonal coordinate system.

        NOTE: only works for orthogonal coord systems

        Parameters
        ----------
        coords_geometry : list[torch.Tensor]
            A set of points and their geometric coordinates.
            List length: dim_geometry, tensor lengths: n_points
        n_points : int
            Number of points that each tensor should have elements for.

        Returns
        -------
        torch.Tensor
            Scale factors for each coordinate axis. Shape: (n_points, dim_geometry)
        """
        # Check that coord system is orthogonal
        if not self.metadata.is_orthogonal:
            raise ValueError(
                "metric_scale_factors function can only be used for orthogonal "
                "coordinate systems."
            )

        # Coordinate system dimensions
        dim_geometry = self.dim_geometry

        # Validate input tensors
        self._validate_coordinates(coords_geometry)

        # Tensor of metric tensors for each point
        G = self.metric_tensor(
            n_points, coords_geometry, device
        )  # Shape: (n_points, dim_geometry, dim_geometry)
        validate_pytorch_tensor(G, (n_points, dim_geometry, dim_geometry))

        # Get diagonal elements for each point
        G_diag = torch.diagonal(G, dim1=1, dim2=2)  # Shape: (n_points, dim_geometry)
        scale_factors = torch.sqrt(torch.abs(G_diag))

        return scale_factors

    def grad(
        self,
        f: torch.Tensor,
        coords_geometry: list[torch.Tensor],
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """
        Compute the gradient of a function in a given coordinate system.

        Parameters
        ----------
        f : torch.Tensor
            Function to be taken gradient of. Shape: (n_points,)
        coords_geometry : list[torch.Tensor]
            A set of points and their geometric coordinates.
            List length: dim_geometry, tensor lengths: n_points

        Returns
        -------
        torch.Tensor
            Gradient of the function: (n_points, dim_geometry)
        """

        # Get contravariant grad of f
        contra_grad_f = self.contravariant_grad(f, coords_geometry, device)

        # Get scale factors (assumes the coordinate system is orthogonal)
        n_points = f.shape[0]
        scale_factors = self.get_scale_factors(coords_geometry, n_points, device)

        return scale_factors * contra_grad_f

    def laplacian(
        self,
        f: torch.Tensor,
        coords_geometry: list[torch.Tensor],
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """
        Compute the laplacian operator of a function in a given coordinate system.

        Parameters
        ----------
        f : torch.Tensor
            Function to be taken gradient of. Shape: (n_points,)
        coords_geometry : list[torch.Tensor]
            A set of points and their geometric coordinates.
            List length: dim_geometry, tensor lengths: n_points

        Returns
        -------
        torch.Tensor
            Laplacian of the function: (n_points,)
        """
        # Coordinate system dimensions
        dim_geometry = self.dim_geometry

        # Validate input tensors
        n_points = f.shape[0]
        validate_pytorch_tensor(f, (n_points,))
        self._validate_coordinates(coords_geometry)

        # Tensor of metric tensors for each point
        G = self.metric_tensor(
            n_points, coords_geometry, device
        )  # Shape: (n_points, dim_geometry, dim_geometry)
        validate_pytorch_tensor(G, (n_points, dim_geometry, dim_geometry))

        # Inverse metric tensor
        G_inv = torch.linalg.inv(G)  # Shape: (n_points, dim_geometry, dim_geometry)
        validate_pytorch_tensor(G_inv, (n_points, dim_geometry, dim_geometry))

        # Metric determinant
        det_G = torch.det(G)  # Shape: (n_points,)
        sqrt_det_G = torch.sqrt(torch.abs(det_G))  # Shape: (n_points,)

        # Get contravariant grad of f
        contra_grad_f = self.contravariant_grad(f, coords_geometry, device)
        inner_terms = sqrt_det_G.unsqueeze(1) * contra_grad_f

        # Now take divergence
        divergence_accum = torch.zeros(n_points, device=device)
        for i, coord in enumerate(coords_geometry):
            inner_term = inner_terms[:, i]  # Shape (n_points,)

            # Differentiate this component w.r.t each coordinate
            derivative_inner_coord = torch.autograd.grad(
                inner_term,
                coord,
                grad_outputs=torch.ones_like(inner_term),
                create_graph=True,
                retain_graph=True,
            )[0]

            # Set derivatives to zero in None (Case where f is indep of coord)
            if derivative_inner_coord is None:
                derivative_inner_coord = torch.zeros_like(coord)

            # Add to cumulative sum
            divergence_accum += derivative_inner_coord

        # Divide by sqrt of det G
        laplacian_f = divergence_accum / (sqrt_det_G)
        validate_pytorch_tensor(laplacian_f, (n_points,))

        return laplacian_f

    def _get_coord(
        self,
        axis_label: str,
        coords_geometry: list[torch.Tensor],
        default: float | None = None,
    ) -> torch.Tensor:
        if axis_label in self.axes_labels_selected:
            idx = self.axes_labels_selected.index(axis_label)
            return coords_geometry[idx]
        elif (
            self.fixed_axes is not None
            and self.fixed_axes_values is not None
            and axis_label in self.fixed_axes
        ):
            idx = self.fixed_axes.index(axis_label)
            return torch.tensor(self.fixed_axes_values[idx]).float()
        elif default is not None:
            return torch.tensor(default).float()
        else:
            raise ValueError(f"Missing required axis: {axis_label}")

    def _validate_selected_dimensions(self) -> None:
        """
        Validate that the coordinate system supports the given number of dimensions.

        Raises
        ------
        TypeError
            If dim_geometry is not an integer.
        ValueError
            If coordinate system is incompatible with the given number of dimensions.
        """
        # Extract metadata
        name = self.metadata.name
        max_dim_geometry = self.metadata.max_dim_geometry

        # Coordinate system dimensions
        dim_geometry = self.dim_geometry

        # Check type of dim_geometry
        if not isinstance(dim_geometry, int):
            raise TypeError(
                "Expected dim_geometry to be an integer, "
                f"got {type(dim_geometry).__name__}"
            )

        # Check value of dim_geometry
        if dim_geometry < 2:
            raise ValueError(f"dim_geometry must be >= 2, got {dim_geometry}")
        if dim_geometry > max_dim_geometry:
            raise ValueError(
                f"{name} only implemented for dim_geometry<={max_dim_geometry}, "
                f"got dim_geometry={dim_geometry}"
            )

    def _validate_selected_axes(self) -> None:
        """
        Validate that the selected axes are compatible coordinate system.

        In cases where two coord systems behave the same, we raise an error in one of
        them to enforce consistent coordinate system labelling i.e. (r, z) cylindrical
        polars redirects you to (x, y) Cartesian.

        Raises
        ------
        ValueError
            If selected axes are not compatible.
        """

        # Extract metadata
        name = self.metadata.name
        coord_redirections = self.metadata.coord_redirections

        # Coordinate system dimensions
        dim_geometry = self.dim_geometry
        axes_labels_selected = self.axes_labels_selected

        # Error to redirect user to alternative coordinate system if necessary
        if (
            coord_redirections
            and (dim_geometry, axes_labels_selected) in coord_redirections
        ):
            alt_coord_system_label = coord_redirections[
                (dim_geometry, axes_labels_selected)
            ]
            alt_coord_system_class = COORD_SYSTEM_REGISTRY[alt_coord_system_label]
            alt_coord_system_name = alt_coord_system_class.metadata.name

            raise ValueError(
                f"To enforce consistent coordinate system labelling, use "
                f"{alt_coord_system_name} instead of {name} when using "
                f"dim_geometry={dim_geometry} with axes {axes_labels_selected}"
            )

    def _validate_required_or_fixed_axes(self) -> None:
        """
        Validate axes that are required either as selected axis or a fixed value.

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

    def _validate_selected_bounds(self) -> None:
        """
        Validate the bounds of axes in the coordinate system.

        Raises
        ------
        ValueError
            If axes bounds are inconsistent.
        """
        # Extract metadata
        dim_geometry = self.dim_geometry
        axes_labels = self.axes_labels_selected
        axes_bounds = self.axes_bounds_selected

        # Check that axes_bounds has correct shape
        if np.shape(axes_bounds) != (dim_geometry, 2):
            raise ValueError(
                f"axes_bounds must be of shape ({dim_geometry}, 2) , got "
                f"{np.shape(axes_bounds)}."
            )

        # Separate lower and upper bounds
        lower_bounds, upper_bounds = zip(*axes_bounds, strict=True)

        for i in range(dim_geometry):
            axis_label = axes_labels[i]
            lb = lower_bounds[i]
            ub = upper_bounds[i]

            # Check that lower bound is less than upper bound
            if lower_bounds[i] >= upper_bounds[i]:
                raise ValueError(
                    f"Coordinate system lower bound must be lower than upper bound, "
                    f"got {lb = } and {ub = } for {axis_label}."
                )

    def _validate_coordinates(
        self, coords_geometry: list[torch.Tensor], n_points: int | None = None
    ) -> None:
        """
        Validate that a set of provided geometric coordinates are in the right form.

        Parameters
        ----------
        coords_geometry : list[torch.Tensor]
            Coordinates to validate. List length: dim_geometry, tensor lengths: n_points
        n_points : int
            Number of points that each tensor should have elements for.

        Raises
        ------
        ValueError
            If coordinates have incorrect form.
        """
        # Coordinate system dimensions
        dim_geometry = self.dim_geometry

        # Validate coords_geometry tensor type and shape
        if len(coords_geometry) != dim_geometry:
            raise ValueError(
                f"Expected coords_geometry to have length {dim_geometry}, "
                f"got {len(coords_geometry)}"
            )

        if n_points is not None:
            for i in range(dim_geometry):
                validate_pytorch_tensor(coords_geometry[i], (n_points,))


class CartesianCoords(CoordSystem):
    """Cartesian coordinate system: x, y, z."""

    # Metadata as class attributes
    metadata = CoordSystemMetadata(
        name="Cartesian Coordinates",
        is_orthogonal=True,
        axes_labels=("x", "y", "z"),
        required_or_fixed_axes=(),
        required_fields=(),
        required_consts=(),
        axes_bounds=((-np.inf, np.inf), (-np.inf, np.inf), (-np.inf, np.inf)),
        internal_coords=CoordSystemLabels.CARTESIAN,
        max_dim_geometry=3,
        coord_redirections=None,
    )

    def __init__(
        self,
        dim_geometry: int,
        coord_axes: tuple[str, ...],
        fixed_axes: tuple[str, ...] | None = None,
        fixed_axes_values: tuple[float, ...] | None = None,
    ):
        super().__init__(dim_geometry, coord_axes, fixed_axes, fixed_axes_values)
        self._cartesian_validate_selected_axes()

    def metric_tensor(
        self,
        n_points: int,
        coords_geometry: list[torch.Tensor] | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """
         Return the metric tensor(s) for Cartesian coordinates.

        Parameters
         ----------
         n_points : int
             Number of points for which to generate metric tensors.
         coords_geometry : list[torch.Tensor]
             Coordinates of the points. Unused for Cartesian coordinates.

         Returns
         -------
         torch.Tensor
             Metric tensors of shape (n_points, dim_geometry, dim_geometry).
        """
        g = torch.eye(self.metadata.max_dim_geometry).to(device)
        G = g.unsqueeze(0).repeat(
            n_points, 1, 1
        )  # Shape: (n_points, dim_geometry, dim_geometry)

        # Only use rows and columns of axes included in coords_geometry
        mask_keep = [
            i
            for i in range(self.metadata.max_dim_geometry)
            if i in self.indices_selected
        ]
        G = G[:, mask_keep, :][:, :, mask_keep]

        # Validate output
        validate_pytorch_tensor(G, (n_points, self.dim_geometry, self.dim_geometry))

        return G

    def map_coord_dict_to_internal_coords(
        self, coords: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        return coords

    def _cartesian_validate_selected_axes(self) -> None:
        """
        Validate that the selected axes are compatible coordinate system.

        In cases where two combinations of selected coords_geometry behave the same, we
        raise an error in one of them to enforce consistent coordinate system labelling
        i.e. (x, y) and (y, z).

        Raises
        ------
        ValueError
            If selected axes are not compatible.
        """

        # Extract metadata
        name = self.metadata.name
        axes_labels = self.metadata.axes_labels

        # Only valid combinations
        valid_axes = [axes_labels[: self.dim_geometry]]

        if self.axes_labels_selected not in valid_axes:
            raise ValueError(
                f"To enforce consistent coordinate system labelling, use "
                f"axes {valid_axes} instead of {self.axes_labels_selected} "
                f"when using {name} with dim_geometry={self.dim_geometry}"
            )


class CylindricalCoords(CoordSystem):
    """Cylindrical polar coordinate system: r, theta, z."""

    # Metadata as class attributes
    metadata = CoordSystemMetadata(
        name="Cylindrical Polar Coordinates",
        is_orthogonal=True,
        axes_labels=("r", "theta", "z"),
        required_or_fixed_axes=("r",),
        required_fields=(),
        required_consts=(),
        axes_bounds=((0, np.inf), (0, 2 * np.pi), (-np.inf, np.inf)),
        internal_coords=CoordSystemLabels.CARTESIAN,
        max_dim_geometry=3,
        coord_redirections={
            (1, ("r",)): CoordSystemLabels.CARTESIAN,
            (1, ("z",)): CoordSystemLabels.CARTESIAN,
            (2, ("r", "z")): CoordSystemLabels.CARTESIAN,
        },
    )

    def __init__(
        self,
        dim_geometry: int,
        coord_axes: tuple[str, ...],
        fixed_axes: tuple[str, ...] | None = None,
        fixed_axes_values: tuple[float, ...] | None = None,
    ):
        super().__init__(dim_geometry, coord_axes, fixed_axes, fixed_axes_values)

    def metric_tensor(
        self,
        n_points: int,
        coords_geometry: list[torch.Tensor],
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """
         Return the metric tensor(s) for cylindrical polar coordinates.

        Parameters
         ----------
         n_points : int
             Number of points for which to generate metric tensors.
         coords_geometry : list[torch.Tensor]
             Coordinates of the points. Shape: (n_points, dim_geometry).

         Returns
         -------
         torch.Tensor
             Metric tensors of shape (n_points, dim_geometry, dim_geometry).
        """
        # Validate coords_geometry
        self._validate_coordinates(coords_geometry)

        # Get quantities needed in metric tensor
        r = self._get_coord(
            "r",
            coords_geometry,
        )

        g = torch.eye(self.metadata.max_dim_geometry).to(device)
        G = g.unsqueeze(0).repeat(
            n_points, 1, 1
        )  # Shape: (n_points, dim_geometry, dim_geometry)
        G[:, 1, 1] = r  # theta theta components

        # Only use rows and columns of axes included in coords_geometry
        mask_keep = [
            i
            for i in range(self.metadata.max_dim_geometry)
            if i in self.indices_selected
        ]
        G = G[:, mask_keep, mask_keep]

        # Validate output
        validate_pytorch_tensor(G, (n_points, self.dim_geometry, self.dim_geometry))

        return G

    def map_coord_dict_to_internal_coords(
        self, coords: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        # dict of internal coords
        internal_coords = coords.copy()

        # Get geometric coordinates in coord system basis
        coords_geometry = []
        for coord_label in self.axes_labels_selected:
            coords_geometry.append(coords[coord_label])

            # Delete geometric coords from dict
            del internal_coords[coord_label]

        # r must always be available in cylindrical polars
        r = self._get_coord(
            "r",
            coords_geometry,
        )

        # theta
        if "theta" in self.axes_labels_selected:
            theta = self._get_coord("theta", coords_geometry)
            internal_coords["x"] = r * torch.cos(theta)
            internal_coords["y"] = r * torch.sin(theta)
        else:
            # If no theta is given
            internal_coords["x"] = r

        # z
        if "z" in self.axes_labels_selected:
            z = self._get_coord("z", coords_geometry)

            internal_coords["z"] = z

        return internal_coords


class SphericalCoords(CoordSystem):
    """Spherical polar coordinate system: r, theta, phi."""

    # Metadata as class attributes
    metadata = CoordSystemMetadata(
        name="Spherical Polar Coordinates",
        is_orthogonal=True,
        axes_labels=("r", "theta", "phi"),
        required_or_fixed_axes=("r", "theta"),
        required_fields=(),
        required_consts=(),
        axes_bounds=((0, np.inf), (0, np.pi), (0, 2 * np.pi)),
        internal_coords=CoordSystemLabels.CARTESIAN,
        max_dim_geometry=3,
        coord_redirections={
            (1, ("r",)): CoordSystemLabels.CARTESIAN,
            (2, ("r", "phi")): CoordSystemLabels.CYLINDRICAL,
        },
    )

    def __init__(
        self,
        dim_geometry: int,
        coord_axes: tuple[str, ...],
        fixed_axes: tuple[str, ...] | None = None,
        fixed_axes_values: tuple[float, ...] | None = None,
    ):
        super().__init__(dim_geometry, coord_axes, fixed_axes, fixed_axes_values)

    def metric_tensor(
        self,
        n_points: int,
        coords_geometry: list[torch.Tensor],
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """
         Return the metric tensor(s) for spherical polar coordinates.

        Parameters
         ----------
         n_points : int
             Number of points for which to generate metric tensors.
         coords_geometry : list[torch.Tensor]
             Coordinates of the points. Shape: (n_points, dim_geometry).

         Returns
         -------
         torch.Tensor
             Metric tensors of shape (n_points, dim_geometry, dim_geometry).
        """
        # Validate coords_geometry
        self._validate_coordinates(coords_geometry)

        # Get quantities needed in metric tensor
        r = self._get_coord(
            "r",
            coords_geometry,
        )
        theta = self._get_coord(
            "theta",
            coords_geometry,
        )

        g = torch.eye(self.metadata.max_dim_geometry).to(device)
        G = g.unsqueeze(0).repeat(
            n_points, 1, 1
        )  # Shape: (n_points, dim_geometry, dim_geometry)
        G[:, 1, 1] = r**2  # theta theta components
        G[:, 2, 2] = r**2 * torch.sin(theta) ** 2  # phi phi components

        # Only use rows and columns of axes included in coords_geometry
        mask_keep = [
            i
            for i in range(self.metadata.max_dim_geometry)
            if i in self.indices_selected
        ]
        G = G[:, mask_keep, mask_keep]

        # Validate output
        validate_pytorch_tensor(G, (n_points, self.dim_geometry, self.dim_geometry))

        return G

    def map_coord_dict_to_internal_coords(
        self, coords: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        # dict of internal coords
        internal_coords = coords.copy()

        # Get geometric coordinates in coord system basis
        coords_geometry = []
        for coord_label in self.axes_labels_selected:
            coords_geometry.append(coords[coord_label])

            # Delete geometric coords from dict
            del internal_coords[coord_label]

        # r must always be available in spherical polars
        r = self._get_coord(
            "r",
            coords_geometry,
        )

        # theta must always be available in spherical polars
        theta = self._get_coord(
            "theta",
            coords_geometry,
        )

        if "phi" in self.axes_labels_selected:
            phi = self._get_coord(
                "phi",
                coords_geometry,
            )
            internal_coords["x"] = r * torch.sin(theta) * torch.cos(phi)
            internal_coords["y"] = r * torch.sin(theta) * torch.sin(phi)
            internal_coords["z"] = r * torch.cos(theta)

        else:
            internal_coords["x"] = r * torch.sin(theta)
            internal_coords["z"] = r * torch.cos(theta)

        return internal_coords


# Mapping from coordinate system labels to classes
COORD_SYSTEM_REGISTRY: dict[CoordSystemLabels, type[CoordSystem]] = {
    CoordSystemLabels.CARTESIAN: CartesianCoords,
    CoordSystemLabels.CYLINDRICAL: CylindricalCoords,
    CoordSystemLabels.SPHERICAL: SphericalCoords,
}
