from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

import numpy as np
import torch

from ..checks.validate_data import validate_pytorch_tensor
from .geometry_utils import select_coord_system_axes
from .metric_tensor import DiagonalMetricTensor, IdentityMetricTensor, MetricTensor


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
        fixed_axes: tuple[str, ...],
        fixed_axes_values: tuple[float, ...],
        device: torch.device,
    ):
        self.dim_geometry = dim_geometry
        self.device = device

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
        geometric_coords: torch.Tensor,
    ) -> MetricTensor:
        """Subclasses must implement this method to return the metric tensor(s)."""
        pass

    @abstractmethod
    def map_geometric_to_internal(
        self, geometric_coords: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Subclasses must implement this method to map geometric coords to internal
        coords."""
        pass

    def _get_coord(
        self,
        axis_label: str,
        geometric_coords: torch.Tensor,
        default: float | None = None,
    ) -> torch.Tensor:
        if axis_label in self.axes_labels_selected:
            idx = self.axes_labels_selected.index(axis_label)
            return geometric_coords[:, idx]
        elif (
            self.fixed_axes is not None
            and self.fixed_axes_values is not None
            and axis_label in self.fixed_axes
        ):
            idx = self.fixed_axes.index(axis_label)
            return torch.tensor(self.fixed_axes_values[idx])
        elif default is not None:
            return torch.tensor(default)
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
        if dim_geometry < 1:
            raise ValueError(f"dim_geometry must be >= 1, got {dim_geometry}")
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
        fixed_axes: tuple[str, ...],
        fixed_axes_values: tuple[float, ...],
        device: torch.device,
    ):
        super().__init__(
            dim_geometry, coord_axes, fixed_axes, fixed_axes_values, device
        )
        self._cartesian_validate_selected_axes()

    def metric_tensor(
        self,
        n_points: int,
        geometric_coords: torch.Tensor | None = None,
    ) -> MetricTensor:
        return IdentityMetricTensor(n_points, self.device)

    def map_geometric_to_internal(
        self, geometric_coords: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        # For Cartesian, internal coords are the same as geometric coords
        return {
            label: geometric_coords[:, i]
            for i, label in enumerate(self.axes_labels_selected)
        }

    def _cartesian_validate_selected_axes(self) -> None:
        """
        Validate that the selected axes are compatible coordinate system.

        In cases where two combinations of selected geometric_coords behave the same, we
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
        fixed_axes: tuple[str, ...],
        fixed_axes_values: tuple[float, ...],
        device: torch.device,
    ):
        super().__init__(
            dim_geometry, coord_axes, fixed_axes, fixed_axes_values, device
        )

    def metric_tensor(
        self,
        n_points: int,
        geometric_coords: torch.Tensor,
    ) -> MetricTensor:
        # Validate geometric_coords
        validate_pytorch_tensor(geometric_coords, (n_points, self.dim_geometry))

        # Get quantities needed in metric tensor
        r = self._get_coord(
            "r",
            geometric_coords,
        )

        diag = torch.ones(
            (n_points, self.metadata.max_dim_geometry), device=self.device
        )
        diag[:, 1] = r**2  # theta-theta component

        # Only use rows and columns of axes included in geometric_coords
        mask_keep = [
            i
            for i in range(self.metadata.max_dim_geometry)
            if i in self.indices_selected
        ]
        diag = diag[:, mask_keep]
        return DiagonalMetricTensor(diag)

    def map_geometric_to_internal(
        self, geometric_coords: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        # r must always be available in cylindrical polars
        r = self._get_coord(
            "r",
            geometric_coords,
        )

        # Transform
        internal_geometric_coords: dict[str, torch.Tensor] = {}

        # theta
        if "theta" in self.axes_labels_selected:
            theta = self._get_coord("theta", geometric_coords)
            internal_geometric_coords["x"] = r * torch.cos(theta)
            internal_geometric_coords["y"] = r * torch.sin(theta)
        else:
            # If no theta is given
            internal_geometric_coords["x"] = r

        # z
        if "z" in self.axes_labels_selected:
            z = self._get_coord("z", geometric_coords)

            internal_geometric_coords["z"] = z

        return internal_geometric_coords


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
        fixed_axes: tuple[str, ...],
        fixed_axes_values: tuple[float, ...],
        device: torch.device,
    ):
        super().__init__(
            dim_geometry, coord_axes, fixed_axes, fixed_axes_values, device
        )

    def metric_tensor(
        self,
        n_points: int,
        geometric_coords: torch.Tensor,
    ) -> MetricTensor:
        # Validate geometric_coords
        validate_pytorch_tensor(geometric_coords, (n_points, self.dim_geometry))

        # Get quantities needed in metric tensor
        r = self._get_coord(
            "r",
            geometric_coords,
        )
        theta = self._get_coord(
            "theta",
            geometric_coords,
        )

        diag = torch.ones(
            (n_points, self.metadata.max_dim_geometry), device=self.device
        )
        diag[:, 1] = r**2  # theta-theta component
        diag[:, 2] = r**2 * torch.sin(theta) ** 2  # phi-phi component

        # Only use rows and columns of axes included in geometric_coords
        mask_keep = [
            i
            for i in range(self.metadata.max_dim_geometry)
            if i in self.indices_selected
        ]
        diag = diag[:, mask_keep]
        return DiagonalMetricTensor(diag)

    def map_geometric_to_internal(
        self, geometric_coords: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        # r must always be available in spherical polars
        r = self._get_coord(
            "r",
            geometric_coords,
        )

        # theta must always be available in spherical polars
        theta = self._get_coord(
            "theta",
            geometric_coords,
        )

        # Transform
        internal_geometric_coords: dict[str, torch.Tensor] = {}
        if "phi" in self.axes_labels_selected:
            phi = self._get_coord(
                "phi",
                geometric_coords,
            )
            internal_geometric_coords["x"] = r * torch.sin(theta) * torch.cos(phi)
            internal_geometric_coords["y"] = r * torch.sin(theta) * torch.sin(phi)
            internal_geometric_coords["z"] = r * torch.cos(theta)

        else:
            internal_geometric_coords["x"] = r * torch.sin(theta)
            internal_geometric_coords["z"] = r * torch.cos(theta)

        return internal_geometric_coords


# Mapping from coordinate system labels to classes
COORD_SYSTEM_REGISTRY: dict[CoordSystemLabels, type[CoordSystem]] = {
    CoordSystemLabels.CARTESIAN: CartesianCoords,
    CoordSystemLabels.CYLINDRICAL: CylindricalCoords,
    CoordSystemLabels.SPHERICAL: SphericalCoords,
}
