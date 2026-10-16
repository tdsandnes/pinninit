from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from itertools import product
from typing import TYPE_CHECKING

import numpy as np
import torch

from .geometry_utils import trig_bounds
from .metric_tensor import DiagonalMetricTensor, IdentityMetricTensor

if TYPE_CHECKING:
    from .metric_tensor import MetricTensor


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
    axes_bounds: dict[str, tuple[float, float]]
    max_dim_geometric: int
    internal_coord_system: CoordSystemLabels
    required_axes: set[str]
    coord_system_redirections: (
        dict[frozenset[str], tuple[CoordSystemLabels, set[str]]] | None
    )


class CoordSystem(ABC):
    """
    Base class for coordinate systems.

    Parameters
    ----------
    coord_axes:
        Labels of the axes the coordinates vary along, e.g. {"r", "theta"}.
        Their number is the number of geometric dimensions.
    device:
        Device of the tensors the coord system makes.
    fixed_axes:
        Values of axes held fixed, e.g. {"r": 1.0} for the surface of a
        cylinder.
    """

    # Placeholder for subclass class attributes
    metadata: CoordSystemMetadata

    def __init__(
        self,
        coord_axes: set[str],
        device: torch.device,
        fixed_axes: dict[str, float] | None = None,
    ) -> None:
        fixed_axes = fixed_axes or {}

        # Validate input
        self._validate_selected_dimensions(len(coord_axes))
        self._validate_selected_axes(coord_axes, fixed_axes)

        # Store values
        self.dim_geometric = len(coord_axes)
        self.device = device

        # Sort axes based on their order in self.metadata.axes_labels
        self.indices_selected = tuple(
            i
            for i, label in enumerate(self.metadata.axes_labels)
            if label in coord_axes
        )
        self.axes_labels_selected = tuple(
            self.metadata.axes_labels[i] for i in self.indices_selected
        )
        self.axes_bounds_selected = {
            label: self.metadata.axes_bounds[label]
            for label in self.axes_labels_selected
        }
        self.fixed_axes = {
            label: fixed_axes[label]
            for label in self.metadata.axes_labels
            if label in fixed_axes
        }

    @abstractmethod
    def metric_tensor(
        self,
        n_points: int,
        geometric_coords: dict[str, torch.Tensor],
    ) -> MetricTensor:
        pass

    @property
    @abstractmethod
    def internal_geometric_labels(self) -> tuple[str, ...]:
        pass

    @abstractmethod
    def map_geometric_to_internal_geometric(
        self, geometric_coords: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        pass

    @abstractmethod
    def internal_bounds_from_geometric(
        self,
        geometric_lb: dict[str, torch.Tensor],
        geometric_ub: dict[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        pass

    @abstractmethod
    def geometric_to_cartesian_rotation(
        self, n_points: int, geometric_coords: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        pass

    def transform_geometric_to_cartesian(
        self, vector: torch.Tensor, geometric_coords: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Geometric vector components to Cartesian components."""
        rotation = self.geometric_to_cartesian_rotation(
            vector.shape[0], geometric_coords
        )
        return torch.einsum("nij,nj->ni", rotation, vector)

    def transform_cartesian_to_geometric(
        self, vector: torch.Tensor, geometric_coords: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Cartesian vector components to geometric components."""
        rotation = self.geometric_to_cartesian_rotation(
            vector.shape[0], geometric_coords
        )
        return torch.einsum("nij,ni->nj", rotation, vector)

    def _get_coord(
        self,
        axis_label: str,
        geometric_coords: dict[str, torch.Tensor],
        default: float | None = None,
    ) -> torch.Tensor:
        if axis_label in self.axes_labels_selected:
            return geometric_coords[axis_label]

        if axis_label in self.fixed_axes:
            value = self.fixed_axes[axis_label]
        elif default is not None:
            value = default
        else:
            raise ValueError(f"Missing required axis: {axis_label}")
        return torch.tensor(
            value, dtype=self._dtype(geometric_coords), device=self.device
        )

    def _dtype(self, geometric_coords: dict[str, torch.Tensor]) -> torch.dtype:
        """The dtype of the given coords, so that results keep their precision."""
        return next(iter(geometric_coords.values())).dtype

    def _validate_selected_dimensions(self, dim_geometric: int) -> None:
        # The number of geometric dimensions is the number of coord axes
        if dim_geometric < 1:
            raise ValueError("coord_axes must not be empty")
        if dim_geometric > self.metadata.max_dim_geometric:
            raise ValueError(
                f"{self.metadata.name} only implemented for up to "
                f"{self.metadata.max_dim_geometric} coord axes, got {dim_geometric}"
            )

    def _validate_selected_axes(
        self, coord_axes: set[str], fixed_axes: dict[str, float]
    ) -> None:
        # In cases where two coord systems behave the same, we raise an error
        # to enforce consistent coordinate system labelling i.e. (r, z) cylindrical
        # polars redirects you to (x, y) Cartesian.
        redirections = self.metadata.coord_system_redirections
        if redirections is not None and frozenset(coord_axes) in redirections:
            alt_coord_system_label = redirections[frozenset(coord_axes)][0]
            alt_coord_system_axes = redirections[frozenset(coord_axes)][1]
            alt_coord_system_name = COORD_SYSTEM_REGISTRY[
                alt_coord_system_label
            ].metadata.name

            raise ValueError(
                f"To enforce consistent coordinate system labelling, use "
                f"{alt_coord_system_name} with axes {alt_coord_system_axes}, instead "
                f"of {self.metadata.name} with axes {coord_axes}"
            )

        # If a coordinate system has any required axes, make sure these have been given
        for axis_label in self.metadata.required_axes:
            if axis_label not in coord_axes and axis_label not in fixed_axes:
                raise ValueError(f"Missing required axis: {axis_label}")

        # Both coord_axes and fixed_axes must be in the coordinate system's axes_labels
        valid_axes = set(self.metadata.axes_labels)
        invalid_coord_axes = coord_axes - valid_axes
        if invalid_coord_axes:
            raise ValueError(
                f"Invalid axes for {self.metadata.name}: {invalid_coord_axes}"
            )

        invalid_fixed_axes = set(fixed_axes) - valid_axes
        if invalid_fixed_axes:
            raise ValueError(
                f"Invalid fixed axes for {self.metadata.name}: {invalid_fixed_axes}"
            )

        # coord_axes and fixed_axes cannot share labels
        overlap = coord_axes & set(fixed_axes)
        if overlap:
            raise ValueError(
                f"Axes {overlap} cannot appear in both coord_axes and fixed_axes"
            )


class CartesianCoords(CoordSystem):
    """Cartesian coordinate system: x, y, z."""

    # Metadata as class attributes
    metadata = CoordSystemMetadata(
        name="Cartesian Coordinates",
        is_orthogonal=True,
        axes_labels=("x", "y", "z"),
        axes_bounds={
            "x": (-np.inf, np.inf),
            "y": (-np.inf, np.inf),
            "z": (-np.inf, np.inf),
        },
        max_dim_geometric=3,
        internal_coord_system=CoordSystemLabels.CARTESIAN,
        required_axes=set(),
        coord_system_redirections={
            frozenset({"y"}): (CoordSystemLabels.CARTESIAN, {"x"}),
            frozenset({"z"}): (CoordSystemLabels.CARTESIAN, {"x"}),
            frozenset({"x", "z"}): (CoordSystemLabels.CARTESIAN, {"x", "y"}),
            frozenset({"y", "z"}): (CoordSystemLabels.CARTESIAN, {"x", "y"}),
        },
    )

    def metric_tensor(
        self,
        n_points: int,
        geometric_coords: dict[str, torch.Tensor],
    ) -> MetricTensor:
        # For Cartesian, metric is identity matrix
        return IdentityMetricTensor(
            n_points, self.device, dtype=self._dtype(geometric_coords)
        )

    @property
    def internal_geometric_labels(self) -> tuple[str, ...]:
        # For Cartesian, internal geometric coords are the same as geometric coords
        return self.axes_labels_selected

    def map_geometric_to_internal_geometric(
        self,
        geometric_coords: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        # For Cartesian, internal geometric coords are the same as geometric coords
        return geometric_coords

    def internal_bounds_from_geometric(
        self,
        geometric_lb: dict[str, torch.Tensor],
        geometric_ub: dict[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        # For Cartesian, internal geometric coords are the same as geometric coords
        return geometric_lb, geometric_ub

    def geometric_to_cartesian_rotation(
        self, n_points: int, geometric_coords: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        # In Cartesian the basis vectors are already the Cartesian axes
        rotation = torch.eye(
            self.metadata.max_dim_geometric,
            dtype=self._dtype(geometric_coords),
            device=self.device,
        )
        rotation = rotation.expand(n_points, -1, -1)

        # Only use the columns of the axes included in geometric_coords
        return rotation[:, :, list(self.indices_selected)]


class CylindricalCoords(CoordSystem):
    """Cylindrical polar coordinate system: r, theta, z."""

    # Metadata as class attributes
    metadata = CoordSystemMetadata(
        name="Cylindrical Polar Coordinates",
        is_orthogonal=True,
        axes_labels=("r", "theta", "z"),
        axes_bounds={"r": (0, np.inf), "theta": (0, 2 * np.pi), "z": (-np.inf, np.inf)},
        max_dim_geometric=3,
        internal_coord_system=CoordSystemLabels.CARTESIAN,
        required_axes={"r"},
        coord_system_redirections={
            frozenset({"r"}): (CoordSystemLabels.CARTESIAN, {"x"}),
            frozenset({"z"}): (CoordSystemLabels.CARTESIAN, {"x"}),
            frozenset({"r", "z"}): (CoordSystemLabels.CARTESIAN, {"x", "y"}),
        },
    )

    def metric_tensor(
        self,
        n_points: int,
        geometric_coords: dict[str, torch.Tensor],
    ) -> MetricTensor:
        # Get coordinates needed in metric tensor
        r = self._get_coord("r", geometric_coords)

        # Construct metric corresponding to max_dim_geometric
        diag = torch.ones(
            (n_points, self.metadata.max_dim_geometric),
            dtype=self._dtype(geometric_coords),
            device=self.device,
        )
        diag[:, 1] = r**2  # theta-theta component

        # Only use rows and columns of axes included in geometric_coords
        diag = diag[:, list(self.indices_selected)]
        return DiagonalMetricTensor(diag)

    @property
    def internal_geometric_labels(self) -> tuple[str, ...]:
        # Because of the coord system redirecting, the possible combinations are small
        # Sort in order of the metadata.axes_labels of the internal coordinate system
        internal_coord_system = COORD_SYSTEM_REGISTRY[
            self.metadata.internal_coord_system
        ]

        # (theta) -> (x, y) and (r, theta) -> (x, y)
        if set(self.axes_labels_selected) == {"theta"} or set(
            self.axes_labels_selected
        ) == {"r", "theta"}:
            return tuple(
                label
                for label in internal_coord_system.metadata.axes_labels
                if label in {"x", "y"}
            )

        # (theta, z) -> (x, y, z) and (r, theta, z) -> (x, y, z)
        elif set(self.axes_labels_selected) == {"theta", "z"} or set(
            self.axes_labels_selected
        ) == {"r", "theta", "z"}:
            return tuple(
                label
                for label in internal_coord_system.metadata.axes_labels
                if label in {"x", "y", "z"}
            )

        else:
            raise ValueError(f"Unexpected axes selection: {self.axes_labels_selected}")

    def map_geometric_to_internal_geometric(
        self,
        geometric_coords: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        # r and theta must always be available in cylindrical polars
        r = self._get_coord("r", geometric_coords)
        theta = self._get_coord("theta", geometric_coords)

        # Transform
        internal_geometric_coords: dict[str, torch.Tensor] = {
            "x": r * torch.cos(theta),
            "y": r * torch.sin(theta),
        }
        if "z" in self.axes_labels_selected:
            internal_geometric_coords["z"] = self._get_coord("z", geometric_coords)
        return internal_geometric_coords

    def internal_bounds_from_geometric(
        self,
        geometric_lb: dict[str, torch.Tensor],
        geometric_ub: dict[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        # r and theta bounds
        r_lb = self._get_coord("r", geometric_lb)
        r_ub = self._get_coord("r", geometric_ub)
        theta_lb = self._get_coord("theta", geometric_lb)
        theta_ub = self._get_coord("theta", geometric_ub)

        # sin(theta) and cos(theta) bounds
        sin_lb, sin_ub = trig_bounds("sin", theta_lb, theta_ub)
        cos_lb, cos_ub = trig_bounds("cos", theta_lb, theta_ub)

        # x = r cos(theta), y = r sin(theta)
        x_candidates = torch.stack(
            [r * c for r, c in product([r_lb, r_ub], [cos_lb, cos_ub])]
        )
        y_candidates = torch.stack(
            [r * s for r, s in product([r_lb, r_ub], [sin_lb, sin_ub])]
        )

        internal_lb = {
            "x": torch.min(x_candidates),
            "y": torch.min(y_candidates),
        }
        internal_ub = {
            "x": torch.max(x_candidates),
            "y": torch.max(y_candidates),
        }

        if "z" in self.axes_labels_selected:
            z_lb = self._get_coord("z", geometric_lb)
            z_ub = self._get_coord("z", geometric_ub)
            internal_lb["z"] = z_lb
            internal_ub["z"] = z_ub

        return internal_lb, internal_ub

    def geometric_to_cartesian_rotation(
        self, n_points: int, geometric_coords: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        # Get coordinates needed for the basis vectors
        theta = self._get_coord("theta", geometric_coords)
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)

        # Each column is a unit basis vector, written in Cartesian (x, y, z)
        rotation = torch.zeros(
            (n_points, 3, 3), dtype=self._dtype(geometric_coords), device=self.device
        )

        # e_r = (cos(theta), sin(theta), 0)
        rotation[:, 0, 0] = cos_theta
        rotation[:, 1, 0] = sin_theta

        # e_theta = (-sin(theta), cos(theta), 0)
        rotation[:, 0, 1] = -sin_theta
        rotation[:, 1, 1] = cos_theta

        # e_z = (0, 0, 1)
        rotation[:, 2, 2] = 1.0

        # Only use the columns of the axes included in geometric_coords
        return rotation[:, :, list(self.indices_selected)]


class SphericalCoords(CoordSystem):
    """
    Spherical polar coordinate system: r, theta, phi.

    Without phi, (r, theta) are polar coordinates in the x-z plane, with
    theta measured from the z axis. They do not describe an axisymmetric 3D
    problem, so e.g. the Laplacian of 1/r is not zero.
    """

    # Metadata as class attributes
    metadata = CoordSystemMetadata(
        name="Spherical Polar Coordinates",
        is_orthogonal=True,
        axes_labels=("r", "theta", "phi"),
        axes_bounds={"r": (0, np.inf), "theta": (0, np.pi), "phi": (0, 2 * np.pi)},
        max_dim_geometric=3,
        internal_coord_system=CoordSystemLabels.CARTESIAN,
        required_axes={"r", "theta"},
        coord_system_redirections={
            frozenset({"r"}): (CoordSystemLabels.CARTESIAN, {"x"}),
            frozenset({"phi"}): (CoordSystemLabels.CYLINDRICAL, {"theta"}),
            frozenset({"r", "phi"}): (CoordSystemLabels.CYLINDRICAL, {"r", "theta"}),
        },
    )

    def metric_tensor(
        self,
        n_points: int,
        geometric_coords: dict[str, torch.Tensor],
    ) -> MetricTensor:
        # Get coordinates needed in metric tensor
        r = self._get_coord("r", geometric_coords)
        theta = self._get_coord("theta", geometric_coords)

        # Construct metric corresponding to max_dim_geometric
        diag = torch.ones(
            (n_points, self.metadata.max_dim_geometric),
            dtype=self._dtype(geometric_coords),
            device=self.device,
        )
        diag[:, 1] = r**2  # theta-theta component
        diag[:, 2] = r**2 * torch.sin(theta) ** 2  # phi-phi component

        # Only use rows and columns of axes included in geometric_coords
        diag = diag[:, list(self.indices_selected)]
        return DiagonalMetricTensor(diag)

    @property
    def internal_geometric_labels(self) -> tuple[str, ...]:
        # Because of the coord system redirecting, the possible combinations are small
        # Sort in order of the metadata.axes_labels of the internal coordinate system
        internal_coord_system = COORD_SYSTEM_REGISTRY[
            self.metadata.internal_coord_system
        ]

        # (theta) -> (x, z) and (r, theta) -> (x, z)
        if set(self.axes_labels_selected) == {"theta"} or set(
            self.axes_labels_selected
        ) == {"r", "theta"}:
            return tuple(
                label
                for label in internal_coord_system.metadata.axes_labels
                if label in {"x", "z"}
            )

        # (theta, phi) -> (x, y, z) and (r, theta, phi) -> (x, y, z)
        elif set(self.axes_labels_selected) == {"theta", "phi"} or set(
            self.axes_labels_selected
        ) == {"r", "theta", "phi"}:
            return tuple(
                label
                for label in internal_coord_system.metadata.axes_labels
                if label in {"x", "y", "z"}
            )

        else:
            raise ValueError(f"Unexpected axes selection: {self.axes_labels_selected}")

    def map_geometric_to_internal_geometric(
        self,
        geometric_coords: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        # r and theta must always be available in spherical polars
        r = self._get_coord("r", geometric_coords)
        theta = self._get_coord("theta", geometric_coords)

        # Transform
        if "phi" in self.axes_labels_selected:
            phi = self._get_coord("phi", geometric_coords)
            return {
                "x": r * torch.sin(theta) * torch.cos(phi),
                "y": r * torch.sin(theta) * torch.sin(phi),
                "z": r * torch.cos(theta),
            }
        else:
            return {
                "x": r * torch.sin(theta),
                "z": r * torch.cos(theta),
            }

    def internal_bounds_from_geometric(
        self,
        geometric_lb: dict[str, torch.Tensor],
        geometric_ub: dict[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        # Extract bounds
        r_lb = self._get_coord("r", geometric_lb)
        r_ub = self._get_coord("r", geometric_ub)
        theta_lb = self._get_coord("theta", geometric_lb)
        theta_ub = self._get_coord("theta", geometric_ub)

        # Trig bounds for theta
        sin_t_lb, sin_t_ub = trig_bounds("sin", theta_lb, theta_ub)
        cos_t_lb, cos_t_ub = trig_bounds("cos", theta_lb, theta_ub)

        # z = r cos(theta)
        z_bound_combinations = torch.stack(
            [r * c for r, c in product([r_lb, r_ub], [cos_t_lb, cos_t_ub])]
        )

        if "phi" in self.axes_labels_selected:
            phi_lb = self._get_coord("phi", geometric_lb)
            phi_ub = self._get_coord("phi", geometric_ub)

            sin_p_lb, sin_p_ub = trig_bounds("sin", phi_lb, phi_ub)
            cos_p_lb, cos_p_ub = trig_bounds("cos", phi_lb, phi_ub)

            # x = r sin(theta) cos(phi), y = r sin(theta) sin(phi)
            x_bound_combinations = torch.stack(
                [
                    r * s * c
                    for r, s, c in product(
                        [r_lb, r_ub], [sin_t_lb, sin_t_ub], [cos_p_lb, cos_p_ub]
                    )
                ]
            )
            y_bound_combinations = torch.stack(
                [
                    r * s * c
                    for r, s, c in product(
                        [r_lb, r_ub], [sin_t_lb, sin_t_ub], [sin_p_lb, sin_p_ub]
                    )
                ]
            )

            internal_lb = {
                "x": torch.min(x_bound_combinations),
                "y": torch.min(y_bound_combinations),
                "z": torch.min(z_bound_combinations),
            }
            internal_ub = {
                "x": torch.max(x_bound_combinations),
                "y": torch.max(y_bound_combinations),
                "z": torch.max(z_bound_combinations),
            }

        else:
            # x = r sin(theta), no phi
            x_bound_combinations = torch.stack(
                [r * s for r, s in product([r_lb, r_ub], [sin_t_lb, sin_t_ub])]
            )

            internal_lb = {
                "x": torch.min(x_bound_combinations),
                "z": torch.min(z_bound_combinations),
            }
            internal_ub = {
                "x": torch.max(x_bound_combinations),
                "z": torch.max(z_bound_combinations),
            }

        return internal_lb, internal_ub

    def geometric_to_cartesian_rotation(
        self, n_points: int, geometric_coords: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        # Get coordinates needed for the basis vectors
        theta = self._get_coord("theta", geometric_coords)
        phi = self._get_coord("phi", geometric_coords, default=0.0)
        sin_theta, cos_theta = torch.sin(theta), torch.cos(theta)
        sin_phi, cos_phi = torch.sin(phi), torch.cos(phi)

        # Each column is a unit basis vector, written in Cartesian (x, y, z)
        rotation = torch.zeros(
            (n_points, 3, 3), dtype=self._dtype(geometric_coords), device=self.device
        )

        # e_r = (sin(theta)cos(phi), sin(theta)sin(phi), cos(theta))
        rotation[:, 0, 0] = sin_theta * cos_phi
        rotation[:, 1, 0] = sin_theta * sin_phi
        rotation[:, 2, 0] = cos_theta

        # e_theta = (cos(theta)cos(phi), cos(theta)sin(phi), -sin(theta))
        rotation[:, 0, 1] = cos_theta * cos_phi
        rotation[:, 1, 1] = cos_theta * sin_phi
        rotation[:, 2, 1] = -sin_theta

        # e_phi = (-sin(phi), cos(phi), 0)
        rotation[:, 0, 2] = -sin_phi
        rotation[:, 1, 2] = cos_phi

        # Only use the columns of the axes included in geometric_coords
        return rotation[:, :, list(self.indices_selected)]


# Mapping from coordinate system labels to classes
COORD_SYSTEM_REGISTRY: dict[CoordSystemLabels, type[CoordSystem]] = {
    CoordSystemLabels.CARTESIAN: CartesianCoords,
    CoordSystemLabels.CYLINDRICAL: CylindricalCoords,
    CoordSystemLabels.SPHERICAL: SphericalCoords,
}
