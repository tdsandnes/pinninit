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
    """Base class for coordinate systems."""

    # Placeholder for subclass class attributes
    metadata: CoordSystemMetadata

    def __init__(
        self,
        dim_geometric: int,  # Number of geometric dimensions
        coord_axes: set[str],  # Labels of coordinate axes to use (not including fixed)
        device: torch.device,  # device
        fixed_axes: dict[str, float] | None = None,  # {Labels: vlaue} of fixed axes
    ):
        # Validate input
        self._validate_selected_dimensions(dim_geometric)
        self._validate_selected_axes(coord_axes, fixed_axes)

        # Store values
        self.dim_geometric = dim_geometric
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
        self.fixed_axes = (
            {
                label: fixed_axes[label]
                for label in self.metadata.axes_labels
                if label in fixed_axes
            }
            if fixed_axes is not None
            else None
        )

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

    def _get_coord(
        self,
        axis_label: str,
        geometric_coords: dict[str, torch.Tensor],
        default: float | None = None,
    ) -> torch.Tensor:
        if axis_label in self.axes_labels_selected:
            return geometric_coords[axis_label]
        elif self.fixed_axes is not None and axis_label in self.fixed_axes:
            return torch.tensor(self.fixed_axes[axis_label])
        elif default is not None:
            return torch.tensor(default)
        else:
            raise ValueError(f"Missing required axis: {axis_label}")

    def _validate_selected_dimensions(self, dim_geometric: int) -> None:
        # Check value of dim_geometric
        if dim_geometric < 1:
            raise ValueError(f"dim_geometric must be >= 1, got {dim_geometric}")
        if dim_geometric > self.metadata.max_dim_geometric:
            raise ValueError(
                f"{self.metadata.name} only implemented for dim_geometric <="
                f"{self.metadata.max_dim_geometric}, got dim_geometric={dim_geometric}"
            )

    def _validate_selected_axes(
        self, coord_axes: set[str], fixed_axes: dict[str, float] | None = None
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
            if (axis_label not in coord_axes) and (
                fixed_axes is None or axis_label not in fixed_axes
            ):
                raise ValueError(f"Missing required axis: {axis_label}")

        # Both coord_axes and fixed_axes must be in the coordinate system's axes_labels
        valid_axes = set(self.metadata.axes_labels)
        invalid_coord_axes = coord_axes - valid_axes
        if invalid_coord_axes:
            raise ValueError(
                f"Invalid axes for {self.metadata.name}: {invalid_coord_axes}"
            )

        if fixed_axes is not None:
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

    def __init__(
        self,
        dim_geometric: int,
        coord_axes: set[str],
        device: torch.device,
        fixed_axes: dict[str, float] | None = None,
    ):
        super().__init__(
            dim_geometric=dim_geometric,
            coord_axes=coord_axes,
            device=device,
            fixed_axes=fixed_axes,
        )

    def metric_tensor(
        self,
        n_points: int,
        geometric_coords: dict[str, torch.Tensor] | None = None,
    ) -> MetricTensor:
        # For Cartesian, metric is identity matrix
        return IdentityMetricTensor(n_points, self.device)

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

    def __init__(
        self,
        dim_geometric: int,
        coord_axes: set[str],
        device: torch.device,
        fixed_axes: dict[str, float] | None = None,
    ):
        super().__init__(
            dim_geometric=dim_geometric,
            coord_axes=coord_axes,
            device=device,
            fixed_axes=fixed_axes,
        )

    def metric_tensor(
        self,
        n_points: int,
        geometric_coords: dict[str, torch.Tensor],
    ) -> MetricTensor:
        # Get coordinates needed in metric tensor
        r = self._get_coord("r", geometric_coords)

        # Construct metric corrsponding to max_dim_geometric
        diag = torch.ones(
            (n_points, self.metadata.max_dim_geometric), device=self.device
        )
        diag[:, 1] = r**2  # theta-theta component

        # Only use rows and columns of axes included in geometric_coords
        mask_keep = [
            i
            for i in range(self.metadata.max_dim_geometric)
            if i in self.indices_selected
        ]
        diag = diag[:, mask_keep]
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


class SphericalCoords(CoordSystem):
    """Spherical polar coordinate system: r, theta, phi."""

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

    def __init__(
        self,
        dim_geometric: int,
        coord_axes: set[str],
        device: torch.device,
        fixed_axes: dict[str, float] | None = None,
    ):
        super().__init__(
            dim_geometric=dim_geometric,
            coord_axes=coord_axes,
            device=device,
            fixed_axes=fixed_axes,
        )

    def metric_tensor(
        self,
        n_points: int,
        geometric_coords: dict[str, torch.Tensor],
    ) -> MetricTensor:
        # Get coordinates needed in metric tensor
        r = self._get_coord("r", geometric_coords)
        theta = self._get_coord("theta", geometric_coords)

        # Construct metric corrsponding to max_dim_geometric
        diag = torch.ones(
            (n_points, self.metadata.max_dim_geometric), device=self.device
        )
        diag[:, 1] = r**2  # theta-theta component
        diag[:, 2] = r**2 * torch.sin(theta) ** 2  # phi-phi component

        # Only use rows and columns of axes included in geometric_coords
        mask_keep = [
            i
            for i in range(self.metadata.max_dim_geometric)
            if i in self.indices_selected
        ]
        diag = diag[:, mask_keep]
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


# Mapping from coordinate system labels to classes
COORD_SYSTEM_REGISTRY: dict[CoordSystemLabels, type[CoordSystem]] = {
    CoordSystemLabels.CARTESIAN: CartesianCoords,
    CoordSystemLabels.CYLINDRICAL: CylindricalCoords,
    CoordSystemLabels.SPHERICAL: SphericalCoords,
}
