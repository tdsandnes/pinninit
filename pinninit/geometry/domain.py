from __future__ import annotations

import numpy as np
import torch

from .coordinate_system import CartesianCoords, CoordSystem


class Domain:
    """
    The bounds of a set of coordinates, in their coord system.

    Parameters
    ----------
    coord_system:
        The coord system of the coordinates.
    geometric_bounds:
        (min, max) of each axis. Geometric axes without bounds, or with a
        bound of None, use the bounds of the coord system (e.g. r >= 0).
        Non-geometric axes (e.g. time) need both bounds.
    periodic_coords:
        Geometric axes that are periodic, with the length of the domain as
        the period. Only supported in Cartesian coordinates.
    """

    def __init__(
        self,
        coord_system: CoordSystem,
        geometric_bounds: dict[str, tuple[float | None, float | None]] | None = None,
        periodic_coords: set[str] | None = None,
    ) -> None:
        self.coord_system = coord_system
        self.geometric_labels = coord_system.axes_labels_selected
        self.device = coord_system.device

        self._set_geometric_bounds(geometric_bounds)
        self._set_non_geometric_bounds(geometric_bounds)
        self._set_internal_bounds()
        self._set_periodic_coords(periodic_coords)

    def _set_geometric_bounds(
        self,
        geometric_bounds: dict[str, tuple[float | None, float | None]] | None,
    ) -> None:
        self.geometric_lb: dict[str, torch.Tensor] = {}
        self.geometric_ub: dict[str, torch.Tensor] = {}

        for axis in self.geometric_labels:
            meta_lb, meta_ub = self.coord_system.axes_bounds_selected[axis]

            # Use user-provided bounds if available, else fall back to metadata
            if geometric_bounds is None or axis not in geometric_bounds:
                lb, ub = meta_lb, meta_ub
            else:
                user_lb, user_ub = geometric_bounds[axis]
                lb = meta_lb if user_lb is None else user_lb
                ub = meta_ub if user_ub is None else user_ub

            # Validate only if metadata bounds are finite
            if not np.isinf(meta_lb) and lb < meta_lb:
                raise ValueError(
                    f"Lower bound for axis '{axis}' must be >= {meta_lb}, got {lb}"
                )
            if not np.isinf(meta_ub) and ub > meta_ub:
                raise ValueError(
                    f"Upper bound for axis '{axis}' must be <= {meta_ub}, got {ub}"
                )

            if lb > ub:
                raise ValueError(
                    f"Lower bound must be <= upper bound for axis '{axis}', "
                    f"got [{lb}, {ub}]"
                )

            self.geometric_lb[axis] = torch.tensor(float(lb), device=self.device)
            self.geometric_ub[axis] = torch.tensor(float(ub), device=self.device)

    def _set_non_geometric_bounds(
        self,
        bounds: dict[str, tuple[float | None, float | None]] | None,
    ) -> None:
        """Bounds of non-geometric axes (e.g. time), used for input scaling."""
        self.non_geometric_lb: dict[str, torch.Tensor] = {}
        self.non_geometric_ub: dict[str, torch.Tensor] = {}

        if bounds is None:
            return

        for axis, (lb, ub) in bounds.items():
            if axis in self.geometric_labels:
                continue
            if lb is None or ub is None:
                raise ValueError(
                    f"Non-geometric axis '{axis}' requires explicit min and max."
                )
            if lb > ub:
                raise ValueError(
                    f"Lower bound must be <= upper bound for axis '{axis}', "
                    f"got [{lb}, {ub}]"
                )
            self.non_geometric_lb[axis] = torch.tensor(float(lb), device=self.device)
            self.non_geometric_ub[axis] = torch.tensor(float(ub), device=self.device)

    def _set_internal_bounds(self) -> None:
        # Derive internal bounds from coord system
        internal_lb, internal_ub = self.coord_system.internal_bounds_from_geometric(
            self.geometric_lb, self.geometric_ub
        )

        # Non-geometric axes pass straight through to the internal bounds. New
        # dicts, as a coord system can return the geometric bounds themselves.
        self.internal_lb = {**internal_lb, **self.non_geometric_lb}
        self.internal_ub = {**internal_ub, **self.non_geometric_ub}

    def contains_geometric(self, geometric: torch.Tensor) -> torch.Tensor:
        lb = torch.stack([self.geometric_lb[label] for label in self.geometric_labels])
        ub = torch.stack([self.geometric_ub[label] for label in self.geometric_labels])
        return ((geometric >= lb) & (geometric <= ub)).all(dim=1)

    def _set_periodic_coords(
        self,
        periodic_coords: set[str] | None,
    ) -> None:
        """Store the period of each periodic axis: the length of the domain."""
        if periodic_coords is None:
            self.periodic_coords: dict[str, float] = {}
            return

        if not isinstance(self.coord_system, CartesianCoords):
            raise ValueError(
                "Periodic boundary conditions are currently only supported for "
                f"Cartesian coordinate systems, got "
                f"'{self.coord_system.metadata.name}'."
            )

        for label in periodic_coords:
            if label not in self.geometric_labels:
                raise ValueError(
                    f"Periodic coord '{label}' is not a geometric axis of this domain. "
                    f"Available axes: {list(self.geometric_labels)}"
                )

        # Infer period from domain bounds for each periodic axis
        self.periodic_coords = {
            label: (self.geometric_ub[label] - self.geometric_lb[label]).item()
            for label in self.geometric_labels
            if label in periodic_coords
        }
        for label, period in self.periodic_coords.items():
            if not (np.isfinite(period) and period > 0):
                raise ValueError(
                    f"Periodic coord '{label}' needs finite bounds with lower < "
                    f"upper, got a period of {period}"
                )
