from __future__ import annotations

import numpy as np
import torch

from .coordinate_system import COORD_SYSTEM_REGISTRY, CoordSystem, CoordSystemLabels


class Domain:
    def __init__(
        self,
        coord_system: CoordSystem,
        geometric_bounds: dict[str, tuple[float | None, float | None]] | None,
        periodic_coords: set[str] | None,
        device: torch.device,
    ):
        self.coord_system = coord_system
        self.geometric_labels = coord_system.axes_labels_selected
        self.metadata_selected_bounds = coord_system.axes_bounds_selected
        self.device = device

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
            if axis not in self.metadata_selected_bounds:
                raise ValueError(f"No metadata bounds defined for axis '{axis}'")

            meta_lb, meta_ub = self.metadata_selected_bounds[axis]

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

            self.geometric_lb[axis] = torch.tensor(lb, device=self.device)
            self.geometric_ub[axis] = torch.tensor(ub, device=self.device)

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
            self.non_geometric_lb[axis] = torch.tensor(lb, device=self.device)
            self.non_geometric_ub[axis] = torch.tensor(ub, device=self.device)

    def _set_internal_bounds(self) -> None:
        # Derive internal bounds from coord system
        self.internal_lb, self.internal_ub = (
            self.coord_system.internal_bounds_from_geometric(
                self.geometric_lb, self.geometric_ub
            )
        )

        # Non-geometric axes pass straight through to the internal bounds
        self.internal_lb.update(self.non_geometric_lb)
        self.internal_ub.update(self.non_geometric_ub)

    def contains_geometric(self, geometric: torch.Tensor) -> torch.Tensor:
        lb = torch.stack([self.geometric_lb[label] for label in self.geometric_labels])
        ub = torch.stack([self.geometric_ub[label] for label in self.geometric_labels])
        return ((geometric >= lb) & (geometric <= ub)).all(dim=1)

    def _set_periodic_coords(
        self,
        periodic_coords: set[str] | None,
    ) -> None:
        """
        Validate and store periodic coordinate labels.

        Only supported for Cartesian coordinate systems. For each periodic axis,
        the period is inferred as (ub - lb) from the domain bounds set by _set_bounds.

        Parameters
        ----------
        periodic_coords : set[str] | None
            Set of geometric axis labels to treat as periodic. None means no
            periodicity.f
        """
        if periodic_coords is None:
            self.periodic_coords: dict[str, float] = {}
            return

        if not isinstance(
            self.coord_system, COORD_SYSTEM_REGISTRY[CoordSystemLabels.CARTESIAN]
        ):
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
