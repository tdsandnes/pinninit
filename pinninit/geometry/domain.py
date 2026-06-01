from __future__ import annotations

import numpy as np
import torch

from .coordinate_system import CoordSystem, CoordSystemLabels


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
        self.device = device

        self._set_bounds(geometric_bounds)
        self._set_periodic_coords(periodic_coords)

    def _set_bounds(
        self,
        geometric_bounds: dict[str, tuple[float | None, float | None]] | None,
    ) -> None:
        metadata_axes = self.coord_system.metadata.axes_labels
        metadata_bounds = self.coord_system.metadata.axes_bounds

        # Map metadata bounds by axis label
        metadata_bounds_dict: dict[str, tuple[float, float]] = {
            axis: metadata_bounds[i] for i, axis in enumerate(metadata_axes)
        }

        lbs: list[float] = []
        ubs: list[float] = []

        for axis in self.geometric_labels:
            if axis not in metadata_bounds_dict:
                raise ValueError(f"No metadata bounds defined for axis '{axis}'")

            meta_lb, meta_ub = metadata_bounds_dict[axis]

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

            if lb >= ub:
                raise ValueError(
                    f"Lower bound must be < upper bound for axis '{axis}', "
                    f"got [{lb}, {ub}]"
                )

            lbs.append(lb)
            ubs.append(ub)

        self.geometric_lb = torch.tensor(lbs, device=self.device)
        self.geometric_ub = torch.tensor(ubs, device=self.device)

    def contains_geometric(
        self,
        geometric: torch.Tensor,
    ) -> torch.Tensor:
        return (
            (geometric >= self.geometric_lb) & (geometric <= self.geometric_ub)
        ).all(dim=1)

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
            periodicity.
        """
        if periodic_coords is None:
            self.periodic_coords: dict[str, float] = {}
            return

        if (
            self.coord_system.metadata.internal_coord_system
            != CoordSystemLabels.CARTESIAN
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
            label: (self.geometric_ub[i] - self.geometric_lb[i]).item()
            for i, label in enumerate(self.geometric_labels)
            if label in periodic_coords
        }
