from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ..geometry.coordinate_system import CoordSystem


class Operators:
    def __init__(self, coord_system: CoordSystem, coords: dict[str, torch.Tensor]):
        self.coord_system = coord_system
        self.coords = coords

    def d_dcoord(
        self,
        f: torch.Tensor,
        coord_label: str,
    ) -> torch.Tensor:
        # Get coordinate from label
        coord = self.coords[coord_label]

        # Calculate derivatives using autograd
        derivative_f = torch.autograd.grad(
            f,
            coord,
            grad_outputs=torch.ones_like(f),
            create_graph=True,
            retain_graph=True,
        )[0]

        return derivative_f

    def grad(
        self,
        f: torch.Tensor,
    ) -> torch.Tensor:
        # Get geometric coordinates
        coords_geometric = [
            self.coords[label] for label in self.coord_system.axes_labels_selected
        ]

        # Take transpose so that eqn. file uses indices like [0] rather than [:, 0]
        grad = self.coord_system.grad(f, coords_geometric).T

        return grad

    def contravariant_grad(
        self,
        f: torch.Tensor,
    ) -> torch.Tensor:
        # Get geometric coordinates
        coords_geometric = [
            self.coords[label] for label in self.coord_system.axes_labels_selected
        ]

        # Take transpose so that eqn. file uses indices like [0] rather than [:, 0]
        contra_grad = self.coord_system.contravariant_grad(f, coords_geometric).T

        return contra_grad

    def div(
        self,
        v: torch.Tensor,
    ) -> torch.Tensor:
        # Get geometric coordinates
        coords_geometric = [
            self.coords[label] for label in self.coord_system.axes_labels_selected
        ]

        return self.coord_system.div(v, coords_geometric)

    def laplacian(
        self,
        f: torch.Tensor,
    ) -> torch.Tensor:
        # Get geometric coordinates
        coords_geometric = [
            self.coords[label] for label in self.coord_system.axes_labels_selected
        ]

        return self.coord_system.laplacian(f, coords_geometric)
