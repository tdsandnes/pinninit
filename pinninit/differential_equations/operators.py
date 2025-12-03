from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ..geometry.coordinate_system import CoordSystem


class Operators:
    def __init__(
        self,
        coord_system: CoordSystem,
        coords: list[torch.Tensor],
        geometric_axis_mask: list[int],
        coord_to_index_dict: dict[str, int],
        device: torch.device,
    ):
        self.coord_system = coord_system
        self.coords = coords
        self.geometric_axis_mask = geometric_axis_mask
        self.coord_to_index_dict = coord_to_index_dict
        self.device = device

    def d_dcoord(
        self,
        f: torch.Tensor,
        coord_label: str,
    ) -> torch.Tensor:
        # Get coordinate from label
        coord_index = self.coord_to_index_dict[coord_label]
        coord = self.coords[coord_index]

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
        coords_geometric = [self.coords[i] for i in self.geometric_axis_mask]

        # Take transpose so that eqn. file uses indices like [0] rather than [:, 0]
        grad = self.coord_system.grad(f, coords_geometric, self.device).T

        return grad

    def contravariant_grad(
        self,
        f: torch.Tensor,
    ) -> torch.Tensor:
        # Get geometric coordinates
        coords_geometric = [self.coords[i] for i in self.geometric_axis_mask]

        # Take transpose so that eqn. file uses indices like [0] rather than [:, 0]
        contra_grad = self.coord_system.contravariant_grad(
            f, coords_geometric, self.device
        ).T

        return contra_grad

    def laplacian(
        self,
        f: torch.Tensor,
    ) -> torch.Tensor:
        # Get geometric coordinates
        coords_geometric = [self.coords[i] for i in self.geometric_axis_mask]

        return self.coord_system.laplacian(f, coords_geometric, self.device)
