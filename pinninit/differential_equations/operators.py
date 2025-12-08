from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ..checks.validate_data import validate_pytorch_tensor

if TYPE_CHECKING:
    from ..geometry.coordinate_system import CoordSystem


class Operators:
    def __init__(self, coord_system: CoordSystem, coords: dict[str, torch.Tensor]):
        self.coord_system = coord_system
        self.coords = coords

        # Get geometric coordinates
        self.coords_geometric = [
            self.coords[label] for label in self.coord_system.axes_labels_selected
        ]

    def df_dcoord(
        self,
        f: torch.Tensor,
        coord_label: str,
    ) -> torch.Tensor:
        # Validate that f is scalar field
        n_points = f.shape[0]
        validate_pytorch_tensor(f, (n_points,))

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

    def dv_dcoord(
        self,
        v: torch.Tensor,
        coord_label: str,
    ) -> torch.Tensor:
        # Validate that v is vector field
        n_points = v.shape[0]
        dim = v.shape[1]
        validate_pytorch_tensor(v, (n_points, dim))

        # Get coordinate from label
        coord = self.coords[coord_label]

        # Calculate derivatives using autograd
        derivative_v_list = []
        for i in range(dim):
            derivative_v_i = torch.autograd.grad(
                v[i],
                coord,
                grad_outputs=torch.ones_like(v[i]),
                create_graph=True,
                retain_graph=True,
            )[0]

            derivative_v_list.append(derivative_v_i)

        # Stack back to tensor with shape (n_points, dim)
        derivative_v = torch.stack(derivative_v_list, dim=1)

        return derivative_v

    def grad(
        self,
        f: torch.Tensor,
    ) -> torch.Tensor:
        # Take transpose so that eqn. file uses indices like [0] rather than [:, 0]
        return self.coord_system.grad(f, self.coords_geometric).T

    def v_dot_grad(
        self,
        v1: torch.Tensor,
        v2: torch.Tensor,
    ) -> torch.Tensor:
        return self.coord_system.v_dot_grad(v1, v2, self.coords_geometric)

    def div(
        self,
        v: torch.Tensor,
    ) -> torch.Tensor:
        return self.coord_system.div(v, self.coords_geometric)

    def laplacian(
        self,
        f: torch.Tensor,
    ) -> torch.Tensor:
        return self.coord_system.laplacian(f, self.coords_geometric)

    def vector_laplacian(
        self,
        v: torch.Tensor,
    ) -> torch.Tensor:
        return self.coord_system.vector_laplacian(v, self.coords_geometric)
