from __future__ import annotations

from typing import TYPE_CHECKING

import torch

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

    def d_dcoord(
        self,
        f: torch.Tensor,
        coord_label: str,
    ) -> torch.Tensor:
        coord = self.coords[coord_label]

        # Scalar field: shape (num_points,)
        if f.dim() == 1:
            derivative_f = torch.autograd.grad(
                f,
                coord,
                grad_outputs=torch.ones_like(f),
                create_graph=True,
                retain_graph=True,
            )[0]
            return derivative_f

        # Vector field: shape (num_points, dim)
        elif f.dim() == 2:
            dim = f.shape[1]
            derivative_vector_f = []
            for i in range(dim):
                derivative_f_i = torch.autograd.grad(
                    f[:, i],
                    coord,
                    grad_outputs=torch.ones_like(f[:, i]),
                    create_graph=True,
                    retain_graph=True,
                )[0]
                derivative_vector_f.append(derivative_f_i)

            return torch.stack(derivative_vector_f, dim=1)

        else:
            raise ValueError(
                "d_dcoord only supports scalar fields (n_points,) or vector fields "
                f"(n_points, dim). Got shape {f.shape}"
            )

    def grad(
        self,
        f: torch.Tensor,
    ) -> torch.Tensor:
        return self.coord_system.grad(f, self.coords_geometric)

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
