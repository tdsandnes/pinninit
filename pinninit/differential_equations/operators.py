from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ..data_structure.coords_dataclass import CoordsProtocol


class Operators:
    def __init__(self, coords: CoordsProtocol):
        # Extract from coords
        self.coords = coords
        self.coord_system = coords.coord_system
        self.geometric_labels = coords.geometric_labels
        self.dim_geometric = self.coord_system.dim_geometric
        self.device = self.coords.device

        # Set metric tensor
        geometric_coords_dict = {
            label: self.coords.get_coord(label) for label in self.geometric_labels
        }
        self.G = self.coord_system.metric_tensor(
            self.coords.n_points, geometric_coords_dict
        )

    def d_dcoord(
        self,
        f: torch.Tensor,
        coord_label: str,
    ) -> torch.Tensor:
        """
        Compute the derivative of all elemets of f wrt a given coordinate

        Parameters
        ----------
        f : torch.Tensor
            Function to be taken derivative of.
            Shape: (n_points,), (n_points, dim), (n_points, dim, dim), ...

        Returns
        -------
        torch.Tensor
            Derivative of each element of f.
        """
        coord = self.coords.get_coord(coord_label)
        return torch.autograd.grad(
            outputs=f,
            inputs=coord,
            grad_outputs=torch.ones_like(f),
            create_graph=True,
            retain_graph=True,
        )[0]

    def contravariant_grad(self, f: torch.Tensor) -> torch.Tensor:
        """
        Compute the contravarient gradient of a function in a given coordinate system.

        contravariant_grad(f) = g^{ij} * partial_j f

        Parameters
        ----------
        f : torch.Tensor
            Function to be taken gradient of. Shape: (n_points,)

        Returns
        -------
        torch.Tensor
            Contravarient gradient of the function: (n_points, dim_geometric)
        """
        # List of derivatives wrt geometric coordinates
        derivative_f = torch.autograd.grad(
            outputs=f,
            inputs=self.coords.geometric,
            grad_outputs=torch.ones_like(f),
            create_graph=True,
            retain_graph=True,
        )[0]

        return self.G.inv_metric_times_vec(derivative_f)

    def grad(
        self,
        f: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the gradient of a function in a physical basis.

        Parameters
        ----------
        f : torch.Tensor
            Function to be taken gradient of. Shape: (n_points,)

        Returns
        -------
        torch.Tensor
            Gradient of the function: (n_points, dim_geometric)
        """
        return self.G.scale_factors_times_vec(self.contravariant_grad(f))

    def v_dot_grad(
        self,
        v1: torch.Tensor,
        v2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the operation (v1 dot grad)(v2) in physical basis.

        Parameters
        ----------
        v1 : torch.Tensor
            The vector field used in dot product with grad.
            Shape (n_points, dim_geometric)
        v2 : torch.Tensor
            The vector field that the operator (v1 dot grad) acts on.
            Shape (n_points, dim_geometric)

        Returns
        -------
        torch.Tensor
            The vector computed from (v1 dot grad)(v2).
            Shape (n_points, dim_geometric)
        """
        # Loop through each component of the target vector field v2
        # v2[:, i] is a scalar field corresponding to component i
        components = [
            (v1 * self.grad(v2[:, i])).sum(dim=1) for i in range(self.dim_geometric)
        ]

        # Stack them back into a vector field of shape (n_points, dim_geometric)
        return torch.stack(components, dim=1)

    def contravariant_div(
        self,
        contra_v: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the divergence of a vector field in contravariant basis.

        Parameters
        ----------
        contra_v : torch.Tensor
            The vector field, given in contravariant basis.
            Shape (n_points, dim_geometric)

        Returns
        -------
        torch.Tensor
            The divergence of the vector field v.
            Shape (n_points,)
        """
        # Term to take derivatives of
        inner_term = self.G.sqrt_det_times_vec(contra_v)

        div_terms = []
        for i, coord_label in enumerate(self.geometric_labels):
            coord = self.coords.get_coord(coord_label)
            derivative = torch.autograd.grad(
                outputs=inner_term[:, i],
                inputs=coord,
                grad_outputs=torch.ones_like(coord),
                create_graph=True,
                retain_graph=True,
            )[0]
            div_terms.append(derivative)

        sum_derivatives = torch.stack(div_terms, dim=1).sum(dim=1)

        return self.G.inv_sqrt_det_times_scalar(sum_derivatives)

    def div(
        self,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the divergence of a vector field in physical basis.

        Parameters
        ----------
        v : torch.Tensor
            The vector field, given in physical basis.
            Shape (n_points, dim_geometric)

        Returns
        -------
        torch.Tensor
            The divergence of the vector field v.
            Shape (n_points,)
        """
        return self.contravariant_div(self.G.inv_scale_factors_times_vec(v))

    def laplacian(
        self,
        f: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the laplacian operator of a function in a given coordinate system.

        Parameters
        ----------
        f : torch.Tensor
            Function to be taken gradient of. Shape: (n_points,)

        Returns
        -------
        torch.Tensor
            Laplacian of the function. Shape (n_points,)
        """
        return self.contravariant_div(self.contravariant_grad(f))

    def vector_laplacian(
        self,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the laplacian operator of a function in a given coordinate system.

        Parameters
        ----------
        v : torch.Tensor
            The vector field.
            Shape (n_points, dim_geometric)

        Returns
        -------
        torch.Tensor
            Vector laplacian of the function and same basis as input vector
            field (n_points, dim_geometric)
        """
        return torch.stack(
            [self.laplacian(v[:, i]) for i in range(self.dim_geometric)],
            dim=1,
        )
