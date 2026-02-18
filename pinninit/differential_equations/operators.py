from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ..checks.validate_data import validate_pytorch_tensor

if TYPE_CHECKING:
    from ..data_structure.coords_dataclass import CoordsProtocol


class Operators:
    def __init__(self, coords: CoordsProtocol):
        # Extract from coords
        self.coords = coords
        self.coord_system = coords.coord_system
        self.geometric_labels = coords.geometric_labels
        self.dim_geometry = self.coord_system.dim_geometry
        self.device = self.coords.device

        # Initial computation
        self.set_quantities()

    def set_quantities(self) -> None:
        """Compute quantities based on current coordinate state."""
        # Always recompute based on current geometric coords
        self.G = self.coord_system.metric_tensor(
            self.coords.n_points, self.coords.geometric
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
        # Compute gradient wrt the specific coordinate
        df_dcoord = torch.autograd.grad(
            outputs=f,
            inputs=coord,
            grad_outputs=torch.ones_like(f),
            create_graph=True,
            retain_graph=True,
        )[0]

        return df_dcoord

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
            Contravarient gradient of the function: (n_points, dim_geometry)
        """
        # Coordinate system dimensions
        dim_geometry = self.dim_geometry
        coords_geometric = self.coords.geometric

        # Validate input tensors
        n_points = self.coords.n_points
        validate_pytorch_tensor(f, (n_points,))
        validate_pytorch_tensor(coords_geometric, (n_points, dim_geometry))

        # List of derivatives wrt geometric coordinates
        derivative_f = torch.autograd.grad(
            outputs=f,
            inputs=coords_geometric,
            grad_outputs=torch.ones_like(f),
            create_graph=True,
            retain_graph=True,
        )[0]
        validate_pytorch_tensor(derivative_f, (n_points, dim_geometry))

        # Batch matrix multiplication with inverse metric tensor
        contra_grad_f = self.G.inv_metric_times_vec(derivative_f)
        validate_pytorch_tensor(contra_grad_f, (n_points, dim_geometry))

        return contra_grad_f

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
            Gradient of the function: (n_points, dim_geometry)
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
            Shape (n_points, dim_geometry)
        v2 : torch.Tensor
            The vector field that the operator (v1 dot grad) acts on.
            Shape (n_points, dim_geometry)

        Returns
        -------
        torch.Tensor
            The vector computed from (v1 dot grad)(v2).
            Shape (n_points, dim_geometry)
        """
        # Coordinate system dimensions
        dim_geometry = self.dim_geometry
        coords_geometric = self.coords.geometric
        n_points = self.coords.n_points

        # Validate inputs
        validate_pytorch_tensor(v1, (n_points, dim_geometry))
        validate_pytorch_tensor(v2, (n_points, dim_geometry))
        validate_pytorch_tensor(coords_geometric, (n_points, dim_geometry))

        # Calculate components
        v1_dot_grad_v2 = []
        for i in range(dim_geometry):
            grad_v2_i = self.grad(v2[:, i])
            validate_pytorch_tensor(grad_v2_i, (n_points, dim_geometry))
            v1_dot_grad_v2_i = (v1 * grad_v2_i).sum(dim=1)
            v1_dot_grad_v2.append(v1_dot_grad_v2_i)

        v1_dot_grad_v2_tensor = torch.stack(
            v1_dot_grad_v2, dim=1
        )  # shape: (n_points, dim_geometry)
        validate_pytorch_tensor(v1_dot_grad_v2_tensor, (n_points, dim_geometry))

        return v1_dot_grad_v2_tensor

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
            Shape (n_points, dim_geometry)

        Returns
        -------
        torch.Tensor
            The divergence of the vector field v.
            Shape (n_points,)
        """
        # Coordinate system dimensions
        dim_geometry = self.dim_geometry
        coords_geometric = self.coords.geometric
        n_points = self.coords.n_points

        # Validate inputs
        validate_pytorch_tensor(contra_v, (n_points, dim_geometry))
        validate_pytorch_tensor(coords_geometric, (n_points, dim_geometry))

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

        contra_div_v = self.G.inv_sqrt_det_times_scalar(sum_derivatives)
        validate_pytorch_tensor(contra_div_v, (n_points,))

        return contra_div_v

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
            Shape (n_points, dim_geometry)

        Returns
        -------
        torch.Tensor
            The divergence of the vector field v.
            Shape (n_points,)
        """
        # Coordinate system dimensions
        dim_geometry = self.dim_geometry
        coords_geometric = self.coords.geometric

        # Validate inputs
        validate_pytorch_tensor(v, (self.coords.n_points, dim_geometry))
        validate_pytorch_tensor(coords_geometric, (self.coords.n_points, dim_geometry))

        # Get v in contravariant basis
        contra_v = self.G.inv_scale_factors_times_vec(v)

        # Get divergence (basis independent)
        div_v = self.contravariant_div(contra_v)

        return div_v

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
        # Validate input tensors
        dim_geometry = self.dim_geometry
        coords_geometric = self.coords.geometric
        validate_pytorch_tensor(f, (self.coords.n_points,))
        validate_pytorch_tensor(coords_geometric, (self.coords.n_points, dim_geometry))

        # Get contravariant grad of f
        contra_grad_f = self.contravariant_grad(f)

        laplacian_f = self.contravariant_div(contra_grad_f)

        return laplacian_f

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
            Shape (n_points, dim_geometry)

        Returns
        -------
        torch.Tensor
            Vector laplacian of the function and same basis as input vector
            field (n_points, dim_geometry)
        """
        # Validate input tensors
        dim_geometry = self.dim_geometry
        coords_geometric = self.coords.geometric
        validate_pytorch_tensor(v, (self.coords.n_points, dim_geometry))
        validate_pytorch_tensor(coords_geometric, (self.coords.n_points, dim_geometry))

        # Compute Laplacian component-wise
        laplacian_v = torch.stack(
            [self.laplacian(v[:, i]) for i in range(dim_geometry)],
            dim=1,
        )  # shape: (n_points, dim_geometry)

        return laplacian_v
