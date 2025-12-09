from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ..checks.validate_data import validate_list_of_tensors, validate_pytorch_tensor

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
        dim_geometry = self.coord_system.dim_geometry
        coords_geometric = self.coords_geometric

        # Validate input tensors
        n_points = f.shape[0]
        validate_pytorch_tensor(f, (n_points,))
        validate_list_of_tensors(coords_geometric, dim_geometry, (n_points,))

        # Tensor of metric tensors for each point
        G = self.coord_system.metric_tensor(
            n_points, coords_geometric
        )  # Shape: (n_points, dim_geometry, dim_geometry)
        validate_pytorch_tensor(G, (n_points, dim_geometry, dim_geometry))

        # Inverse metric tensor
        G_inv = torch.linalg.inv(G)  # Shape: (n_points, dim_geometry, dim_geometry)
        validate_pytorch_tensor(G_inv, (n_points, dim_geometry, dim_geometry))

        derivative_f: list[torch.Tensor] = []
        for coord in coords_geometric:
            # Calculate derivative wrt a coord using autograd
            derivative_f_coord = torch.autograd.grad(
                f,
                coord,
                grad_outputs=torch.ones_like(f),
                create_graph=True,
                retain_graph=True,
            )[0]

            # Set derivatives to zero in None (Case where f is indep of coord)
            if derivative_f_coord is None:
                derivative_f_coord = torch.zeros_like(coord)

            # Append to list
            derivative_f.append(derivative_f_coord)

        derivative_f_tensor = torch.stack(derivative_f, dim=1)
        validate_pytorch_tensor(derivative_f_tensor, (n_points, dim_geometry))

        # Batch matrix multiplication:
        # (n_points, dim_geometry, dim_geometry) @ (n_points, dim_geometry, 1) ->
        # (n_points, dim_geometry, 1)
        # Squeeze the result to get the final (n_points, dim_geometry) vector
        contra_grad_f = torch.bmm(G_inv, derivative_f_tensor.unsqueeze(2)).squeeze(2)
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
        coords_geometric = self.coords_geometric

        # Get scale factors (assumes the coordinate system is orthogonal)
        n_points = f.shape[0]
        scale_factors = self.coord_system.get_scale_factors(coords_geometric, n_points)

        # Get contravariant grad of f
        contra_grad_f = self.contravariant_grad(f)

        return scale_factors * contra_grad_f

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
        dim_geometry = self.coord_system.dim_geometry
        coords_geometric = self.coords_geometric
        n_points = v1.shape[0]

        # Validate inputs
        validate_pytorch_tensor(v1, (n_points, dim_geometry))
        validate_pytorch_tensor(v2, (n_points, dim_geometry))
        validate_list_of_tensors(coords_geometric, dim_geometry, (n_points,))

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
        dim_geometry = self.coord_system.dim_geometry
        coords_geometric = self.coords_geometric
        n_points = contra_v.shape[0]

        # Validate inputs
        validate_pytorch_tensor(contra_v, (n_points, dim_geometry))
        validate_list_of_tensors(coords_geometric, dim_geometry, (n_points,))

        # Tensor of metric tensors for each point
        G = self.coord_system.metric_tensor(
            n_points, coords_geometric
        )  # Shape: (n_points, dim_geometry, dim_geometry)
        validate_pytorch_tensor(G, (n_points, dim_geometry, dim_geometry))

        # Metric determinant
        det_G = torch.det(G)  # Shape: (n_points,)
        sqrt_det_G = torch.sqrt(torch.abs(det_G))  # Shape: (n_points,)

        # Initialise div_v
        contra_div_v = torch.zeros(n_points, device=self.coord_system.device)

        # Sum of partial derivatives
        for i, coord in enumerate(coords_geometric):
            inner_term = sqrt_det_G * contra_v[:, i]  # Shape (n_points,)

            # Differentiate the i-th component w.r.t the i-th coordinate
            derivative_inner_coord = torch.autograd.grad(
                inner_term,
                coord,
                grad_outputs=torch.ones_like(inner_term),
                create_graph=True,
                retain_graph=True,
            )[0]

            # Set derivatives to zero in None (Case where f is indep of coord)
            if derivative_inner_coord is None:
                derivative_inner_coord = torch.zeros_like(coord)

            # Add to cumulative sum of partial derivatives
            contra_div_v += derivative_inner_coord

        # Divide by sqrt of det g to get divergence
        contra_div_v /= sqrt_det_G
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
            The vector field, given in contravariant basis.
            Shape (n_points, dim_geometry)

        Returns
        -------
        torch.Tensor
            The divergence of the vector field v.
            Shape (n_points, dim_geometry)
        """
        # Coordinate system dimensions
        dim_geometry = self.coord_system.dim_geometry
        coords_geometric = self.coords_geometric
        n_points = v.shape[0]

        # Validate inputs
        validate_pytorch_tensor(v, (n_points, dim_geometry))
        validate_list_of_tensors(coords_geometric, dim_geometry, (n_points,))

        # Get scale factors (assumes the coordinate system is orthogonal)
        scale_factors = self.coord_system.get_scale_factors(coords_geometric, n_points)

        # Get v in contravariant basis
        contra_v = v / scale_factors

        # Get divergence (basis intep.)
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
        dim_geometry = self.coord_system.dim_geometry
        coords_geometric = self.coords_geometric
        n_points = f.shape[0]
        validate_pytorch_tensor(f, (n_points,))
        validate_list_of_tensors(coords_geometric, dim_geometry, (n_points,))

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
            field (n_points,)
        """
        # Validate input tensors
        dim_geometry = self.coord_system.dim_geometry
        coords_geometric = self.coords_geometric
        n_points = v.shape[0]
        validate_pytorch_tensor(v, (n_points, dim_geometry))
        validate_list_of_tensors(coords_geometric, dim_geometry, (n_points,))

        # Compute Laplacian component-wise
        laplacian_v = torch.stack(
            [self.laplacian(v[:, i]) for i in range(dim_geometry)],
            dim=1,
        )  # shape: (n_points, dim_geometry)

        return laplacian_v
