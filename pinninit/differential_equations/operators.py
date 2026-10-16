from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ..data_structure.coords import Coords


class Operators:
    def __init__(self, coords: Coords):
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
        Compute the derivative of all elements of f wrt a given coordinate

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
        self._check_differentiable(f)
        coord = self.coords.get_coord(coord_label)

        flat = f.reshape(f.shape[0], -1)

        derivatives = [
            torch.autograd.grad(
                flat[:, i],
                coord,
                grad_outputs=torch.ones_like(flat[:, i]),
                create_graph=True,
                allow_unused=True,
            )[0]
            for i in range(flat.shape[1])
        ]

        # Deal with case where f is indep of coord
        if any(d is None for d in derivatives):
            self._check_computed_from_coords(f)
        derivatives = [
            d if d is not None else torch.zeros_like(coord) for d in derivatives
        ]

        return torch.stack(derivatives, dim=1).reshape_as(f)

    def contravariant_grad(self, f: torch.Tensor) -> torch.Tensor:
        """
        Compute the contravariant gradient of a function in a given coordinate system.

        contravariant_grad(f) = g^{ij} * partial_j f

        Parameters
        ----------
        f : torch.Tensor
            Function to be taken gradient of. Shape: (n_points,)

        Returns
        -------
        torch.Tensor
            Contravariant gradient of the function: (n_points, dim_geometric)
        """
        if f.ndim != 1:
            raise ValueError(
                f"contravariant_grad expects a scalar field, got shape {f.shape}"
            )

        self._check_differentiable(f)
        coords = [self.coords.get_coord(label) for label in self.geometric_labels]

        derivatives = torch.autograd.grad(
            outputs=f,
            inputs=coords,
            grad_outputs=torch.ones_like(f),
            create_graph=True,
            allow_unused=True,
        )

        # Deal with case where f is indep of coord
        if any(d is None for d in derivatives):
            self._check_computed_from_coords(f)
        derivatives_list = [
            d if d is not None else torch.zeros_like(coord)
            for d, coord in zip(derivatives, coords, strict=True)
        ]

        derivative_f = torch.stack(derivatives_list, dim=1)

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

        Since v2 is given to the function in the physical bases, this operation
        can be viewed as:

        (v1 dot grad)(v2_i * e_i) = (v1 dot grad)(v2_i)*e_i + v2_i*(v1 dot grad)(e_i)

        To make this easier, we first transform v2 to Cartesian (so that the
        (v1 dot grad)(e_i) terms are 0) and then transform back.

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
        # Transform v2 to Cartesian
        geometric_coords = {
            label: self.coords.get_coord(label) for label in self.geometric_labels
        }
        v2_cartesian = self.coord_system.transform_geometric_to_cartesian(
            v2, geometric_coords
        )

        # Apply operation
        out = torch.stack(
            [
                (v1 * self.grad(v2_cartesian[:, i])).sum(dim=1)
                for i in range(v2_cartesian.shape[1])
            ],
            dim=1,
        )

        # Transform back
        return self.coord_system.transform_cartesian_to_geometric(out, geometric_coords)

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

        div_terms = [
            self.d_dcoord(inner_term[:, i], coord_label)
            for i, coord_label in enumerate(self.geometric_labels)
        ]

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
        Compute the laplacian of a vector field in a given coordinate system.

        Since v is given to the function in the physical bases, this operation
        can be viewed as:

        lap(v2_i * e_i) = lap(v2_i) * e_i + v2_i * lap(e_i)

        To make this easier, we first transform v to Cartesian (so that the
        lap(e_i) terms are 0) and then transform back.

        Parameters
        ----------
        v : torch.Tensor
            The vector field.
            Shape (n_points, dim_geometric)

        Returns
        -------
        torch.Tensor
            Laplacian of the vector field and same basis as input vector
            field (n_points, dim_geometric)
        """
        # Transform v to Cartesian
        geometric_coords = {
            label: self.coords.get_coord(label) for label in self.geometric_labels
        }
        v_cartesian = self.coord_system.transform_geometric_to_cartesian(
            v, geometric_coords
        )

        # Apply operation
        lap = torch.stack(
            [self.laplacian(v_cartesian[:, i]) for i in range(v_cartesian.shape[1])],
            dim=1,
        )

        # Transform back
        return self.coord_system.transform_cartesian_to_geometric(lap, geometric_coords)

    def _check_differentiable(self, f: torch.Tensor) -> None:
        if not f.requires_grad:
            raise ValueError(
                "Can only differentiate tensors computed from the operators' coords."
            )

    def _check_computed_from_coords(self, f: torch.Tensor) -> None:
        """Raise if f has no autograd path to any of these coords."""
        coords = [self.coords.get_coord(label) for label in self.coords.coord_labels]
        grads = torch.autograd.grad(
            f.sum(), coords, retain_graph=True, allow_unused=True
        )
        if all(grad is None for grad in grads):
            raise ValueError(
                "Can only differentiate tensors computed from the operators' coords."
            )
