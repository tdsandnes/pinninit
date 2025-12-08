from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from pinninit.differential_equations.operators import Operators


def velocity_from_psi_2d(
    psi: torch.Tensor, operators: Operators
) -> tuple[torch.Tensor, torch.Tensor]:
    """Calculate velocity from stream function in 2D"""

    grad_psi = operators.grad(psi)

    return grad_psi[1], -grad_psi[0]


def NS_residual_2d(
    u: torch.Tensor,
    v: torch.Tensor,
    p: torch.Tensor,
    l1: torch.Tensor,
    l2: torch.Tensor,
    operators: Operators,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Calculate residual if Navier-Stokes Eqn. in 2D with two learnable params"""

    # Apply operators
    du_dt = operators.df_dcoord(u, "t")
    dv_dt = operators.df_dcoord(v, "t")
    grad_u = operators.grad(u)
    grad_v = operators.grad(v)
    grad_p = operators.grad(p)
    laplace_u = operators.laplacian(u)
    laplace_v = operators.laplacian(v)

    # Calculate residual of NS equation in 2D
    residual_u = (
        du_dt + l1 * (u * grad_u[0] + v * grad_u[1]) + grad_p[0] - l2 * laplace_u
    )
    residual_v = (
        dv_dt + l1 * (u * grad_v[0] + v * grad_v[1]) + grad_p[1] - l2 * laplace_v
    )

    return residual_u, residual_v
