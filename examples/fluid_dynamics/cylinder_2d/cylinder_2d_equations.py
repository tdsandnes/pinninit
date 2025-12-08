from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from pinninit.differential_equations.operators import Operators


def velocity_from_psi_2d(psi: torch.Tensor, operators: Operators) -> torch.Tensor:
    """Calculate velocity from stream function in 2D"""

    grad_psi = operators.grad(psi)
    ### temp: ideally only want to use operators and not torch.
    ### curl might be tricky do define when z is not in coord system
    v = torch.stack([grad_psi[:, 1], -grad_psi[:, 0]], dim=1)

    return v


def NS_residual_2d(
    v: torch.Tensor,
    p: torch.Tensor,
    l1: torch.Tensor,
    l2: torch.Tensor,
    operators: Operators,
) -> torch.Tensor:
    """Calculate residual if Navier-Stokes Eqn. with two learnable params"""

    # Apply operators
    dv_dt = operators.d_dcoord(v, "t")
    v_dot_grad_v = operators.v_dot_grad(v, v)
    grad_p = operators.grad(p)
    vector_laplace_v = operators.vector_laplacian(v)

    # Calculate residual of NS equation in 2D
    residuals = dv_dt + l1 * v_dot_grad_v + grad_p - l2 * vector_laplace_v

    return residuals
