from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Import pinninit
    import sys
    from pathlib import Path

    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from pinninit import Operators


def zp_eqn(
    b: torch.Tensor,
    u: torch.Tensor,
    Re: torch.Tensor,
    Rm: torch.Tensor,
    Ma: torch.Tensor,
    operators: Operators,
) -> torch.Tensor:
    nu_p = Ma * 0.5 * (1 / Re + 1 / Rm)
    nu_m = Ma * 0.5 * (1 / Re - 1 / Rm)
    zp = u + b
    zm = u - b

    dzp_dt = operators.d_dcoord(zp, "t")
    dzp_dx = operators.d_dcoord(zp, "x")
    d2zp_dx2 = operators.d_dcoord(dzp_dx, "x")
    dzm_dx = operators.d_dcoord(zm, "x")
    d2zm_dx2 = operators.d_dcoord(dzm_dx, "x")

    return dzp_dt - dzp_dx + Ma * zm * dzp_dx - nu_p * d2zp_dx2 - nu_m * d2zm_dx2


def zm_eqn(
    b: torch.Tensor,
    u: torch.Tensor,
    Re: torch.Tensor,
    Rm: torch.Tensor,
    Ma: torch.Tensor,
    operators: Operators,
) -> torch.Tensor:
    nu_p = Ma * 0.5 * (1 / Re + 1 / Rm)
    nu_m = Ma * 0.5 * (1 / Re - 1 / Rm)
    zp = u + b
    zm = u - b

    dzm_dt = operators.d_dcoord(zm, "t")
    dzm_dx = operators.d_dcoord(zm, "x")
    d2zm_dx2 = operators.d_dcoord(dzm_dx, "x")
    dzp_dx = operators.d_dcoord(zp, "x")
    d2zp_dx2 = operators.d_dcoord(dzp_dx, "x")

    return dzm_dt + dzm_dx + Ma * zp * dzm_dx - nu_p * d2zm_dx2 - nu_m * d2zp_dx2


def u_eqn(nn_out: torch.Tensor, x: torch.Tensor, operators: Operators) -> torch.Tensor:
    """u, which is zero at x = 1 by construction."""
    return (1.0 - x) * nn_out


def dbdx_bc_eqn(b: torch.Tensor, operators: Operators) -> torch.Tensor:
    return operators.d_dcoord(b, "x")
