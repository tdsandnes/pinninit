from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Import pinninit
    import sys
    from pathlib import Path

    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from pinninit.differential_equations.operators import Operators


def momentum_eqn(
    b: torch.Tensor,
    u: torch.Tensor,
    S: torch.Tensor,
    Pm: torch.Tensor,
    operators: Operators,
) -> torch.Tensor:
    du_dt = operators.d_dcoord(u, "t")
    du_dx = operators.d_dcoord(u, "x")
    d2u_dx2 = operators.d_dcoord(du_dx, "x")
    db_dx = operators.d_dcoord(b, "x")

    return du_dt + S * u * du_dx - S * b * db_dx - Pm * d2u_dx2


def induction_eqn(
    b: torch.Tensor, u: torch.Tensor, S: torch.Tensor, operators: Operators
) -> torch.Tensor:
    db_dt = operators.d_dcoord(b, "t")
    db_dx = operators.d_dcoord(b, "x")
    d2b_dx2 = operators.d_dcoord(db_dx, "x")
    du_dx = operators.d_dcoord(u, "x")

    return db_dt + S * u * db_dx - S * b * du_dx - d2b_dx2
