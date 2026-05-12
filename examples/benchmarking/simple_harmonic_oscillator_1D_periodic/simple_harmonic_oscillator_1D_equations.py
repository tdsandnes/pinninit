from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Import pinninit
    import sys
    from pathlib import Path

    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from pinninit.differential_equations.operators import Operators


def simple_harmonic_oscillator_1D_pde_residual(
    y: torch.Tensor, operators: Operators
) -> torch.Tensor:
    dy_dx = operators.d_dcoord(y, "x")
    d2y_dx2 = operators.d_dcoord(dy_dx, "x")

    return d2y_dx2 + y
