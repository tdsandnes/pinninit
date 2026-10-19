from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Import pinninit
    import sys
    from pathlib import Path

    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from pinninit import Operators


def simple_harmonic_oscillator_1D_pde_residual(
    f: torch.Tensor, omega: torch.Tensor, operators: Operators
) -> torch.Tensor:
    df_dx = operators.d_dcoord(f, "x")
    d2f_dx2 = operators.d_dcoord(df_dx, "x")

    return d2f_dx2 + omega**2 * f
