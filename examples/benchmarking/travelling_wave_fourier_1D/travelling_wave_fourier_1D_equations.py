from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from pinninit import Operators


def wavenumbers(c: torch.Tensor) -> torch.Tensor:
    """The wavenumber k of each coefficient (a_1, b_1, a_2, b_2, ...): 1, 1,
    2, 2, ..."""
    k = torch.arange(1, c.shape[1] // 2 + 1, dtype=c.dtype, device=c.device)
    return k.repeat_interleave(2)


def f_eqn(c: torch.Tensor, x: torch.Tensor, operators: Operators) -> torch.Tensor:
    """The field at the observatories' positions x, from the Fourier
    coefficients c: f = sum_k a_k cos(k x) + b_k sin(k x)."""
    kx = wavenumbers(c) * x.unsqueeze(1)
    is_cos = torch.arange(c.shape[1], device=c.device) % 2 == 0
    basis = torch.where(is_cos, kx.cos(), kx.sin())
    return (c * basis).sum(dim=1)


def pde_residual(
    c: torch.Tensor, omega: torch.Tensor, operators: Operators
) -> torch.Tensor:
    """Each coefficient of mode k is a simple harmonic oscillator with
    frequency k omega, so the residual is an array like c."""
    dc_dt = operators.d_dcoord(c, "t")
    d2c_dt2 = operators.d_dcoord(dc_dt, "t")

    return d2c_dt2 + (wavenumbers(c) * omega) ** 2 * c
