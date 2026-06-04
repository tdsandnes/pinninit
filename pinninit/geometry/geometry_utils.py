from __future__ import annotations

import torch


def trig_bounds(
    function: str,
    theta_lb: torch.Tensor,
    theta_ub: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Exact bounds of sin(theta) or cos(theta) on an interval.
    """
    pi = torch.tensor(torch.pi)
    if function == "cos":
        candidates = [
            torch.cos(theta_lb),
            torch.cos(theta_ub),
        ]

        k_start = int(torch.ceil(theta_lb / pi))
        k_end = int(torch.floor(theta_ub / pi))

        for k in range(k_start, k_end + 1):
            candidates.append(torch.cos(k * pi))

    elif function == "sin":
        candidates = [
            torch.sin(theta_lb),
            torch.sin(theta_ub),
        ]

        k_start = int(torch.ceil((theta_lb - pi / 2) / pi))
        k_end = int(torch.floor((theta_ub - pi / 2) / pi))

        for k in range(k_start, k_end + 1):
            candidates.append(torch.sin(pi / 2 + k * pi))

    else:
        raise ValueError(f"Unknown trig function '{function}'")

    return torch.min(torch.stack(candidates)), torch.max(torch.stack(candidates))
