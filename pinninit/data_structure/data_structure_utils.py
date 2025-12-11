from __future__ import annotations

from itertools import product
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from typing import Any


def generate_tensor_component_labels(
    geometric_labels: tuple[str, ...], rank: int
) -> tuple[str, ...]:
    """
    Generate labels for each component of a rank-n tensor field.

    Parameters
    ----------
    geometric_labels : list[str]
        List of geometric labels.
    rank : int
        Rank of tensor

    Returns
    -------
    tuple[str]
        Tuple for tensor components.
    """
    if rank == 0:
        # Scalar field
        return ("value",)
    elif rank > 0:
        # product gives all combinations, and .join makes them each a single str
        return tuple("".join(p) for p in product(geometric_labels, repeat=rank))
    else:
        raise ValueError(f"Tensor rank must be greater than or equal to 0, got {rank}")


def convert_to_tensor(
    value: Any, device: torch.device, requires_grad: bool
) -> torch.Tensor:
    if isinstance(value, (np.ndarray | list | tuple)):
        # Convert to tensor
        return torch.tensor(
            value, dtype=torch.float32, device=device, requires_grad=requires_grad
        )
    elif isinstance(value, torch.Tensor):
        # Already tensor, but make sure on correct device and requires grad
        return value.to(device).float().requires_grad_(requires_grad)
    else:
        raise TypeError(
            f"Component must be a torch.Tensor, np.ndarray, list, or tuple, "
            f"got {type(value)}"
        )
