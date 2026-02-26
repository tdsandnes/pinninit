from __future__ import annotations

from itertools import product
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from typing import Any


def generate_tensor_component_labels(
    geometric_labels: tuple[str, ...], rank: int
) -> tuple[tuple[str, ...], dict[tuple[int, ...], str]]:
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
    dict[tuple[int, ...], str]
        Mapping from index tuples to component labels.
        e.g.    (0,0): "x_x",
                (0,1): "x_y",
                (1,0): "y_x",
                (1,1): "y_y"
    """
    if rank < 0:
        raise ValueError("rank must be >= 0")
    if rank == 0:
        return ("value",), {(): "value"}

    # product gives all combinations, and .join makes them each a single string
    indices = tuple(product(range(len(geometric_labels)), repeat=rank))
    index_to_name = {idx: "_".join(geometric_labels[i] for i in idx) for idx in indices}
    labels = tuple(index_to_name[idx] for idx in indices)
    return labels, index_to_name


def convert_to_tensor(
    value: Any, device: torch.device, requires_grad: bool
) -> torch.Tensor:
    if isinstance(value, (np.ndarray | list | tuple)):
        # Convert to tensor
        return torch.tensor(value, device=device, requires_grad=requires_grad)
    elif isinstance(value, torch.Tensor):
        # Already tensor, but make sure on correct device and requires grad
        return value.detach().clone().to(device).requires_grad_(requires_grad)
    else:
        raise TypeError(
            f"Component must be a torch.Tensor, np.ndarray, list, or tuple, "
            f"got {type(value)}"
        )
