from __future__ import annotations

from itertools import product


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
