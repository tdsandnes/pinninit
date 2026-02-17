from __future__ import annotations

import numpy as np
import torch


def validate_numpy_array(arr: np.ndarray, shape: tuple[int, ...]) -> None:
    """
    Validate that the input is a NumPy array with the expected shape.

    Parameters
    ----------
    arr : ndarray
        array to validate.
    shape : tuple
        Expected shape of the array.

    Raises
    ------
    TypeError
        If arr is not a NumPy array.
    ValueError
        If arr does not have expected shape.
    """

    # Check that arr is a NumPy array
    if not isinstance(arr, np.ndarray):
        raise TypeError(
            "[validate_numpy_array] Expected input to be a NumPy array, "
            f"got {type(arr).__name__}"
        )

    # Check that arr has correct shape
    if arr.shape != shape:
        raise ValueError(
            f"[validate_numpy_array] Expected input to have shape {shape}, "
            f"got {arr.shape}"
        )


def validate_pytorch_tensor(tensor: torch.Tensor, shape: tuple[int, ...]) -> None:
    """
    Validate that the input is a PyTorch tensor with the expected shape.

    Parameters
    ----------
    tensor : torch.Tensor
        Tensor to validate.
    shape : tuple
        Expected shape of the tensor.

    Raises
    ------
    TypeError
        If tensor is not a PyTorch tensor.
    ValueError
        If tensor does not have expected shape.
    """

    # Check that tensor is a PyTorch tensor
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(
            "[validate_pytorch_tensor] Expected input to be a PyTorch tensor, "
            f"got {type(tensor).__name__}"
        )

    # Check that tensor has correct shape
    if tensor.shape != shape:
        raise ValueError(
            f"[validate_pytorch_tensor] Expected input to have shape {shape}, "
            f"got {tensor.shape}"
        )


def validate_list_of_tensors(
    vector: list[torch.Tensor], list_length: int, tensor_shape: tuple[int, ...]
) -> None:
    """
    Validate that a set of provided vector (list[torch.Tensor] in the form of the
    geometric coordinates) is in the right form.

    Parameters
    ----------
    vector : list[torch.Tensor]
        Vector to validate. List length: dim_geometry, tensor lengths: n_points
    list_length : int
        Length of list of tensors
    tensor_shape : tuple
        Expected shape of all tensors in list.

    Raises
    ------
    ValueError
        If vector has incorrect form.
    """

    # Validate coords_geometry tensor type and shape
    if len(vector) != list_length:
        raise ValueError(
            f"[validate_list_of_tensors] Expected vector to have length {list_length}, "
            f"got {len(vector)}"
        )

    for i in range(list_length):
        validate_pytorch_tensor(vector[i], tensor_shape)
