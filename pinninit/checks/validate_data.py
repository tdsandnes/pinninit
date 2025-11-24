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
