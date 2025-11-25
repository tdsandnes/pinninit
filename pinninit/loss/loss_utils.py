import torch


def is_loss_input_residual(
    f_pred: torch.Tensor | None = None,
    f_target: torch.Tensor | None = None,
    f_residual: torch.Tensor | None = None,
) -> bool:
    """
    Validates that either:
        (1) Both f_pred and f_target are provided
        (2) Only f_residual is provided

    Parameters
    ----------
    f_pred : torch.Tensor, optional
        Predicted function values.
    f_target : torch.Tensor, optional
        Target function values.
    f_residual : torch.Tensor, optional
        Precomputed residual (f_pred - f_target)

    Returns
    -------
    bool
        True if f_residual is given
        False if f_pred and f_target are given

    Raises
    ------
    ValueError
        If combinations of given parameters are not valid.
    """
    if f_residual is not None:
        if f_pred is not None or f_target is not None:
            raise ValueError(
                "If f_residual is provided, f_pred and f_target must NOT be provided."
            )
        return True

    if f_pred is None or f_target is None:
        raise ValueError(
            "f_pred and f_target must both be provided if f_residual is not given."
        )

    return False
