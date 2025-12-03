from __future__ import annotations

import torch


def select_coord_system_axes(
    axes_labels: tuple[str, ...],
    axes_bounds: tuple[tuple[float | None, float | None], ...],
    axes_bounds_inclusive: tuple[tuple[bool | None, bool | None], ...],
    input_labels: tuple[str, ...],
) -> tuple[
    tuple[str, ...],
    tuple[tuple[float | None, float | None], ...],
    tuple[tuple[bool | None, bool | None], ...],
    tuple[int, ...],
]:
    """
    Select a subset of axis labels and their corresponding bounds input labels.

    Parameters
    ----------
    axes_labels : tuple of str
        All axis labels in the coordinate system.
    axes_bounds : tuple of float or None
        Bounds associated with each axis label.
    axes_bounds_inclusive : tuple of float or None
        Inclusive conditions associated with each axis label.
    input_labels : tuple of str
        Labels to select.

    Returns
    ----------
    tuple
        A tuple containing:
        - axes_labels_selected : tuple of str
            Selected labels in the order they appear in axes_labels.
        - axes_bounds_selected : tuple of tuples of float or None
            Bounds corresponding to axes_labels_selected.
        - axes_bounds_inclusive_selected : tuple of tuples of bool or None
            Inclusive conditions corresponding to axes_labels_selected.
        - indices_selected : tuple of int
            The indices of elements in axes_labels also in input_labels.

    Raises
    ----------
    ValueError
        If input lengths differ, input_labels contains duplicates, or invalid label.
    """
    # Validate lengths
    if len(axes_labels) != len(axes_bounds) or len(axes_labels) != len(
        axes_bounds_inclusive
    ):
        raise ValueError(
            "axes_bounds, axes_labels, and axes_bounds_inclusive must all "
            "have the same length."
        )

    # Check for duplicates
    if len(input_labels) != len(set(input_labels)):
        raise ValueError(f"input_labels {input_labels} contains duplicates.")

    # Validate presence of all input_labels in axes_labels
    label_set = set(axes_labels)
    missing = [lbl for lbl in input_labels if lbl not in label_set]
    if missing:
        raise ValueError(f"These input labels are not in axes_labels: {missing}")

    # Select labels and values in the order they appear in axes_labels
    indices_selected = tuple(
        i for i, lbl in enumerate(axes_labels) if lbl in input_labels
    )
    axes_labels_selected = tuple(axes_labels[i] for i in indices_selected)
    axes_bounds_selected = tuple(axes_bounds[i] for i in indices_selected)
    axes_bounds_inclusive_selected = tuple(
        axes_bounds_inclusive[i] for i in indices_selected
    )

    return (
        axes_labels_selected,
        axes_bounds_selected,
        axes_bounds_inclusive_selected,
        indices_selected,
    )


def get_required_or_fixed_coord(
    axis_label: str,
    coords_geometry: list[torch.Tensor],
    axes_labels_selected: tuple[str, ...],
    fixed_axes: tuple[str, ...] | None,
    fixed_axes_values: tuple[float, ...] | None,
) -> torch.Tensor:
    if axis_label in axes_labels_selected:
        idx = axes_labels_selected.index(axis_label)
        return coords_geometry[idx]
    elif (
        fixed_axes is not None
        and fixed_axes_values is not None
        and axis_label in fixed_axes
    ):
        idx = fixed_axes.index(axis_label)
        return torch.tensor(fixed_axes_values[idx]).float()
    else:
        raise ValueError(f"Missing required axis: {axis_label}")
