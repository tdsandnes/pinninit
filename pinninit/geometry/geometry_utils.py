from __future__ import annotations


def select_coord_system_axes(
    axes_labels: tuple[str, ...],
    axes_bounds: tuple[tuple[float, float], ...],
    input_labels: tuple[str, ...],
) -> tuple[
    tuple[str, ...],
    tuple[tuple[float, float], ...],
    tuple[int, ...],
]:
    """
    Select a subset of axis labels and their corresponding bounds input labels.

    Parameters
    ----------
    axes_labels : tuple of str
        All axis labels in the coordinate system.
    axes_bounds : tuple of tuples containing floats
        Bounds associated with each axis label.
    input_labels : tuple of str
        Labels to select.

    Returns
    ----------
    tuple
        A tuple containing:
        - axes_labels_selected : tuple of str
            Selected labels in the order they appear in axes_labels.
        - axes_bounds_selected :  tuple of tuples containing floats
            Bounds corresponding to axes_labels_selected.
        - indices_selected : tuple of int
            The indices of elements in axes_labels also in input_labels.

    Raises
    ----------
    ValueError
        If input lengths differ, input_labels contains duplicates, or invalid label.
    """
    # Validate lengths
    if len(axes_labels) != len(axes_bounds):
        raise ValueError("axes_bounds and axes_labels must have the same length.")

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

    return (
        axes_labels_selected,
        axes_bounds_selected,
        indices_selected,
    )
