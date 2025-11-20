def check_number_within_bounds(
    value: float,
    bounds: tuple[float | None, float | None],
    bounds_inclusive: tuple[bool | None, bool | None],
) -> bool:
    """
    Check if a number is within specified bounds.

    Parameters
    ----------
    value : float
        The number to check.
    bounds : tuple
        A tuple (lower_bound, upper_bound) specifying the bounds.
    bounds_inclusive : tuple
        A tuple (is_lower_inclusive, is_upper_inclusive) specifying if the bounds are
        inclusive.

    Returns
    -------
    bool
        True if the number is within bounds, False otherwise.
    """

    lower_bound, upper_bound = bounds
    is_lower_inclusive, is_upper_inclusive = bounds_inclusive

    # Check lower bound
    if lower_bound is not None:
        if is_lower_inclusive:
            if value < lower_bound:
                return False
        else:
            if value <= lower_bound:
                return False

    # Check upper bound
    if upper_bound is not None:
        if is_upper_inclusive:
            if value > upper_bound:
                return False
        else:
            if value >= upper_bound:
                return False

    return True


def interval_str(
    interval: tuple[float | None, float | None],
    inclusive: tuple[bool | None, bool | None],
) -> str:
    """
    Convert an interval and its inclusivity to a string representation.

    Parameters
    ----------
    interval : tuple
        A tuple (lower_bound, upper_bound) specifying the interval.
    inclusive : tuple
        A tuple (is_lower_inclusive, is_upper_inclusive) specifying if the bounds are
        inclusive.

    Returns
    -------
    str
        String representation of the interval.
    """

    lower_bound, upper_bound = interval
    is_lower_inclusive, is_upper_inclusive = inclusive

    left_bracket = "[" if is_lower_inclusive else "("
    right_bracket = "]" if is_upper_inclusive else ")"

    lower_str = str(lower_bound) if lower_bound is not None else "-∞"
    upper_str = str(upper_bound) if upper_bound is not None else "∞"

    return f"{left_bracket}{lower_str}, {upper_str}{right_bracket}"
