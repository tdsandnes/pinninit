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

    # Left bracket
    if lower_bound is None:
        left_bracket = "("
        lower_str = "-∞"
    else:
        left_bracket = "[" if is_lower_inclusive else "("
        lower_str = str(lower_bound)

    # Right bracket
    if upper_bound is None:
        right_bracket = ")"
        upper_str = "∞"
    else:
        right_bracket = "]" if is_upper_inclusive else ")"
        upper_str = str(upper_bound)

    return f"{left_bracket}{lower_str}, {upper_str}{right_bracket}"
