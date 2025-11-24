from collections.abc import Callable
from enum import Enum
from typing import Any, TypeVar

# Generic type for Enum
EnumType = TypeVar("EnumType", bound=Enum)
# Generic type for the callable function
FuncType = TypeVar("FuncType", bound=Callable[..., Any])


def lookup_func_in_registry(
    label: EnumType, registry: dict[EnumType, FuncType]
) -> FuncType:
    """
    Lookup a function in a registry by its Enum label.

    Parameters
    ----------
    label : EnumType
        The Enum member.
    registry : Dict[EnumType, FuncType]
        The dictionary mapping Enum members to functions.

    Returns
    -------
    FuncType
        The function corresponding to the label.

    Raises
    ------
    ValueError
        If the label is not found in the registry.
    """
    if label in registry:
        # Return function
        return registry[label]
    else:
        enum_class_name = type(label).__name__
        raise ValueError(
            f"Unknown label: '{label.value}' in registry of type '{enum_class_name}'. "
            f"Available options are: {[e.value for e in registry]}"
        )


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
